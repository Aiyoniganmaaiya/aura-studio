import { useState, useRef, useCallback, useEffect } from "react";
import { Link } from "react-router-dom";
import { useAppStore, type GenerationMode, type GenerationParams } from "../stores/appStore";
import { generateImageWS } from "../api/backend";
import { saveGeneration } from "../lib/historyDB";
import { detectFamily, capabilitiesFor } from "../lib/modelUtils";
import { PromptInput } from "../components/PromptInput";
import { ParamPanel } from "../components/ParamPanel";
import { ImageDisplay } from "../components/ImageDisplay";
import { ProgressBar } from "../components/ProgressBar";
import { ModeTabs } from "../components/ModeTabs";
import { ImageDropzone } from "../components/ImageDropzone";
import { InpaintCanvas } from "../components/InpaintCanvas";
import { ControlPanel } from "../components/ControlPanel";
import {
  SparkleIcon,
  CheckIcon,
  StopIcon,
  CubeIcon,
  ZapIcon,
  AlertCircleIcon,
} from "../components/icons";

/** Snap an uploaded image's size onto the generation params, aligned to the
 * family's latent grid (Z-Image requires ÷16, the rest ÷8) with sane bounds. */
function fitParamsToSize(
  w: number,
  h: number,
  mult: number
): { width: number; height: number } {
  const snap = (v: number) =>
    Math.max(256, Math.min(2048, Math.round(v / mult) * mult));
  return { width: snap(w), height: snap(h) };
}

export default function GeneratePage() {
  const {
    params,
    initImage,
    maskImage,
    controlImage,
    setInitImage,
    setMaskImage,
    setControlImage,
    isGenerating,
    setIsGenerating,
    progress,
    setProgress,
    addResult,
    currentResult,
    backendConnected,
    engineLoaded,
    updateParams,
    pushToast,
  } = useAppStore();

  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const abortRef = useRef<(() => void) | null>(null);
  const cancelBatchRef = useRef(false);

  const family = detectFamily(params.modelId || "");
  const caps = capabilitiesFor(family);
  // Z-Image pipelines reject sizes that are not multiples of 16.
  const sizeMult = family === "z-image" ? 16 : 8;
  // A persisted mode the current model can't run falls back to txt2img.
  const requestedMode = params.mode ?? "txt2img";
  const mode: GenerationMode =
    requestedMode === "txt2img" || caps[requestedMode] ? requestedMode : "txt2img";

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current();
    };
  }, []);

  const needsInitImage = mode === "img2img" || mode === "inpaint";
  const missingInput =
    (needsInitImage && !initImage) ||
    (mode === "inpaint" && !maskImage) ||
    (mode === "controlnet" && !controlImage);

  /** Run ONE generation over the WS and resolve when it finishes.
   * Resolves false on cancel/error, true on a completed image. */
  const runOne = (
    snapshotParams: GenerationParams,
    seedOverride: number | null,
    batchIndex: number,
    batchTotal: number
  ): Promise<boolean> =>
    new Promise((resolve) => {
      let done = false;
      const finish = (ok: boolean) => {
        if (!done) {
          done = true;
          resolve(ok);
        }
      };
      const effParams = { ...snapshotParams, ...(seedOverride !== null ? { seed: seedOverride } : {}) };
      const labelPrefix = batchTotal > 1 ? `Image ${batchIndex + 1}/${batchTotal} — ` : "";

      const abort = generateImageWS(
        effParams,
        { initImage, maskImage, controlImage },
        {
          onProgress: (step, total, frameStatus) => {
            const overallStep = step + batchIndex * total;
            const overallTotal = total * batchTotal;
            setProgress({
              step: overallStep,
              total: overallTotal,
              progressPct: Math.round((overallStep / overallTotal) * 100),
              status: "generating",
            });
            setStatusText(
              frameStatus === "starting"
                ? `${labelPrefix}Preparing engine — moving model weights to GPU…`
                : `${labelPrefix}Step ${step} / ${total}`
            );
          },
          onComplete: (result) => {
            const entry = {
              id: `gen-${Date.now()}-${batchIndex}`,
              imageBase64: result.image_base64,
              prompt: effParams.prompt,
              negativePrompt: effParams.negativePrompt,
              params: effParams,
              seed: result.seed,
              elapsed: result.elapsed,
              timestamp: Date.now(),
            };
            addResult(entry);
            // Persist to IndexedDB (best-effort; history survives restarts).
            saveGeneration(entry).catch((e) => console.error("Failed to save history:", e));
            setStatusText(
              `${labelPrefix}Done — ${result.elapsed.toFixed(1)}s`
            );
            finish(true);
          },
          onCancelled: () => {
            setStatusText("Cancelled");
            finish(false);
          },
          onError: (errMsg) => {
            setError(errMsg);
            pushToast("error", errMsg);
            setStatusText("");
            finish(false);
          },
        }
      );
      // The header Cancel button interrupts the in-flight image AND the loop.
      abortRef.current = () => {
        cancelBatchRef.current = true;
        abort();
      };
    });

  const handleGenerate = useCallback(async () => {
    if (!params.prompt.trim() || isGenerating) return;
    if (!backendConnected) {
      setError("Backend engine is not connected. Start it with run.bat or `npm run backend:dev`.");
      return;
    }
    if (missingInput) {
      pushToast(
        "error",
        mode === "inpaint"
          ? !initImage
            ? "Upload a source image first"
            : "Paint an area to redraw first"
          : "Upload an image for this mode first"
      );
      return;
    }

    const batch = Math.max(1, Math.min(6, params.batchCount ?? 1));
    const baseSeed = params.seed;
    // Freeze params for the whole batch so mid-run edits can't mix states.
    const snapshot = { ...params, mode };

    setError(null);
    setIsGenerating(true);
    cancelBatchRef.current = false;
    setProgress({ step: 0, total: params.steps * batch, progressPct: 0, status: "starting" });
    setStatusText(batch > 1 ? `Image 1/${batch} — connecting...` : "Connecting to engine...");

    for (let i = 0; i < batch; i++) {
      if (cancelBatchRef.current) break;
      // Fixed seed + batch > 1 walks seed, seed+1, … so each image differs
      // deterministically; random stays random per image.
      const seedForBatch = baseSeed !== null && batch > 1 ? baseSeed + i : i === 0 ? baseSeed : null;
      const ok = await runOne(snapshot, seedForBatch, i, batch);
      if (!ok) break;
    }

    setProgress(null);
    setIsGenerating(false);
    abortRef.current = null;
  }, [
    params, mode, missingInput, initImage, maskImage, controlImage,
    isGenerating, backendConnected, addResult, setIsGenerating, setProgress, pushToast,
  ]);

  // Allow abort — asks the backend to interrupt the pipeline for real.
  const handleAbort = () => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setIsGenerating(false);
    setProgress(null);
    setStatusText("Cancelling...");
  };

  // Ctrl+Enter anywhere on the page starts a generation.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleGenerate();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleGenerate]);

  return (
    <div className="h-full flex flex-col animate-fadeIn">
      {/* Header */}
      <header className="flex items-center justify-between px-6 h-14 border-b border-surface-800/30 flex-shrink-0">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold text-white tracking-tight">
            Generate
          </h2>
          <ModeTabs
            value={mode}
            family={family}
            onChange={(m) => updateParams({ mode: m })}
          />
          {isGenerating && progress && (
            <div className="w-40">
              <ProgressBar
                current={progress.step}
                total={progress.total}
                showLabel={false}
                color="bg-aura-500"
              />
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden lg:flex items-center text-[10px] text-surface-600 border border-surface-800/60 rounded-md px-1.5 py-0.5 font-mono">
            Ctrl+↵
          </span>
          {!engineLoaded && backendConnected && (
            <Link
              to="/models"
              className="flex items-center gap-1.5 text-xs text-amber-300 bg-amber-400/10 border border-amber-400/20 px-2.5 py-1.5 rounded-lg transition-colors hover:bg-amber-400/20"
            >
              <CubeIcon className="w-3.5 h-3.5" />
              No model loaded — pick one in Models
            </Link>
          )}
          {isGenerating ? (
            <button onClick={handleAbort} className="btn-secondary !px-4">
              <StopIcon className="w-4 h-4" />
              Cancel
            </button>
          ) : (
            <button
              onClick={handleGenerate}
              disabled={!params.prompt.trim() || !backendConnected || missingInput}
              title={missingInput ? "Add the required input image for this mode" : undefined}
              className="btn-primary !px-4"
            >
              <SparkleIcon className="w-4 h-4" />
              Generate
            </button>
          )}
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left - Main image area */}
        <div className="flex-1 flex flex-col items-center justify-center p-6 overflow-y-auto">
          {isGenerating && progress ? (
            <div className="flex flex-col items-center gap-6 max-w-md w-full animate-fadeIn">
              {/* Live progress ring */}
              <div className="relative w-40 h-40">
                <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="6"
                    className="text-surface-800"
                  />
                  <circle
                    cx="50" cy="50" r="42"
                    fill="none"
                    strokeWidth="6"
                    strokeLinecap="round"
                    className="text-aura-500 drop-shadow-[0_0_6px_rgba(37,85,255,0.5)]"
                    strokeDasharray={`${2 * Math.PI * 42}`}
                    strokeDashoffset={`${2 * Math.PI * 42 * (1 - progress.progressPct / 100)}`}
                    style={{ transition: "stroke-dashoffset 0.3s ease-out" }}
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-semibold text-white font-mono">
                    {progress.progressPct}%
                  </span>
                </div>
              </div>
              <div className="text-center">
                <p className="text-sm text-surface-300">{statusText}</p>
                <p className="text-xs text-surface-500 mt-1">
                  {params.steps} steps · {params.width}×{params.height}
                </p>
              </div>
            </div>
          ) : currentResult ? (
            <div className="animate-popIn w-full flex justify-center">
              <ImageDisplay
                imageBase64={currentResult.imageBase64}
                prompt={currentResult.prompt}
                seed={currentResult.seed}
                elapsed={currentResult.elapsed}
                params={currentResult.params}
              />
            </div>
          ) : !backendConnected ? (
            <EngineOffGuide />
          ) : (
            <div className="text-center max-w-md animate-fadeIn">
              <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-aura-500/20 to-purple-500/20 border border-aura-500/10 flex items-center justify-center shadow-lg shadow-aura-900/30">
                <SparkleIcon className="w-8 h-8 text-surface-400" />
              </div>
              <h3 className="text-lg font-medium text-surface-200 mb-2">
                Ready to create
              </h3>
              <p className="text-sm text-surface-500 leading-relaxed">
                Enter a prompt on the right, adjust your settings, and click
                Generate to bring your imagination to life.
              </p>
              <div className="mt-6 flex justify-center gap-6 text-xs text-surface-600">
                <div className="flex items-center gap-1.5">
                  <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                  FLUX / SDXL / SD 1.5
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                  Local &amp; Private
                </div>
                <div className="flex items-center gap-1.5">
                  <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                  {engineLoaded ? "GPU Ready" : "GPU Accelerated"}
                </div>
              </div>
            </div>
          )}

          {/* Error display */}
          {error && (
            <div className="mt-4 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-400 max-w-lg animate-slideDown">
              {error}
            </div>
          )}
        </div>

        {/* Right - Mode inputs + parameters */}
        <div className="w-80 flex-shrink-0 border-l border-surface-800/30 overflow-y-auto bg-surface-900/20">
          <div className="p-4 space-y-4">
            {mode === "img2img" && (
              <ImageDropzone
                value={initImage}
                onChange={setInitImage}
                onSize={(w, h) => updateParams(fitParamsToSize(w, h, sizeMult))}
                label="Source image"
                hint="The prompt describes the NEW image"
              />
            )}
            {mode === "inpaint" && (
              <>
                <ImageDropzone
                  value={initImage}
                  onChange={(b64) => {
                    setInitImage(b64);
                    if (!b64) setMaskImage(null);
                  }}
                  onSize={(w, h) => updateParams(fitParamsToSize(w, h, sizeMult))}
                  label="Source image"
                />
                {initImage ? (
                  <InpaintCanvas imageB64={initImage} onMaskChange={setMaskImage} />
                ) : (
                  <p className="text-[11px] text-surface-600 leading-relaxed px-1">
                    Upload a source image above, then paint over what you want
                    redrawn.
                  </p>
                )}
              </>
            )}
            {mode === "controlnet" && (
              <>
                <ImageDropzone
                  value={controlImage}
                  onChange={setControlImage}
                  label="Reference image"
                  hint="Its structure will steer the generation"
                />
                <ControlPanel />
              </>
            )}
            <PromptInput
              value={params.prompt}
              onChange={(v) => updateParams({ prompt: v })}
              negativeValue={params.negativePrompt}
              onNegativeChange={(v) => updateParams({ negativePrompt: v })}
            />
            <ParamPanel
              params={params}
              onChange={updateParams}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/** Shown when the Python engine isn't reachable — tells the user exactly what to do. */
function EngineOffGuide() {
  return (
    <div className="card p-6 max-w-md text-center">
      <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
        <AlertCircleIcon className="w-7 h-7 text-red-400" />
      </div>
      <h3 className="text-base font-semibold text-white mb-1.5">Engine not connected</h3>
      <p className="text-xs text-surface-400 leading-relaxed mb-4">
        The AI engine runs as a local Python service. Start it, then this page
        will connect automatically.
      </p>
      <div className="space-y-2 text-left">
        <div className="rounded-lg bg-surface-950/80 border border-surface-800/60 p-3">
          <p className="text-[10px] uppercase tracking-wider text-surface-500 mb-1.5 flex items-center gap-1.5">
            <ZapIcon className="w-3 h-3 text-aura-400" />
            Easiest (Windows)
          </p>
          <code className="text-xs text-emerald-300 font-mono">run.bat</code>
          <p className="text-[10px] text-surface-600 mt-1">Starts the engine and the UI together.</p>
        </div>
        <div className="rounded-lg bg-surface-950/80 border border-surface-800/60 p-3">
          <p className="text-[10px] uppercase tracking-wider text-surface-500 mb-1.5 flex items-center gap-1.5">
            <ZapIcon className="w-3 h-3 text-aura-400" />
            Manual
          </p>
          <code className="text-xs text-emerald-300 font-mono">npm run backend:dev</code>
        </div>
      </div>
    </div>
  );
}
