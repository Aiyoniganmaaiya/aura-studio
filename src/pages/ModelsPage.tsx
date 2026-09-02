import { useEffect, useState, useRef, useCallback } from "react";
import { useAppStore } from "../stores/appStore";
import {
  listModels,
  downloadModel,
  getDownloadProgress,
  loadModel,
  deleteModel,
  DownloadProgress,
} from "../api/backend";
import { ProgressBar, IndeterminateBar } from "../components/ProgressBar";
import {
  DownloadIcon,
  CheckCircleIcon,
  Spinner,
  TrashIcon,
  ZapIcon,
} from "../components/icons";

export default function ModelsPage() {
  const {
    models,
    setModels,
    activeModel,
    setActiveModel,
    backendConnected,
    engineModelId,
    setEngineStatus,
    pushToast,
  } = useAppStore();

  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({});
  const [customModelId, setCustomModelId] = useState("");
  const [filePattern, setFilePattern] = useState("");
  const [customError, setCustomError] = useState("");
  const pollRef = useRef<Record<string, number>>({});

  // Fetch models
  const fetchModels = useCallback(async () => {
    if (!backendConnected) return;
    setLoading(true);
    try {
      const updated = await listModels();
      setModels(updated);
    } catch (err: any) {
      console.error("Failed to fetch models:", err);
      pushToast("error", `Could not list models: ${err.message ?? err}`);
    }
    setLoading(false);
  }, [backendConnected, setModels, pushToast]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Cleanup pollers
  useEffect(() => {
    const polls = pollRef.current;
    return () => {
      Object.values(polls).forEach(clearInterval);
    };
  }, []);

  // Start polling download progress. Transient fetch errors are tolerated
  // (a network blip shouldn't orphan a running download) — give up only
  // after several consecutive failures.
  const startPolling = (taskId: string, modelId: string) => {
    let failures = 0;
    const interval = window.setInterval(async () => {
      try {
        const progress = await getDownloadProgress(taskId);
        failures = 0;
        setDownloadProgress((prev) => ({ ...prev, [modelId]: progress }));

        if (progress.status === "completed") {
          clearInterval(interval);
          delete pollRef.current[taskId];
          setDownloading(null);
          fetchModels();
          pushToast("success", `${modelId.split("/").pop()} downloaded — click Load to activate it.`);
        } else if (progress.status === "error") {
          clearInterval(interval);
          delete pollRef.current[taskId];
          setDownloading(null);
          pushToast("error", `Download failed: ${progress.error ?? "unknown error"}`);
        }
      } catch {
        failures++;
        if (failures >= 3) {
          clearInterval(interval);
          delete pollRef.current[taskId];
          setDownloading(null);
          pushToast("error", "Lost track of the download — the engine may have restarted.");
          fetchModels();
        }
      }
    }, 1000);
    pollRef.current[taskId] = interval;
  };

  // Download from catalog
  const handleDownload = async (modelId: string, modelType: string) => {
    setDownloading(modelId);
    setCustomError("");
    try {
      const result = await downloadModel(modelId, modelType);
      setDownloadProgress((prev) => ({
        ...prev,
        [modelId]: {
          task_id: result.task_id,
          model_id: modelId,
          status: "starting",
          progress_pct: 0,
          error: null,
          elapsed: 0,
        },
      }));
      startPolling(result.task_id, modelId);
    } catch (err: any) {
      console.error("Download failed:", err);
      setDownloading(null);
      pushToast("error", `Could not start download: ${err.message ?? err}`);
    }
  };

  // Download custom model
  const handleCustomDownload = async () => {
    const id = customModelId.trim();
    if (!id) return;
    if (!id.includes("/")) {
      setCustomError("Model ID should be in format: org/model-name");
      return;
    }
    const pattern = filePattern.trim();
    if (pattern && (pattern.includes("..") || pattern.startsWith("/"))) {
      setCustomError("File pattern looks invalid — use a glob like *Q6_K*.gguf");
      return;
    }
    setCustomError("");
    setDownloading(id);
    try {
      const result = await downloadModel(id, "", pattern || undefined);
      setDownloadProgress((prev) => ({
        ...prev,
        [id]: {
          task_id: result.task_id,
          model_id: id,
          status: "starting",
          progress_pct: 0,
          error: null,
          elapsed: 0,
        },
      }));
      startPolling(result.task_id, id);
    } catch (err: any) {
      const msg = err.message || "Download failed";
      setCustomError(msg);
      pushToast("error", msg);
      setDownloading(null);
    }
  };

  // Load model (with visible in-flight state — loading can take a minute)
  const handleLoad = async (modelId: string, modelType: string) => {
    if (loadingModelId) return;
    setLoadingModelId(modelId);
    try {
      const result = await loadModel(modelId, modelType);
      setEngineStatus(true, result.device, result.model_id);
      const model = models.find((m) => m.id === modelId);
      if (model) setActiveModel(model);
      else {
        // Custom models may not be in the fetched list yet — synthesize one.
        setActiveModel({
          id: modelId,
          name: modelId.split("/").pop() ?? modelId,
          type: modelType || "custom",
          local: true,
          downloaded: true,
        });
      }
      pushToast("success", `${modelId.split("/").pop()} loaded on ${result.device.toUpperCase()}`);
    } catch (err: any) {
      console.error("Load failed:", err);
      pushToast("error", `Load failed: ${err.message ?? err}`);
    }
    setLoadingModelId(null);
  };

  // Delete a downloaded model's files from disk
  const handleDelete = async (modelId: string, modelType: string) => {
    if (!window.confirm(`Delete local files for ${modelId}? You can re-download it anytime.`)) return;
    try {
      await deleteModel(modelId, modelType);
      pushToast("success", "Model files deleted");
      fetchModels();
    } catch (err: any) {
      pushToast("error", `Delete failed: ${err.message ?? err}`);
    }
  };

  const modelTypeColors: Record<string, string> = {
    sd15: "bg-blue-500/15 text-blue-300 border-blue-500/25",
    sd21: "bg-indigo-500/15 text-indigo-300 border-indigo-500/25",
    sdxl: "bg-purple-500/15 text-purple-300 border-purple-500/25",
    flux: "bg-amber-500/15 text-amber-300 border-amber-500/25",
    custom: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  };

  const modelTypeLabels: Record<string, string> = {
    sd15: "SD 1.5",
    sd21: "SD 2.1",
    sdxl: "SDXL",
    flux: "FLUX",
    custom: "Custom",
  };

  const dlProgress = (modelId: string) => downloadProgress[modelId];
  const catalog = models.filter((m) => m.type !== "custom");
  // Recommended first, then the rest of the catalog order.
  const orderedCatalog = [...catalog].sort(
    (a, b) => Number(b.recommended ?? false) - Number(a.recommended ?? false)
  );

  return (
    <div className="p-6 max-w-4xl animate-fadeIn">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-white">Models</h2>
        <p className="text-sm text-surface-400 mt-1">
          Download and manage AI image generation models. Supports any model from Hugging Face.
        </p>
      </div>

      {/* ── Custom Model Download ─────────────────────────────────── */}
      <div className="card p-4 mb-6">
        <h3 className="text-sm font-semibold text-white mb-3">
          <span className="flex items-center gap-2">
            <DownloadIcon className="w-4 h-4 text-aura-400" />
            Download Custom Model
          </span>
        </h3>
        <p className="text-xs text-surface-500 mb-3">
          Enter any Hugging Face model ID, e.g.{" "}
          <code className="text-aura-300 bg-aura-500/10 px-1 py-0.5 rounded text-[10px]">
            prompthero/openjourney-v4
          </code>{" "}
          or{" "}
          <code className="text-aura-300 bg-aura-500/10 px-1 py-0.5 rounded text-[10px]">
            nitrosocke/mo-di-diffusion
          </code>
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={customModelId}
            onChange={(e) => { setCustomModelId(e.target.value); setCustomError(""); }}
            onKeyDown={(e) => e.key === "Enter" && !downloading && handleCustomDownload()}
            placeholder="org/model-name (e.g. prompthero/openjourney-v4)"
            className="input-field flex-1 text-xs font-mono"
          />
          <button
            onClick={handleCustomDownload}
            disabled={!customModelId.trim() || downloading !== null}
            className="btn-primary !px-4 !py-2 text-xs whitespace-nowrap"
          >
            {downloading === customModelId.trim() ? <Spinner className="w-3 h-3" /> : "Download"}
          </button>
        </div>

        {/* Optional per-file filter (GGUF quantizations etc.) */}
        <div className="mt-3">
          <label className="text-[10px] uppercase tracking-wider text-surface-500 mb-1 block">
            File filter <span className="normal-case tracking-normal text-surface-600">(optional)</span>
          </label>
          <input
            type="text"
            value={filePattern}
            onChange={(e) => setFilePattern(e.target.value)}
            placeholder='e.g. *Q6_K*.gguf — grab one quantization instead of the whole repo'
            className="input-field font-mono text-xs"
          />
          <p className="text-[10px] text-surface-600 mt-1">
            Multi-quant GGUF repos ship every size (up to ~80 GB total) — a filter like{" "}
            <code className="text-aura-300">*Q6_K*.gguf</code> downloads just that variant.
          </p>
        </div>
        {customError && (
          <p className="mt-2 text-xs text-red-400 animate-slideDown">{customError}</p>
        )}
        {dlProgress(customModelId.trim()) && (
          <div className="mt-3">
            <DownloadTaskProgress progress={dlProgress(customModelId.trim())!} />
          </div>
        )}
      </div>

      {/* ── Catalog Models ────────────────────────────────────────── */}
      <h3 className="text-sm font-semibold text-surface-300 mb-3">Available Models</h3>

      {!backendConnected ? (
        <div className="card p-8 text-center">
          <p className="text-sm text-surface-400">
            Not connected to the engine. Start it with{" "}
            <code className="text-emerald-300 font-mono text-xs">run.bat</code> or{" "}
            <code className="text-emerald-300 font-mono text-xs">npm run backend:dev</code>.
          </p>
        </div>
      ) : loading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card p-4 animate-pulse">
              <div className="h-4 bg-surface-800 rounded w-1/3 mb-2" />
              <div className="h-3 bg-surface-800 rounded w-2/3" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid gap-3">
          {orderedCatalog.map((model) => {
            // The engine's actually-loaded model is the source of truth for
            // "Active"; fall back to the store's selection until /health confirms.
            const isActive = model.id === (engineModelId ?? activeModel?.id);
            const isDownloading = downloading === model.id;
            const isLoadLoading = loadingModelId === model.id;
            const prog = dlProgress(model.id);

            return (
              <div
                key={model.id}
                className={`card p-4 transition-all duration-200 ${
                  isActive
                    ? "ring-1 ring-aura-500/40 border-aura-500/30 shadow-aura-950/40"
                    : "hover:border-surface-700"
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <h3 className="text-sm font-semibold text-white truncate">
                        {model.name}
                      </h3>
                      <span className={`chip ${modelTypeColors[model.type] || modelTypeColors.custom}`}>
                        {modelTypeLabels[model.type] || model.type}
                      </span>
                      {model.recommended && (
                        <span className="chip bg-emerald-500/10 text-emerald-300 border-emerald-500/25">
                          Recommended
                        </span>
                      )}
                      {model.downloaded && (
                        <span className="chip bg-surface-800/60 text-surface-400 border-surface-700/50">
                          Downloaded
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-surface-500 truncate">
                      {model.description || model.id}
                    </p>
                    <p className="text-[10px] text-surface-600 font-mono mt-1 truncate">
                      {model.id}
                    </p>
                  </div>

                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    {model.downloaded && !isActive && (
                      <button
                        onClick={() => handleDelete(model.id, model.type)}
                        disabled={isDownloading || loadingModelId !== null}
                        title="Delete downloaded files"
                        className="p-2 rounded-lg text-surface-600 hover:text-red-400 hover:bg-red-500/10 transition-all disabled:opacity-40"
                      >
                        <TrashIcon className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {model.downloaded ? (
                      <button
                        onClick={() => handleLoad(model.id, model.type)}
                        disabled={loadingModelId !== null || downloading !== null}
                        className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all inline-flex items-center gap-1.5 ${
                          isActive
                            ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 cursor-default"
                            : "bg-gradient-to-b from-aura-500 to-aura-600 text-white border border-aura-400/30 shadow-md shadow-aura-600/20 hover:from-aura-400 hover:to-aura-500 disabled:opacity-50"
                        }`}
                      >
                        {isLoadLoading ? (
                          <>
                            <Spinner className="w-3 h-3" />
                            Loading…
                          </>
                        ) : isActive ? (
                          <>
                            <CheckCircleIcon className="w-3.5 h-3.5" />
                            Active
                          </>
                        ) : (
                          <>
                            <ZapIcon className="w-3 h-3" />
                            Load
                          </>
                        )}
                      </button>
                    ) : (
                      <button
                        onClick={() => handleDownload(model.id, model.type)}
                        disabled={isDownloading || loadingModelId !== null}
                        className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-surface-800/80 text-surface-300 border border-surface-700 hover:bg-surface-700 transition-all disabled:opacity-50 inline-flex items-center gap-1.5"
                      >
                        {isDownloading ? (
                          <>
                            <Spinner className="w-3 h-3" />
                            Downloading…
                          </>
                        ) : (
                          <>
                            <DownloadIcon className="w-3.5 h-3.5" />
                            Download
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {/* Download progress bar */}
                {(isDownloading || prog) && prog && prog.status !== "completed" && (
                  <div className="mt-3">
                    <DownloadTaskProgress progress={prog} />
                  </div>
                )}
              </div>
            );
          })}

          {/* ── Custom Models List ──────────────────────────────────── */}
          {models.filter((m) => m.type === "custom").length > 0 && (
            <>
              <h3 className="text-sm font-semibold text-surface-300 mt-6 mb-3">
                Custom Models
              </h3>
              {models
                .filter((m) => m.type === "custom")
                .map((model) => {
                  // Same "Active" rule as the catalog above.
                  const isActive = model.id === (engineModelId ?? activeModel?.id);
                  const isLoadLoading = loadingModelId === model.id;
                  return (
                    <div key={model.id} className="card p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            <h3 className="text-sm font-semibold text-white truncate">
                              {model.name}
                            </h3>
                            <span className="chip bg-emerald-500/15 text-emerald-300 border-emerald-500/25">
                              Custom
                            </span>
                            <span className="chip bg-surface-800/60 text-surface-400 border-surface-700/50">
                              {model.default_size}px default
                            </span>
                          </div>
                          <p className="text-[10px] text-surface-600 font-mono truncate">
                            {model.id}
                          </p>
                        </div>
                        <div className="flex items-center gap-1.5 flex-shrink-0">
                          {!isActive && (
                            <button
                              onClick={() => handleDelete(model.id, model.type)}
                              disabled={loadingModelId !== null}
                              title="Delete downloaded files"
                              className="p-2 rounded-lg text-surface-600 hover:text-red-400 hover:bg-red-500/10 transition-all disabled:opacity-40"
                            >
                              <TrashIcon className="w-3.5 h-3.5" />
                            </button>
                          )}
                          <button
                            onClick={() => handleLoad(model.id, model.type)}
                            disabled={loadingModelId !== null || downloading !== null}
                            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all inline-flex items-center gap-1.5 ${
                              isActive
                                ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 cursor-default"
                                : "bg-gradient-to-b from-aura-500 to-aura-600 text-white border border-aura-400/30 shadow-md shadow-aura-600/20 hover:from-aura-400 hover:to-aura-500 disabled:opacity-50"
                            }`}
                          >
                            {isLoadLoading ? (
                              <>
                                <Spinner className="w-3 h-3" />
                                Loading…
                              </>
                            ) : isActive ? (
                              <>
                                <CheckCircleIcon className="w-3.5 h-3.5" />
                                Active
                              </>
                            ) : (
                              <>
                                <ZapIcon className="w-3 h-3" />
                                Load
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
            </>
          )}

          {models.length === 0 && !loading && (
            <div className="card p-8 text-center">
              <p className="text-sm text-surface-500">
                No models available. Make sure the backend engine is running.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Download Progress Sub-component ──────────────────────────────────

function DownloadTaskProgress({ progress }: { progress: DownloadProgress }) {
  if (!progress) return null;

  if (progress.status === "completed") {
    return (
      <div className="text-xs text-emerald-300 flex items-center gap-1.5">
        <CheckCircleIcon className="w-3.5 h-3.5" />
        Download complete
      </div>
    );
  }

  if (progress.status === "error") {
    return (
      <div className="text-xs text-red-400 animate-slideDown">Error: {progress.error}</div>
    );
  }

  if (progress.status === "starting") {
    return (
      <div className="space-y-1">
        <IndeterminateBar />
        <p className="text-[10px] text-surface-500">Starting download...</p>
      </div>
    );
  }

  return (
    <ProgressBar
      current={progress.progress_pct}
      total={100}
      label={`Downloading — ${(progress.progress_pct).toFixed(0)}% (${progress.elapsed.toFixed(0)}s)`}
      showLabel={true}
      color="bg-gradient-to-r from-aura-600 to-aura-400"
    />
  );
}
