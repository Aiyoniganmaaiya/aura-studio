import { GenerationMode, GenerationParams } from "../stores/appStore";
import { useAppStore } from "../stores/appStore";
import {
  SeedIcon,
  DownloadIcon,
  CopyIcon,
  ImageIcon,
  RefreshIcon,
  SparkleIcon,
} from "./icons";

interface Props {
  imageBase64: string;
  prompt: string;
  seed: number;
  elapsed: number;
  params: GenerationParams;
}

/** Turn a prompt into a safe, readable filename segment. */
function promptSlug(prompt: string): string {
  const slug = prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return slug || "image";
}

const MODE_BADGE: Partial<Record<GenerationMode, { label: string; cls: string }>> = {
  img2img: { label: "IMG2IMG", cls: "text-sky-300 bg-sky-400/10 border-sky-400/20" },
  inpaint: { label: "INPAINT", cls: "text-violet-300 bg-violet-400/10 border-violet-400/20" },
  controlnet: { label: "CONTROLNET", cls: "text-emerald-300 bg-emerald-400/10 border-emerald-400/20" },
};

export function ImageDisplay({ imageBase64, prompt, seed, elapsed, params }: Props) {
  const { pushToast, setInitImage, setControlImage, updateParams } = useAppStore();
  const dataUrl = `data:image/png;base64,${imageBase64}`;
  const filename = `aura-${promptSlug(prompt)}-${seed}.png`;
  const mode = params.mode ?? "txt2img";
  const badge = MODE_BADGE[mode];

  /** Send this image into an image-mode workflow and switch the UI to it.
   * Generation is NOT auto-fired — the user picks a fresh prompt first. */
  const useAsInput = (target: GenerationMode) => {
    if (target === "controlnet") setControlImage(imageBase64);
    else setInitImage(imageBase64);
    // Upscale-style flows keep size; others snap the canvas to this image.
    const double = Math.round((params.width * 2) / 8) * 8;
    updateParams({
      mode: target,
      ...(target === "img2img"
        ? {
            width: Math.min(2048, double),
            height: Math.min(2048, Math.round((params.height * 2) / 8) * 8),
          }
        : {}),
      seed,
    });
    pushToast("info", "Input ready — adjust the prompt, then Generate");
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    a.click();
    pushToast("success", "Image saved");
  };

  const handleCopy = async () => {
    try {
      const res = await fetch(dataUrl);
      const blob = await res.blob();
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": blob }),
      ]);
      pushToast("success", "Image copied to clipboard");
    } catch {
      // Some WebViews block image writes — offer the file instead.
      pushToast("info", "Clipboard not available — downloading the PNG instead");
      handleDownload();
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full">
      {/* Image */}
      <div className="relative rounded-2xl overflow-hidden bg-surface-900 border border-surface-800/50 shadow-2xl shadow-black/40 max-w-full">
        <img
          src={dataUrl}
          alt={prompt}
          className="max-w-full max-h-[62vh] object-contain"
        />
        {badge && (
          <span
            className={`absolute top-3 left-3 text-[10px] font-semibold tracking-wider px-2 py-0.5 rounded-md border backdrop-blur-sm ${badge.cls}`}
          >
            {badge.label}
          </span>
        )}
      </div>

      {/* Prompt recap */}
      <p className="text-xs text-surface-400 text-center max-w-xl leading-relaxed px-4">
        {prompt}
      </p>

      {/* Info bar */}
      <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 px-4 py-2 rounded-xl bg-surface-900/80 border border-surface-800/30 backdrop-blur-sm">
        <div className="flex items-center gap-1.5 text-xs text-surface-400" title="Seed — fix it to reproduce this image">
          <SeedIcon className="w-3.5 h-3.5 text-surface-500" />
          <span className="font-mono">{seed}</span>
        </div>
        <Divider />
        <div className="text-xs text-surface-400 font-mono" title="Generation time">
          {elapsed.toFixed(1)}s
        </div>
        <Divider />
        <div className="text-xs text-surface-400 font-mono" title="Resolution">
          {params.width}×{params.height}
        </div>
        <Divider />
        <div className="text-xs text-surface-400 font-mono" title="Sampler settings">
          {params.steps} steps · CFG {params.guidanceScale.toFixed(1)}
          {(mode === "img2img" || mode === "inpaint") && params.strength !== undefined && (
            <> · S {params.strength.toFixed(2)}</>
          )}
        </div>
        {params.modelId && (
          <>
            <Divider />
            <div className="text-xs text-surface-500 font-mono max-w-[180px] truncate" title={params.modelId}>
              {params.modelId.split("/").pop()}
            </div>
          </>
        )}
      </div>

      {params.negativePrompt && (
        <p className="text-[11px] text-surface-600 text-center max-w-xl px-4">
          <span className="text-surface-500">Negative:</span> {params.negativePrompt}
        </p>
      )}

      {/* Actions */}
      <div className="flex flex-wrap justify-center gap-2">
        <button onClick={handleDownload} className="btn-secondary !px-4 !py-2 text-xs">
          <DownloadIcon className="w-3.5 h-3.5" />
          Download
        </button>
        <button onClick={handleCopy} className="btn-secondary !px-4 !py-2 text-xs">
          <CopyIcon className="w-3.5 h-3.5" />
          Copy
        </button>
        {/* Workflow loops — prepare inputs for the image modes */}
        <button
          onClick={() => useAsInput("img2img")}
          title="Refine this image with img2img at 2× size (AI upscale)"
          className="btn-secondary !px-4 !py-2 text-xs"
        >
          <RefreshIcon className="w-3.5 h-3.5" />
          Enhance ×2
        </button>
        <button
          onClick={() => useAsInput("inpaint")}
          title="Paint over parts of this image to redraw"
          className="btn-secondary !px-4 !py-2 text-xs"
        >
          <SparkleIcon className="w-3.5 h-3.5" />
          Inpaint
        </button>
        <button
          onClick={() => useAsInput("controlnet")}
          title="Use this image as a ControlNet reference"
          className="btn-secondary !px-4 !py-2 text-xs"
        >
          <ImageIcon className="w-3.5 h-3.5" />
          As Reference
        </button>
      </div>
    </div>
  );
}

function Divider() {
  return <div className="w-px h-3.5 bg-surface-700/70" />;
}
