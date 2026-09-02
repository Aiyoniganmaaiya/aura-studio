import { GenerationParams } from "../stores/appStore";
import { detectFamily, isFluxSchnell, ModelFamily } from "../lib/modelUtils";
import { useAppStore } from "../stores/appStore";
import { InfoIcon } from "./icons";

interface Props {
  params: GenerationParams;
  onChange: (partial: Partial<GenerationParams>) => void;
}

const SIZE_PRESETS = [
  { label: "Square", w: 1024, h: 1024 },
  { label: "Portrait", w: 768, h: 1024 },
  { label: "Landscape", w: 1024, h: 768 },
  { label: "Wide", w: 1280, h: 720 },
  { label: "SD 512", w: 512, h: 512 },
  { label: "SD 768", w: 768, h: 768 },
];

const FAMILY_LABEL: Record<ModelFamily, string> = {
  sd15: "SD 1.5",
  sd21: "SD 2.1",
  sdxl: "SDXL",
  flux: "FLUX",
  "z-image": "Z-Image",
};

export function ParamPanel({ params, onChange }: Props) {
  const activeModel = useAppStore((s) => s.activeModel);
  // Sniff from the ID actually being generated with — activeModel can lag
  // (or be null after a restart), which used to mislabel the family chip.
  const family = detectFamily(params.modelId || activeModel?.id || "");
  const schnell = isFluxSchnell(params.modelId);
  const isZImage = family === "z-image";
  const mode = params.mode ?? "txt2img";
  const showStrength = mode === "img2img" || mode === "inpaint";
  const useSeed = params.seed !== null;

  // FLUX.1 Schnell (4 steps) and Z-Image Turbo (8 steps) are distilled
  // samplers; guidance is unused on both.
  const stepsMax = schnell ? 4 : isZImage ? 8 : 50;
  const effectiveSteps = Math.min(params.steps, stepsMax);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-surface-400 uppercase tracking-wider">
          Parameters
        </label>
        <span className={`chip bg-aura-500/10 text-aura-300 border-aura-500/20`}>
          {FAMILY_LABEL[family]}
        </span>
      </div>

      {/* Size presets */}
      <div>
        <label className="text-xs text-surface-500 mb-1.5 block">Size</label>
        <div className="flex flex-wrap gap-1.5 mb-2">
          {SIZE_PRESETS.map((preset) => (
            <button
              key={`${preset.w}x${preset.h}`}
              onClick={() => onChange({ width: preset.w, height: preset.h })}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                params.width === preset.w && params.height === preset.h
                  ? "bg-aura-500/20 text-aura-300 border border-aura-500/30"
                  : "bg-surface-800/50 text-surface-400 border border-surface-700/30 hover:bg-surface-700/50"
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="number"
            value={params.width}
            onChange={(e) => onChange({ width: Number(e.target.value) })}
            min={256}
            max={2048}
            step={64}
            className="input-field w-20 text-center text-xs"
          />
          <span className="text-surface-500 text-sm self-center">×</span>
          <input
            type="number"
            value={params.height}
            onChange={(e) => onChange({ height: Number(e.target.value) })}
            min={256}
            max={2048}
            step={64}
            className="input-field w-20 text-center text-xs"
          />
        </div>
      </div>

      {/* Steps */}
      <div>
        <div className="flex justify-between mb-1">
          <label className="text-xs text-surface-500">Steps</label>
          <span className="text-xs text-surface-400 font-mono">{effectiveSteps}</span>
        </div>
        <input
          type="range"
          min={1}
          max={stepsMax}
          value={effectiveSteps}
          onChange={(e) => onChange({ steps: Number(e.target.value) })}
          className="slider-track w-full"
        />
        <div className="flex justify-between text-[10px] text-surface-600 mt-0.5">
          <span>1</span>
          <span>{stepsMax}</span>
        </div>
      </div>

      {/* Strength — img2img / inpaint only */}
      {showStrength && (
        <div>
          <div className="flex justify-between mb-1">
            <label className="text-xs text-surface-500">
              {mode === "inpaint" ? "Redraw Strength" : "Strength"}
            </label>
            <span className="text-xs text-surface-400 font-mono">
              {(params.strength ?? 0.6).toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={params.strength ?? 0.6}
            onChange={(e) => onChange({ strength: Number(e.target.value) })}
            className="slider-track w-full"
          />
          <div className="flex justify-between text-[10px] text-surface-600 mt-0.5">
            <span>keep original</span>
            <span>ignore</span>
          </div>
        </div>
      )}

      {/* Guidance Scale — not used by FLUX.1 Schnell */}
      {!schnell && !isZImage && (
        <div>
          <div className="flex justify-between mb-1">
            <label className="text-xs text-surface-500">Guidance Scale</label>
            <span className="text-xs text-surface-400 font-mono">{params.guidanceScale.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={1}
            max={20}
            step={0.5}
            value={params.guidanceScale}
            onChange={(e) => onChange({ guidanceScale: Number(e.target.value) })}
            className="slider-track w-full"
          />
          <div className="flex justify-between text-[10px] text-surface-600 mt-0.5">
            <span>1</span>
            <span>20</span>
          </div>
        </div>
      )}
      {(schnell || isZImage) && (
        <div className="flex items-start gap-2 rounded-lg bg-amber-400/5 border border-amber-400/15 px-2.5 py-2 animate-slideDown">
          <InfoIcon className="w-3.5 h-3.5 text-amber-400/80 mt-0.5 flex-shrink-0" />
          <p className="text-[11px] leading-relaxed text-amber-200/70">
            {isZImage
              ? "Z-Image Turbo is CFG-distilled (max 8 steps) — guidance scale doesn't apply."
              : "FLUX.1 Schnell uses a fixed 4-step turbo sampler — guidance scale doesn't apply to this model."}
          </p>
        </div>
      )}

      {/* Batch count */}
      <div>
        <div className="flex justify-between mb-1">
          <label className="text-xs text-surface-500">Batch</label>
          <span className="text-xs text-surface-400 font-mono">{params.batchCount ?? 1}</span>
        </div>
        <input
          type="range"
          min={1}
          max={6}
          step={1}
          value={params.batchCount ?? 1}
          onChange={(e) => onChange({ batchCount: Number(e.target.value) })}
          className="slider-track w-full"
        />
        <p className="text-[10px] text-surface-600 mt-0.5">
          {(params.batchCount ?? 1) > 1
            ? "Images run one after another; a fixed seed walks seed+1, +2…"
            : "Number of images to generate in sequence."}
        </p>
      </div>

      {/* Seed */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs text-surface-500">Seed</label>
          <button
            onClick={() =>
              onChange({
                seed: useSeed
                  ? null
                  : // A real random seed (the old code pinned 42 here).
                    Math.floor(Math.random() * 2 ** 31),
              })
            }
            className={`text-xs px-2 py-0.5 rounded-md transition-colors ${
              useSeed
                ? "bg-aura-500/15 text-aura-300 hover:bg-aura-500/25"
                : "bg-surface-800/50 text-surface-400 hover:bg-surface-700/60"
            }`}
          >
            {useSeed ? "Fixed" : "Random"}
          </button>
        </div>
        {useSeed ? (
          <input
            type="number"
            value={params.seed ?? 0}
            onChange={(e) =>
              // Empty input means "back to random", not seed 0.
              onChange({ seed: e.target.value === "" ? null : Number(e.target.value) })
            }
            className="input-field font-mono text-xs"
            placeholder="Seed value"
          />
        ) : (
          <p className="text-[10px] text-surface-600">
            Each image gets a fresh seed; the used seed is shown under the result.
          </p>
        )}
      </div>
    </div>
  );
}
