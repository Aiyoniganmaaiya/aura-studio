import type { GenerationMode } from "../stores/appStore";
import { capabilitiesFor, type ModelFamily } from "../lib/modelUtils";

interface Props {
  value: GenerationMode;
  onChange: (mode: GenerationMode) => void;
  family: ModelFamily;
}

/** Why a mode is unavailable for a given family (shown as the tab tooltip). */
const UNSUPPORTED_HINT: Partial<Record<ModelFamily, Partial<Record<GenerationMode, string>>>> = {
  flux: {
    inpaint: "FLUX inpainting needs the separate FLUX.1-Fill model — not supported yet.",
  },
  "z-image": {
    controlnet: "No ControlNet ecosystem exists for Z-Image yet (diffusers ships no pipeline).",
  },
  sd21: {
    controlnet: "SD 2.1 has no usable ControlNet weights.",
  },
};

const TABS: { mode: GenerationMode; label: string }[] = [
  { mode: "txt2img", label: "Text" },
  { mode: "img2img", label: "Image" },
  { mode: "inpaint", label: "Inpaint" },
  { mode: "controlnet", label: "ControlNet" },
];

export function ModeTabs({ value, onChange, family }: Props) {
  const capabilities = capabilitiesFor(family);

  return (
    <div className="flex items-center gap-1 bg-surface-900/60 border border-surface-800/50 rounded-xl p-1">
      {TABS.map(({ mode, label }) => {
        const supported = mode === "txt2img" || Boolean(capabilities[mode]);
        const active = value === mode;
        const hint = !supported
          ? UNSUPPORTED_HINT[family]?.[mode] ?? `Not available for ${family} models`
          : undefined;
        return (
          <button
            key={mode}
            disabled={!supported}
            title={hint}
            onClick={() => onChange(mode)}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
              active
                ? "bg-aura-500/20 text-aura-300 shadow-inner"
                : supported
                  ? "text-surface-400 hover:text-white hover:bg-surface-800/60"
                  : "text-surface-700 cursor-not-allowed"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
