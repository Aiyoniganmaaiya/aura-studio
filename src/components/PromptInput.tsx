import { useState } from "react";
import { ChevronIcon, SparkleIcon } from "./icons";

interface Props {
  value: string;
  onChange: (v: string) => void;
  negativeValue: string;
  onNegativeChange: (v: string) => void;
}

const PROMPT_IDEAS = [
  "a cozy cabin in a snowy forest at dusk, warm light in the windows",
  "cyberpunk city street after rain, neon reflections, cinematic",
  "watercolor painting of a hot air balloon over misty mountains",
  "portrait of an astronaut with flowers growing from the helmet, soft light",
];

export function PromptInput({ value, onChange, negativeValue, onNegativeChange }: Props) {
  const [showNegative, setShowNegative] = useState(false);

  // Cycle through example prompts — a gentle onboarding for new users.
  const handleIdea = () => {
    const idx = PROMPT_IDEAS.indexOf(value);
    onChange(PROMPT_IDEAS[(idx + 1) % PROMPT_IDEAS.length]);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-surface-400 uppercase tracking-wider">
          Prompt
        </label>
        {value && (
          <button
            onClick={() => onChange("")}
            className="text-[10px] text-surface-600 hover:text-surface-400 transition-colors"
          >
            Clear
          </button>
        )}
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Describe the image you want to create..."
        rows={4}
        autoFocus
        className="input-field resize-none min-h-[100px] leading-relaxed"
      />

      {/* Stuck for words? Insert a rotating example prompt. */}
      <button
        onClick={handleIdea}
        className="flex items-center gap-1.5 text-xs text-surface-500 hover:text-aura-300 transition-colors"
      >
        <SparkleIcon className="w-3 h-3" />
        Need an idea?
      </button>

      {/* Negative prompt toggle */}
      <div>
        <button
          onClick={() => setShowNegative(!showNegative)}
          className="flex items-center gap-1.5 text-xs text-surface-500 hover:text-surface-300 transition-colors py-0.5"
        >
          <ChevronIcon className={`w-3 h-3 transition-transform duration-150 ${showNegative ? "rotate-90" : ""}`} />
          Negative Prompt
          {negativeValue && (
            <span className="chip bg-purple-500/10 text-purple-300 border-purple-500/20 ml-1">
              set
            </span>
          )}
        </button>
        {showNegative && (
          <div className="mt-2 animate-slideDown">
            <textarea
              value={negativeValue}
              onChange={(e) => onNegativeChange(e.target.value)}
              placeholder="What you don't want in the image (e.g. blurry, low quality, extra fingers)"
              rows={2}
              className="input-field resize-none leading-relaxed"
            />
            {!negativeValue && (
              <p className="text-[10px] text-surface-600 mt-1">
                Tip: works best with SD 1.5 / SDXL models.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
