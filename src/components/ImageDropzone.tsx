import { useCallback, useEffect, useRef, useState } from "react";
import { ImageIcon, XIcon } from "./icons";

interface Props {
  /** Raw base64 PNG (no data: prefix), or null when empty. */
  value: string | null;
  onChange: (b64: string | null) => void;
  /** Fired with the image's natural size after a successful load. */
  onSize?: (width: number, height: number) => void;
  label?: string;
  hint?: string;
}

/** Read a File into a raw base64 payload + natural size. */
function readImageFile(file: File): Promise<{ b64: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the file"));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      const img = new Image();
      img.onerror = () => reject(new Error("Not a readable image"));
      img.onload = () => {
        const b64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
        resolve({ b64, width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

export function ImageDropzone({ value, onChange, onSize, label = "Input image", hint }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const acceptFile = useCallback(
    async (file: File | null | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        setError("That file is not an image");
        return;
      }
      try {
        const { b64, width, height } = await readImageFile(file);
        setError(null);
        onChange(b64);
        onSize?.(width, height);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not load the image");
      }
    },
    [onChange, onSize]
  );

  // Paste-to-load, but never steal the clipboard from text inputs.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable))
        return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
      const file = item?.getAsFile();
      if (file) {
        e.preventDefault();
        acceptFile(file);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [acceptFile]);

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs text-surface-500">{label}</label>
        {value && (
          <button
            onClick={() => onChange(null)}
            className="text-[11px] text-surface-500 hover:text-red-400 transition-colors flex items-center gap-1"
          >
            <XIcon className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      {value ? (
        <div className="relative rounded-xl overflow-hidden border border-surface-700/50 bg-surface-950">
          <img
            src={`data:image/png;base64,${value}`}
            alt={label}
            className="w-full max-h-48 object-contain"
          />
          <button
            onClick={() => inputRef.current?.click()}
            className="absolute inset-0 bg-black/0 hover:bg-black/40 transition-colors flex items-center justify-center opacity-0 hover:opacity-100"
          >
            <span className="text-xs text-white bg-surface-900/90 rounded-lg px-2.5 py-1">Replace</span>
          </button>
        </div>
      ) : (
        <button
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            acceptFile(e.dataTransfer.files?.[0]);
          }}
          className={`w-full h-28 rounded-xl border border-dashed flex flex-col items-center justify-center gap-1.5 transition-colors ${
            dragOver
              ? "border-aura-400/60 bg-aura-500/10"
              : "border-surface-700/50 bg-surface-900/40 hover:border-surface-600 hover:bg-surface-800/30"
          }`}
        >
          <ImageIcon className="w-5 h-5 text-surface-500" />
          <span className="text-xs text-surface-400">Click, drop or paste an image</span>
          {hint && <span className="text-[10px] text-surface-600">{hint}</span>}
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          acceptFile(e.target.files?.[0]);
          e.target.value = ""; // allow re-picking the same file
        }}
      />

      {error && <p className="text-[11px] text-red-400 mt-1.5">{error}</p>}
    </div>
  );
}
