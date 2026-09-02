import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshIcon, TrashIcon } from "./icons";

interface Props {
  /** Raw base64 source image (no data: prefix). */
  imageB64: string;
  /** Receives the painted mask as raw base64 PNG (white = repaint), or null when empty. */
  onMaskChange: (b64: string | null) => void;
}

/** Longest side the working canvas uses — the backend resizes to the target
 * generation size anyway, so we just keep memory and pointer math sane. */
const MAX_WORKING_DIM = 1024;

export function InpaintCanvas({ imageB64, onMaskChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastPtRef = useRef<{ x: number; y: number } | null>(null);
  const brushRef = useRef(48);
  const [brush, setBrushState] = useState(48);

  const setBrush = (px: number) => {
    brushRef.current = px;
    setBrushState(px);
  };

  // Load the source into the canvas whenever a new image arrives.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageB64) return;
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, MAX_WORKING_DIM / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.round(img.naturalWidth * scale);
      const h = Math.round(img.naturalHeight * scale);
      canvas.width = w;
      canvas.height = h;
      // Default brush scales with the image so it's usable immediately.
      setBrush(Math.max(24, Math.round(Math.min(w, h) / 10)));
      onMaskChange(null); // fresh image — old mask no longer matches
    };
    img.src = `data:image/png;base64,${imageB64}`;
  }, [imageB64, onMaskChange]);

  // Keep the base photo in a ref for repaints.
  const imgElRef = useRef<HTMLImageElement | null>(null);

  /** Repaint the visible layer: source dimmed under red strokes. */
  const repaintBase = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imgElRef.current) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(imgElRef.current, 0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    if (!imageB64) return;
    const img = new Image();
    img.onload = () => {
      imgElRef.current = img;
      repaintBase();
    };
    img.src = `data:image/png;base64,${imageB64}`;
  }, [imageB64, repaintBase]);

  const toCanvasPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const strokeTo = (pt: { x: number; y: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const last = lastPtRef.current ?? pt;
    ctx.strokeStyle = "rgba(255,70,70,0.6)";
    ctx.lineWidth = brushRef.current;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();
    lastPtRef.current = pt;
  };

  /** Flatten strokes to a strict black/white mask (white = inpaint region). */
  const exportMask = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const src = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
    const out = new ImageData(canvas.width, canvas.height);
    let anyPainted = false;
    for (let i = 0; i < src.data.length; i += 4) {
      const painted = src.data[i + 3] > 8;
      if (painted) anyPainted = true;
      out.data[i] = 255; // R
      out.data[i + 1] = 255; // G
      out.data[i + 2] = 255; // B
      out.data[i + 3] = painted ? 255 : 0; // white opaque = repaint
    }
    if (!anyPainted) {
      onMaskChange(null);
      return;
    }
    const tmp = document.createElement("canvas");
    tmp.width = canvas.width;
    tmp.height = canvas.height;
    tmp.getContext("2d")!.putImageData(out, 0, 0);
    const dataUrl = tmp.toDataURL("image/png");
    onMaskChange(dataUrl.slice(dataUrl.indexOf(",") + 1));
  }, [onMaskChange]);

  const clearMask = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    repaintBase();
    onMaskChange(null);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs text-surface-500">Paint the area to redraw</label>
        <div className="flex items-center gap-2">
          {/* Brush size */}
          <input
            type="range"
            min={8}
            max={160}
            value={brush}
            onChange={(e) => setBrush(Number(e.target.value))}
            className="slider-track w-20"
            title={`Brush ${brush}px`}
          />
          <button
            onClick={clearMask}
            className="text-[11px] text-surface-500 hover:text-red-400 transition-colors flex items-center gap-1"
          >
            <TrashIcon className="w-3 h-3" />
            Clear
          </button>
        </div>
      </div>

      <div className="rounded-xl overflow-hidden border border-surface-700/50 bg-surface-950 select-none">
        <canvas
          ref={canvasRef}
          className="w-full block touch-none cursor-crosshair"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            drawingRef.current = true;
            lastPtRef.current = null;
            strokeTo(toCanvasPoint(e));
          }}
          onPointerMove={(e) => {
            if (drawingRef.current) strokeTo(toCanvasPoint(e));
          }}
          onPointerUp={() => {
            if (drawingRef.current) {
              drawingRef.current = false;
              exportMask();
            }
          }}
          onPointerLeave={() => {
            if (drawingRef.current) {
              drawingRef.current = false;
              exportMask();
            }
          }}
        />
      </div>

      <p className="text-[10px] text-surface-600 mt-1 flex items-center gap-1">
        <RefreshIcon className="w-3 h-3" />
        White areas are regenerated; everything else is preserved.
      </p>
    </div>
  );
}
