import { useEffect, useRef, useState } from "react";
import { useAppStore } from "../stores/appStore";
import {
  listControlnets,
  downloadControlnet,
  getDownloadProgress,
  type ControlNetInfo,
} from "../api/backend";
import { detectFamily } from "../lib/modelUtils";
import { CheckIcon, DownloadIcon, InfoIcon } from "./icons";

/**
 * ControlNet setup: pick weights for the current model family, choose the
 * control type, and steer how strongly the condition image guides generation.
 * Downloads poll the same task endpoint base models use.
 */
export function ControlPanel() {
  const { params, updateParams, pushToast } = useAppStore();
  const [catalog, setCatalog] = useState<ControlNetInfo[]>([]);
  const [busyTask, setBusyTask] = useState<string | null>(null);
  const [taskProgress, setTaskProgress] = useState(0);
  const pollRef = useRef<number | null>(null);

  const family = detectFamily(params.modelId || "");
  const familyEntries = catalog.filter((c) => c.family === family);

  useEffect(() => {
    let cancelled = false;
    listControlnets()
      .then((list) => {
        if (!cancelled) setCatalog(list);
      })
      .catch(() => pushToast("error", "Could not load the ControlNet list"));
    return () => {
      cancelled = true;
    };
  }, [pushToast]);

  // Stop any in-flight progress poll on unmount.
  useEffect(
    () => () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    },
    []
  );

  const startDownload = async (cn: ControlNetInfo) => {
    try {
      const { task_id } = await downloadControlnet(cn.id);
      setBusyTask(cn.id);
      setTaskProgress(0);
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const p = await getDownloadProgress(task_id);
          setTaskProgress(p.progress_pct);
          if (p.status === "completed") {
            window.clearInterval(pollRef.current!);
            pollRef.current = null;
            setBusyTask(null);
            // Mark downloaded + select it right away.
            setCatalog((prev) =>
              prev.map((c) => (c.id === cn.id ? { ...c, downloaded: true } : c))
            );
            updateParams({ cnModelId: cn.id });
            pushToast("success", `${cn.name} ready`);
          } else if (p.status === "error") {
            window.clearInterval(pollRef.current!);
            pollRef.current = null;
            setBusyTask(null);
            pushToast("error", p.error ?? "Download failed");
          }
        } catch {
          // transient polling errors are non-fatal
        }
      }, 1000);
    } catch (e) {
      pushToast("error", e instanceof Error ? e.message : "Could not start download");
    }
  };

  const selectedId =
    params.cnModelId && familyEntries.some((c) => c.id === params.cnModelId)
      ? params.cnModelId
      : familyEntries.find((c) => c.downloaded)?.id ?? "";
  const selected = familyEntries.find((c) => c.id === selectedId) ?? null;
  // Keep params in sync with the effective selection.
  const effectiveType =
    params.controlType && selected?.control_types.includes(params.controlType)
      ? params.controlType
      : selected?.control_types[0];

  // A stale controlType from a previously selected weight would silently feed
  // the wrong preprocessor (e.g. a depth map into a canny net) — normalize it.
  useEffect(() => {
    if (effectiveType && effectiveType !== params.controlType) {
      updateParams({ controlType: effectiveType });
    }
  }, [effectiveType, params.controlType, updateParams]);

  if (familyEntries.length === 0) {
    return (
      <div className="flex items-start gap-2 rounded-lg bg-surface-800/40 border border-surface-700/30 px-2.5 py-2">
        <InfoIcon className="w-3.5 h-3.5 text-surface-500 mt-0.5 flex-shrink-0" />
        <p className="text-[11px] leading-relaxed text-surface-500">
          No ControlNet weights for this model family. Switch to an SD 1.5,
          SDXL or FLUX model to use it.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <label className="text-xs text-surface-500 block">ControlNet</label>

      {/* Weight picker */}
      <div className="space-y-1.5">
        {familyEntries.map((cn) => {
          const isSelected = cn.id === selectedId;
          const downloading = busyTask === cn.id;
          return (
            <div
              key={cn.id}
              className={`rounded-lg border px-2.5 py-2 transition-colors ${
                isSelected
                  ? "bg-aura-500/10 border-aura-500/30"
                  : "bg-surface-900/50 border-surface-800/50 hover:border-surface-700"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <button
                  onClick={() => {
                    if (cn.downloaded) updateParams({ cnModelId: cn.id });
                  }}
                  title={cn.downloaded ? undefined : "Download this weight first to use it"}
                  className={`text-xs font-medium text-left flex-1 transition-colors ${
                    isSelected
                      ? "text-aura-300"
                      : cn.downloaded
                        ? "text-surface-300 hover:text-white"
                        : "text-surface-500 cursor-not-allowed"
                  }`}
                >
                  {cn.name}
                </button>
                {downloading ? (
                  <span className="text-[10px] font-mono text-aura-300">
                    {Math.round(taskProgress)}%
                  </span>
                ) : cn.downloaded ? (
                  <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                ) : (
                  <button
                    onClick={() => startDownload(cn)}
                    className="flex items-center gap-1 text-[10px] text-surface-400 hover:text-white bg-surface-800/70 hover:bg-surface-700 px-1.5 py-0.5 rounded-md transition-colors"
                    title={`Download from Hugging Face (${cn.size_hint})`}
                  >
                    <DownloadIcon className="w-3 h-3" />
                    Get
                  </button>
                )}
              </div>
              {downloading && (
                <div className="mt-1.5 h-1 rounded-full bg-surface-800 overflow-hidden">
                  <div
                    className="h-full bg-aura-500 transition-all"
                    style={{ width: `${taskProgress}%` }}
                  />
                </div>
              )}
              {!downloading && isSelected && (
                <p className="text-[10px] text-surface-600 mt-1 leading-relaxed">{cn.description}</p>
              )}
            </div>
          );
        })}
      </div>

      {/* Control type */}
      {selected && (
        <div>
          <label className="text-xs text-surface-500 mb-1.5 block">Control type</label>
          <div className="flex flex-wrap gap-1.5">
            {selected.control_types.map((type) => (
              <button
                key={type}
                onClick={() => updateParams({ controlType: type })}
                className={`px-2 py-0.5 rounded-md text-[11px] capitalize transition-all ${
                  effectiveType === type
                    ? "bg-aura-500/20 text-aura-300 border border-aura-500/30"
                    : "bg-surface-800/50 text-surface-400 border border-transparent hover:bg-surface-700/50"
                }`}
              >
                {type.replace(/-/g, " ")}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Conditioning scale */}
      <div>
        <div className="flex justify-between mb-1">
          <label className="text-xs text-surface-500">Control strength</label>
          <span className="text-xs text-surface-400 font-mono">
            {(params.cnWeight ?? 1.0).toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0.3}
          max={1.5}
          step={0.05}
          value={params.cnWeight ?? 1.0}
          onChange={(e) => updateParams({ cnWeight: Number(e.target.value) })}
          className="slider-track w-full"
        />
      </div>

      <p className="text-[10px] text-surface-600 leading-relaxed">
        Canny &amp; depth references are preprocessed automatically; pose-type
        images are used as-is.
      </p>
    </div>
  );
}
