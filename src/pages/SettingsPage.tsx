import { useState } from "react";
import { useAppStore } from "../stores/appStore";
import { APP_VERSION } from "../version";
import { MAX_HISTORY, clearGenerations } from "../lib/historyDB";
import {
  CheckIcon,
  AlertCircleIcon,
  InfoIcon,
  ZapIcon,
} from "../components/icons";

export default function SettingsPage() {
  const {
    backendUrl,
    setBackendUrl,
    backendConnected,
    engineLoaded,
    engineDevice,
    engineModelId,
    clearResults,
    pushToast,
  } = useAppStore();

  const [urlDraft, setUrlDraft] = useState(backendUrl);

  const handleSaveUrl = () => {
    const trimmed = urlDraft.trim();
    if (!/^https?:\/\//.test(trimmed)) {
      pushToast("error", "URL must start with http:// or https://");
      return;
    }
    setBackendUrl(trimmed);
    pushToast("success", "Backend URL saved — reconnecting…");
  };

  const handleClearHistory = async () => {
    if (!window.confirm("Delete all saved generation history? This cannot be undone.")) return;
    clearResults();
    try {
      await clearGenerations();
      pushToast("success", "History cleared");
    } catch (err) {
      console.error("Failed to clear history:", err);
      pushToast("error", "Could not clear history");
    }
  };

  return (
    <div className="p-6 max-w-2xl animate-fadeIn">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-white">Settings</h2>
        <p className="text-sm text-surface-400 mt-1">
          Configure Aura Studio and view system status.
        </p>
      </div>

      <div className="space-y-4">
        {/* ── Backend Connection ──────────────────────────────────── */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-white mb-3">Backend Connection</h3>

          <label className="text-xs text-surface-500 mb-1.5 block">
            Engine URL
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={urlDraft}
              onChange={(e) => setUrlDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveUrl()}
              placeholder="http://127.0.0.1:8766"
              className="input-field flex-1 font-mono text-xs"
            />
            <button
              onClick={handleSaveUrl}
              disabled={!urlDraft.trim() || urlDraft.trim() === backendUrl}
              className="btn-primary !px-4 !py-2 text-xs whitespace-nowrap disabled:opacity-40"
            >
              Save
            </button>
          </div>
          <p className="text-[10px] text-surface-600 mt-1.5">
            Default is http://127.0.0.1:8766. Only change this if you moved the
            engine to another port or machine.
          </p>

          <div className="mt-4 pt-4 border-t border-surface-800/50 space-y-3">
            <StatusRow label="Connection">
              <span className="flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    backendConnected
                      ? "bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.7)]"
                      : "bg-red-500"
                  }`}
                />
                <span className={`text-xs font-medium ${backendConnected ? "text-emerald-400" : "text-red-400"}`}>
                  {backendConnected ? "Connected" : "Disconnected"}
                </span>
              </span>
            </StatusRow>

            <StatusRow label="Engine">
              {engineLoaded ? (
                <span className="text-xs font-medium text-emerald-400 flex items-center gap-1">
                  <CheckIcon className="w-3 h-3" />
                  Model loaded
                </span>
              ) : (
                <span className="text-xs font-medium text-amber-400 flex items-center gap-1">
                  <AlertCircleIcon className="w-3 h-3" />
                  No model loaded
                </span>
              )}
            </StatusRow>

            {engineModelId && (
              <StatusRow label="Active model">
                <span className="text-xs font-mono text-surface-300 max-w-[220px] truncate" title={engineModelId}>
                  {engineModelId}
                </span>
              </StatusRow>
            )}
            {engineDevice && (
              <StatusRow label="Compute device">
                <span className={`chip ${
                  engineDevice === "cuda"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    : "bg-surface-800/60 text-surface-400 border-surface-700/50"
                }`}>
                  {engineDevice === "cuda" ? "GPU (CUDA)" : engineDevice === "cpu" ? "CPU" : engineDevice}
                </span>
              </StatusRow>
            )}
          </div>
        </div>

        {/* ── Data ────────────────────────────────────────────────── */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-white mb-3">Data</h3>
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs text-surface-300">Generation history</p>
              <p className="text-[10px] text-surface-600 mt-0.5">
                Stored locally on this machine (IndexedDB), up to {MAX_HISTORY} images.
              </p>
            </div>
            <button
              onClick={handleClearHistory}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 transition-all flex-shrink-0"
            >
              Clear history
            </button>
          </div>
        </div>

        {/* ── Quick Tips ──────────────────────────────────────────── */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <InfoIcon className="w-4 h-4 text-aura-400" />
            Quick Tips
          </h3>
          <ul className="space-y-2 text-xs text-surface-400 leading-relaxed">
            <Tip>Download a model in the <strong className="text-surface-200">Models</strong> tab first — FLUX.1 Schnell is recommended: it's fast and high quality.</Tip>
            <Tip>Press <kbd className="kbd">Ctrl+Enter</kbd> anywhere on the Generate page to start a generation.</Tip>
            <Tip>Lower steps = faster; higher steps = more refined detail. Schnell models are fixed at 4 steps.</Tip>
            <Tip>Fix a seed to reproduce the exact same image, or leave it random to explore variations.</Tip>
            <Tip>Negative prompts work best with SD 1.5 / SDXL — they tell the model what to avoid.</Tip>
            <Tip>
              <ZapIcon className="w-3 h-3 inline-block text-amber-400 mr-1 -mt-0.5" />
              First generation after loading a model is slower (warm-up). Subsequent ones are much faster.
            </Tip>
          </ul>
        </div>

        {/* ── About ───────────────────────────────────────────────── */}
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-white mb-3">About</h3>
          <div className="space-y-1 text-xs text-surface-500">
            <p>Aura Studio v{APP_VERSION}</p>
            <p>Cross-platform AI image generation studio — everything runs locally on your machine.</p>
            <p>Hugging Face Diffusers · FastAPI · React · Tauri</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-surface-500">{label}</span>
      {children}
    </div>
  );
}

function Tip({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <span className="w-1 h-1 rounded-full bg-aura-400 mt-1.5 flex-shrink-0" />
      <span>{children}</span>
    </li>
  );
}
