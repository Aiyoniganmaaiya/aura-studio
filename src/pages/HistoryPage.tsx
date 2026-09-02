import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAppStore, GenerationResult } from "../stores/appStore";
import {
  listGenerations,
  deleteGeneration,
  clearGenerations,
} from "../lib/historyDB";
import {
  ClockIcon,
  TrashIcon,
  SparkleIcon,
  Spinner,
} from "../components/icons";

export default function HistoryPage() {
  const { results, setResults, removeResult, clearResults, setCurrentResult, pushToast } =
    useAppStore();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  // History lives in IndexedDB, so it survives restarts — load on mount.
  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const items = await listGenerations();
      setResults(items);
    } catch (err) {
      console.error("Failed to load history:", err);
      pushToast("error", "Could not load history from local storage");
    }
    setLoading(false);
  }, [setResults, pushToast]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleOpen = (result: GenerationResult) => {
    setCurrentResult(result);
    navigate("/");
  };

  const handleDelete = async (result: GenerationResult, e: React.MouseEvent) => {
    e.stopPropagation();
    removeResult(result.id);
    try {
      await deleteGeneration(result.id);
    } catch (err) {
      console.error("Failed to delete entry:", err);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm("Delete all generation history? Images cannot be recovered.")) return;
    clearResults();
    try {
      await clearGenerations();
      pushToast("success", "History cleared");
    } catch (err) {
      console.error("Failed to clear history:", err);
      pushToast("error", "Could not clear history");
    }
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    const today = new Date();
    const isToday = d.toDateString() === today.toDateString();
    return isToday
      ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString([], { month: "short", day: "numeric" });
  };

  return (
    <div className="p-6 max-w-6xl animate-fadeIn">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">History</h2>
          <p className="text-sm text-surface-400 mt-1">
            Your generated images, saved locally on this machine.
            {results.length > 0 && (
              <span className="text-surface-500"> · {results.length} image{results.length !== 1 ? "s" : ""}</span>
            )}
          </p>
        </div>
        {results.length > 0 && (
          <button
            onClick={handleClearAll}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-surface-400 bg-surface-800/60 border border-surface-700/50 hover:text-red-400 hover:border-red-500/30 hover:bg-red-500/10 transition-all inline-flex items-center gap-1.5 flex-shrink-0"
          >
            <TrashIcon className="w-3.5 h-3.5" />
            Clear all
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-24 text-surface-500">
          <Spinner className="w-6 h-6" />
        </div>
      ) : results.length === 0 ? (
        <div className="card p-12 flex flex-col items-center justify-center text-center">
          <div className="w-14 h-14 mb-4 rounded-2xl bg-surface-800/60 border border-surface-700/40 flex items-center justify-center">
            <ClockIcon className="w-7 h-7 text-surface-500" />
          </div>
          <p className="text-sm text-surface-300 font-medium">No generations yet</p>
          <p className="text-xs text-surface-500 mt-1 mb-5">
            Everything you create will be saved here automatically.
          </p>
          <Link to="/" className="btn-primary !px-4 !py-2 text-xs">
            <SparkleIcon className="w-3.5 h-3.5" />
            Start creating
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {results.map((result) => (
            <button
              key={result.id}
              onClick={() => handleOpen(result)}
              title="Click to view full size"
              className="group relative rounded-xl overflow-hidden bg-surface-900 border border-surface-800/50
                         hover:border-aura-500/30 hover:shadow-lg hover:shadow-aura-950/30 transition-all duration-200 text-left"
            >
              <div className="aspect-square overflow-hidden relative">
                <img
                  src={`data:image/png;base64,${result.imageBase64}`}
                  alt={result.prompt}
                  loading="lazy"
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
                />
                {/* Delete overlay */}
                <div
                  role="button"
                  tabIndex={0}
                  onClick={(e) => handleDelete(result, e)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleDelete(result, e as unknown as React.MouseEvent);
                  }}
                  title="Delete this image"
                  className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/60 backdrop-blur-sm text-surface-300
                             opacity-0 group-hover:opacity-100 hover:!text-red-400 hover:!bg-black/80
                             transition-all duration-150 cursor-pointer"
                >
                  <TrashIcon className="w-3.5 h-3.5" />
                </div>
              </div>
              <div className="p-2.5">
                <p className="text-xs text-surface-300 truncate" title={result.prompt}>
                  {result.prompt}
                </p>
                <div className="flex items-center gap-1.5 mt-1 text-[10px] text-surface-500 font-mono">
                  <span>{formatTime(result.timestamp)}</span>
                  <span className="text-surface-700">·</span>
                  <span>{result.elapsed.toFixed(1)}s</span>
                  <span className="text-surface-700">·</span>
                  <span>seed {result.seed}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
