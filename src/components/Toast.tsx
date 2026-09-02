import { useAppStore } from "../stores/appStore";
import { AlertCircleIcon, CheckCircleIcon, InfoIcon } from "./icons";

const KIND_STYLES = {
  success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  error: "border-red-500/30 bg-red-500/10 text-red-300",
  info: "border-sky-500/30 bg-sky-500/10 text-sky-300",
} as const;

const KIND_ICONS = {
  success: CheckCircleIcon,
  error: AlertCircleIcon,
  info: InfoIcon,
} as const;

/** Bottom-right toast stack, rendered once from Layout. */
export function Toaster() {
  const toasts = useAppStore((s) => s.toasts);
  const dismissToast = useAppStore((s) => s.dismissToast);

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
      {toasts.map((toast) => {
        const Icon = KIND_ICONS[toast.kind];
        return (
          <button
            key={toast.id}
            onClick={() => dismissToast(toast.id)}
            className={`pointer-events-auto flex items-start gap-2.5 px-4 py-3 rounded-xl border backdrop-blur-md
                        shadow-xl shadow-black/30 text-left animate-toastIn
                        transition-opacity hover:opacity-80 ${KIND_STYLES[toast.kind]}`}
          >
            <Icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span className="text-xs leading-relaxed">{toast.message}</span>
          </button>
        );
      })}
    </div>
  );
}
