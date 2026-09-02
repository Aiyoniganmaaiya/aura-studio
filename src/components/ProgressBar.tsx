interface ProgressBarProps {
  current: number;
  total: number;
  label?: string;
  showLabel?: boolean;
  className?: string;
  color?: string;
}

export function ProgressBar({
  current,
  total,
  label,
  showLabel = true,
  className = "",
  color = "bg-aura-500",
}: ProgressBarProps) {
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-surface-400">
            {label || `${current} / ${total}`}
          </span>
          <span className="text-xs font-mono text-surface-500">{pct}%</span>
        </div>
      )}
      <div className="w-full h-1.5 bg-surface-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function IndeterminateBar({ className = "" }: { className?: string }) {
  return (
    <div className={`w-full ${className}`}>
      <div className="w-full h-1 bg-surface-800 rounded-full overflow-hidden relative">
        <div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-aura-400 to-transparent rounded-full animate-indeterminate"
          style={{
            width: "40%",
            animation: "indeterminate 1.5s ease-in-out infinite",
          }}
        />
      </div>
      <style>{`
        @keyframes indeterminate {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
      `}</style>
    </div>
  );
}
