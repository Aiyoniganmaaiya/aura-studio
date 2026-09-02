import { Outlet, NavLink, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { checkHealth } from "../api/backend";
import { useAppStore } from "../stores/appStore";
import { APP_VERSION } from "../version";
import { Toaster } from "./Toast";
import {
  SparkleIcon,
  CubeIcon,
  ClockIcon,
  GearIcon,
} from "./icons";

const navItems = [
  { path: "/", label: "Generate", icon: SparkleIcon },
  { path: "/models", label: "Models", icon: CubeIcon },
  { path: "/history", label: "History", icon: ClockIcon },
  { path: "/settings", label: "Settings", icon: GearIcon },
];

export function Layout() {
  const location = useLocation();
  const { backendUrl, backendConnected, engineLoaded, engineDevice, setBackendConnected, setEngineStatus } =
    useAppStore();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let disposed = false;

    const check = async () => {
      try {
        const health = await checkHealth();
        if (disposed) return;
        setBackendConnected(true);
        // Track both loaded and unloaded states so a model swap/unload is
        // reflected everywhere (health carries model_id, fixing the empty
        // "Active Model" row after a restart).
        setEngineStatus(health.engine_loaded, health.device, health.model_id);
      } catch {
        if (disposed) return;
        setBackendConnected(false);
      }
      if (!disposed) setChecking(false);
    };

    setChecking(true);
    check();
    const interval = setInterval(check, 5000);
    return () => {
      disposed = true;
      clearInterval(interval);
    };
  }, [backendUrl, setBackendConnected, setEngineStatus]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-surface-900/50 border-r border-surface-800/40 backdrop-blur-xl flex flex-col">
        {/* Logo */}
        <div className="h-14 flex items-center gap-2.5 px-5 border-b border-surface-800/40">
          <div className="relative w-7 h-7 rounded-lg bg-gradient-to-br from-aura-400 to-aura-700 flex items-center justify-center shadow-lg shadow-aura-600/30">
            <SparkleIcon className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-tight leading-tight">
              Aura Studio
            </h1>
            <p className="text-[10px] text-surface-500 font-medium">v{APP_VERSION}</p>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 px-2 space-y-1">
          {navItems.map((item) => {
            const isActive = item.path === "/"
              ? location.pathname === "/"
              : location.pathname.startsWith(item.path);
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? "bg-aura-500/15 text-aura-300"
                    : "text-surface-400 hover:text-surface-200 hover:bg-surface-800/40"
                }`}
              >
                {isActive && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full bg-aura-400 shadow-[0_0_8px_rgba(77,124,255,0.6)]" />
                )}
                <item.icon className="w-4 h-4" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>

        {/* Bottom status */}
        <div className="px-4 py-3.5 border-t border-surface-800/40 space-y-2">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full transition-colors ${
                checking
                  ? "bg-surface-600 animate-pulse"
                  : backendConnected
                  ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]"
                  : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.7)]"
              }`}
            />
            <span className="text-xs text-surface-500">
              {checking
                ? "Connecting..."
                : backendConnected
                ? engineLoaded
                  ? "Engine Ready"
                  : "Connected — no model"
                : "Disconnected"}
            </span>
          </div>
          {backendConnected && engineLoaded && engineDevice && (
            <div className="flex items-center gap-1.5 text-[10px] text-surface-500">
              <span className={`chip ${
                engineDevice === "cuda"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : "bg-surface-800/60 text-surface-400 border-surface-700/50"
              }`}>
                {engineDevice === "cuda" ? "GPU accelerated" : "CPU mode"}
              </span>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>

      <Toaster />
    </div>
  );
}
