import { motion } from "framer-motion";
import { Icon, StatusDot } from "./ui";
import { cn } from "../utils/cn";
import { usePipelineState } from "../hooks/usePipelineState";
import { useDesktopRuntime } from "../hooks/useDesktopRuntime";

export type Tab = "dashboard" | "gestures" | "hardware" | "voice";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "dashboard", label: "Dashboard", icon: <Icon.Grid /> },
  { id: "gestures", label: "Gesture Guide", icon: <Icon.Hand /> },
  { id: "hardware", label: "Hardware", icon: <Icon.Chip /> },
  { id: "voice", label: "Voice Assistant", icon: <Icon.Mic /> },
];

export function Navbar({
  tab,
  onTab,
  isFullscreen,
  onToggleFullscreen,
}: {
  tab: Tab;
  onTab: (t: Tab) => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
}) {
  const pipelineState = usePipelineState();
  const runtime = useDesktopRuntime();
  const isDesktop = Boolean(window.neurogrip?.isDesktop);
  const backendReady = isDesktop && runtime?.backend === "ready";
  const backendStatus = !isDesktop ? "busy" : backendReady ? "ok" : runtime?.backend === "error" || runtime?.backend === "stopped" ? "off" : "busy";
  const backendLabel = !isDesktop
    ? "Browser demo"
    : backendReady
      ? "Pipeline ready"
      : runtime?.backend === "error" || runtime?.backend === "stopped"
        ? "Backend error"
        : "Starting Python…";
  return (
    <header className="relative z-40 shrink-0 border-b border-white/5 bg-black/40 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1700px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-violet-500/20 ring-1 ring-cyan-400/30">
            <Icon.Hand className="h-5 w-5 text-cyan-300" />
            <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_10px_#22d3ee]" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide text-white">
              Kriya <span className="text-cyan-300">Sahay</span>
            </div>
            <div className="mono text-[10px] uppercase tracking-[0.25em] text-slate-500">Robotics Control System</div>
          </div>
        </div>

        <nav className="hidden items-center gap-1 rounded-full border border-white/5 bg-white/[0.02] p-1 md:flex">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onTab(t.id)}
              className={cn(
                "relative flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium transition-colors",
                tab === t.id ? "text-white" : "text-slate-400 hover:text-slate-200",
              )}
            >
              {tab === t.id && (
                <motion.span
                  layoutId="tab-pill"
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-500/20 to-violet-500/20 ring-1 ring-cyan-400/30"
                  transition={{ type: "spring", stiffness: 400, damping: 32 }}
                />
              )}
              <span className="relative">{t.icon}</span>
              <span className="relative">{t.label}</span>
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-3 lg:flex">
            <StatusDot status={backendStatus} label={backendLabel} />
            {!isDesktop ? (
              <StatusDot status="busy" label="Simulated CV" />
            ) : (
              <StatusDot
                status={!backendReady ? "busy" : pipelineState.cameraStatus === "connected" ? (pipelineState.handDetected ? "ok" : "warn") : "off"}
                label={!backendReady ? "Camera pending" : pipelineState.cameraStatus === "connected" ? (pipelineState.handDetected ? "Tracking" : "No hand") : pipelineState.cameraStatus === "stopped" ? "Camera paused" : "Camera unavailable"}
              />
            )}
            {/* Serial status comes from the Python pipeline; browser adapters are never shown as real hardware. */}
            {!isDesktop ? (
              <StatusDot status="busy" label="No hardware · demo" />
            ) : (
              <StatusDot
                status={!backendReady ? "busy" : !pipelineState.serialConnected ? "off" : pipelineState.serialMode === "real" ? "ok" : "busy"}
                label={
                  !backendReady
                    ? "Serial pending"
                    : !pipelineState.serialConnected
                      ? "Serial disconnected"
                      : pipelineState.serialMode === "real"
                        ? pipelineState.serialPort
                        : "Mock serial"
                }
              />
            )}
          </div>
          <button
            onClick={onToggleFullscreen}
            title="Toggle fullscreen (F11)"
            className="flex h-8 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 text-xs text-slate-300 transition hover:border-cyan-400/40 hover:text-cyan-200"
          >
            {isFullscreen ? <Icon.Compress /> : <Icon.Expand />}
            <span className="mono hidden text-[10px] uppercase tracking-wider sm:inline">F11</span>
          </button>
        </div>
      </div>
      {/* mobile tabs */}
      <nav className="flex gap-1 overflow-x-auto px-4 pb-2 md:hidden">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => onTab(t.id)}
            className={cn(
              "flex shrink-0 items-center gap-2 rounded-full px-3 py-1.5 text-xs",
              tab === t.id ? "bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-400/30" : "text-slate-400",
            )}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
