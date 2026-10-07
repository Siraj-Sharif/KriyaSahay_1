import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Icon, StatusDot } from "./ui";
import { cn } from "../utils/cn";
import { useSystem } from "../services/SystemContext";
import { usePipelineState } from "../hooks/usePipelineState";

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
  const { frame, serial } = useSystem();
  const pipelineState = usePipelineState();
  const [bridgeConnected, setBridgeConnected] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined" && (window as any).neurogrip?.onBridgeStatus) {
      const unsub = (window as any).neurogrip.onBridgeStatus((s: { connected: boolean }) => {
        setBridgeConnected(Boolean(s && s.connected));
      });
      return () => unsub();
    }
  }, []);

  const isBackendActive = bridgeConnected || pipelineState.fps > 0 || pipelineState.pipelineState !== "STARTING";
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
              Neuro<span className="text-cyan-300">Grip</span>
            </div>
            <div className="mono text-[10px] uppercase tracking-[0.25em] text-slate-500">Robotic Hand Control</div>
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
            <StatusDot
              status={isBackendActive ? "ok" : "off"}
              label={isBackendActive ? "Backend Live" : "Connecting..."}
            />
            <StatusDot
              status={pipelineState.handDetected ? "ok" : pipelineState.cameraStatus === "connected" ? "warn" : "off"}
              label={pipelineState.handDetected ? (pipelineState.gesture || "Tracking") : pipelineState.cameraStatus === "connected" ? "No Hand" : "Camera Off"}
            />
            {/* Serial indicator from the pipeline's own `serial_info`, not a string guess. */}
            <StatusDot
              status={!pipelineState.serialConnected ? "off" : pipelineState.serialMode === "real" ? "ok" : "busy"}
              label={
                !pipelineState.serialConnected
                  ? "Serial Off"
                  : pipelineState.serialMode === "real"
                    ? pipelineState.serialPort
                    : "Mock Serial"
              }
            />
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
