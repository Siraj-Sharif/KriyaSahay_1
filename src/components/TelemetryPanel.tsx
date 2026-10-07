import { motion } from "framer-motion";
import { usePipelineState } from "../hooks/usePipelineState";
import { GlassCard, Icon, StatusDot } from "./ui";
import { cn } from "../utils/cn";

function Row({ k, v, accent, status }: { k: string; v: React.ReactNode; accent?: boolean; status?: "ok" | "warn" | "off" | "busy" }) {
  return (
    <div className="flex items-center justify-between border-b border-white/[0.04] py-2 last:border-0">
      <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">{k}</span>
      <span className={cn("mono flex items-center gap-2 text-[13px] font-medium", accent ? "text-cyan-300" : "text-slate-200")}>
        {status && <StatusDot status={status} />}
        {v}
      </span>
    </div>
  );
}

export function TelemetryPanel() {
  const {
    command,
    confidence,
    fps,
    latencyMs,
    handDetected,
    handCount,
    handedness,
    pipelineState,
    modelUsed,
    routingStr,
    safetyStatus,
    transportStatus,
    lastTx,
    serialPort,
    baudRate,
    cameraStatus,
    cameraResolution,
    landmarksCount,
    message,
    isStopActive,
    isAmbiguous,
  } = usePipelineState();

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {/* Computer vision */}
      <GlassCard title="Computer Vision" icon={<Icon.Eye />}>
        <div className="px-5 py-2">
          <Row k="Command" v={<motion.span key={command} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-glow">{handDetected ? command : "—"}</motion.span>} accent />
          <Row k="Confidence" v={handDetected ? `${(confidence * 100).toFixed(0)}%` : "--"} />
          <Row k="FPS" v={fps ?? "--"} />
          <Row k="Latency" v={latencyMs ? `${latencyMs}ms` : "--"} />
          <Row k="Hand" v={handDetected ? "DETECTED" : "NOT DETECTED"} status={handDetected ? "ok" : "warn"} />
          <Row k="Hands" v={handCount} />
          <Row k="Route" v={<span className="max-w-[170px] truncate text-right text-[11px]" title={routingStr}>{routingStr ?? "--"}</span>} />
          <Row k="Pipeline" v={pipelineState} />
          <Row k="Safety" v={safetyStatus} status={isStopActive ? "warn" : "ok"} />
        </div>
      </GlassCard>

      {/* Camera */}
      <GlassCard title="Camera" icon={<Icon.Camera />}>
        <div className="px-5 py-2">
          <Row k="Status" v={cameraStatus.toUpperCase()} status={cameraStatus === "connected" ? "ok" : "warn"} />
          <Row k="Resolution" v={cameraResolution} />
          <Row k="Landmarks" v={landmarksCount} />
          <Row k="Handedness" v={handedness ?? "—"} />
          <Row k="Pipeline" v="MediaPipe Hands (Python)" />
        </div>
      </GlassCard>

      {/* Hardware */}
      <GlassCard title="Hardware Link" icon={<Icon.Serial />}>
        <div className="px-5 py-2">
          <Row k="Serial" v={transportStatus} status={transportStatus.includes("CONNECTED") ? "ok" : transportStatus.includes("MOCK") ? "busy" : "off"} />
          <Row k="Port" v={serialPort} />
          <Row k="Baud" v={baudRate} />
          <Row k="TX" v={lastTx} status={lastTx !== "NONE" && lastTx !== "N/A" ? "busy" : "ok"} />
          <Row k="Last TX" v={<motion.span key={lastTx} initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} className="rounded bg-cyan-500/10 px-2 py-0.5 text-cyan-200 ring-1 ring-cyan-400/20">{lastTx ?? "--"}</motion.span>} />
          <Row k="Message" v={<span className="max-w-[170px] truncate text-right text-[11px]" title={message}>{message ?? "--"}</span>} />
        </div>
      </GlassCard>
    </div>
  );
}
