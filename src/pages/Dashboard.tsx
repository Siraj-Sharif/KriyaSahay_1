import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";
import { RoboticHand } from "../components/3D/RoboticHand";
import { CameraPanel } from "../components/CameraPanel";
import { SystemOverview } from "../components/SystemOverview";
import { TelemetryPanel } from "../components/TelemetryPanel";
import { ConfidenceChart } from "../components/charts/ConfidenceChart";
import { Corner, GlassCard, Icon } from "../components/ui";
import { GESTURE_DEFS, GESTURE_LIST } from "../services/gestures";
import { useSystem } from "../services/SystemContext";
import { usePipelineState, useConfidenceHistory, useGestureCounts } from "../hooks/usePipelineState";
import type { FingerPose } from "../services/types";
import { cn } from "../utils/cn";

const FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

export function Dashboard() {
  const { overrideGesture, setOverrideGesture } = useSystem();
  const pipelineState = usePipelineState();
  const history = useConfidenceHistory();
  const gestureCounts = useGestureCounts();
  const poseRef = useRef<FingerPose>([0, 0, 0, 0, 0, 0]);

  // feed the 3D hand from real pipeline state or manual override
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (overrideGesture) {
        poseRef.current = GESTURE_DEFS[overrideGesture]?.pose ?? [0.15, 0.15, 0.15, 0.15, 0.15, 0];
        return;
      }
      if (!pipelineState.handDetected) {
        poseRef.current = [0.15, 0.15, 0.15, 0.15, 0.15, 0];
        return;
      }
      poseRef.current = GESTURE_DEFS[pipelineState.gesture]?.pose ?? [0.15, 0.15, 0.15, 0.15, 0.15, 0];
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [pipelineState.handDetected, pipelineState.gesture, overrideGesture]);

  const activeGesture = overrideGesture ?? (pipelineState.handDetected ? pipelineState.gesture : null);
  const pose = activeGesture ? GESTURE_DEFS[activeGesture]?.pose ?? [0, 0, 0, 0, 0] : [0, 0, 0, 0, 0];
  const topGestures = [...GESTURE_LIST].sort((a, b) => (gestureCounts[b.id] || 0) - (gestureCounts[a.id] || 0)).slice(0, 5);

  return (
    <div className="flex flex-col gap-5">
      <SystemOverview />

      {/* Primary split */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <GlassCard className="lg:col-span-3" strong>
          <CameraPanel />
        </GlassCard>

        <GlassCard className="relative min-h-[520px] lg:col-span-2" strong>
          <div className="absolute inset-0 grid-bg opacity-60 [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_60%,rgba(139,92,246,0.18),transparent_55%)]" />
          <Corner />
          <div className="absolute left-5 top-4 z-10 flex items-center gap-2">
            <Icon.Hand className="h-4 w-4 text-cyan-300/80" />
            <h3 className="mono text-[11px] font-medium uppercase tracking-[0.2em] text-slate-400">Digital Twin · NG-1</h3>
          </div>
          <div className="absolute right-5 top-4 z-10 text-right">
            <div className="mono text-[9px] uppercase tracking-[0.3em] text-slate-500">
              {overrideGesture ? "Override" : "Mirroring"}
            </div>
            <AnimatePresence mode="wait">
              <motion.div key={activeGesture ?? "x"} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mono text-sm font-semibold text-cyan-200">
                {activeGesture ?? "IDLE"}
              </motion.div>
            </AnimatePresence>
          </div>

          <RoboticHand poseRef={poseRef} className="absolute inset-0" />

          {/* finger curl readout */}
          <div className="absolute bottom-4 left-5 right-5 z-10 grid grid-cols-5 gap-2">
            {FINGER_NAMES.map((n, i) => (
              <div key={n} className="rounded-lg border border-white/5 bg-black/50 px-2 py-1.5 backdrop-blur">
                <div className="mono text-[9px] uppercase tracking-widest text-slate-500">{n}</div>
                <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-white/10">
                  <motion.div className="h-full bg-gradient-to-r from-cyan-400 to-violet-400" animate={{ width: `${pose[i] * 100}%` }} transition={{ type: "spring", stiffness: 100, damping: 18 }} />
                </div>
                <div className="mono mt-1 text-[10px] text-slate-300">{Math.round(pose[i] * 180)}°</div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Telemetry */}
      <TelemetryPanel />

      {/* Control + chart */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <GlassCard title="Confidence History" icon={<Icon.Chart />} className="xl:col-span-2" right={<span className="mono text-[10px] text-slate-500">120 samples · 4 Hz</span>}>
          <ConfidenceChart data={history} />
        </GlassCard>

        <div className="flex flex-col gap-5">
          <GlassCard title="Manual Override" icon={<Icon.Hand />} right={
            overrideGesture && (
              <button onClick={() => setOverrideGesture(null)} className="mono rounded-md bg-rose-500/15 px-2 py-0.5 text-[10px] uppercase tracking-wider text-rose-300 ring-1 ring-rose-400/30 hover:bg-rose-500/25">
                Release
              </button>
            )
          }>
            <div className="grid grid-cols-4 gap-1.5 p-3">
              {GESTURE_LIST.map((g) => (
                <button
                  key={g.id}
                  onClick={() => setOverrideGesture(overrideGesture === g.id ? null : g.id)}
                  title={g.id}
                  className={cn(
                    "flex flex-col items-center gap-1 rounded-lg border px-1 py-2 transition",
                    overrideGesture === g.id
                      ? "border-cyan-400/50 bg-cyan-500/15 shadow-[0_0_16px_rgba(34,211,238,0.25)]"
                      : "border-white/5 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.05]",
                  )}
                >
                  <span className="text-lg leading-none">{g.emoji}</span>
                  <span className="mono w-full truncate text-center text-[8px] uppercase tracking-wider text-slate-400">{g.id.replace(/_/g, " ")}</span>
                </button>
              ))}
            </div>
          </GlassCard>

          <GlassCard title="Session Activity" icon={<Icon.Chart />}>
            <div className="flex flex-col gap-2 p-4">
              {topGestures.map((g) => {
                const max = Math.max(1, ...Object.values(gestureCounts));
                return (
                  <div key={g.id} className="flex items-center gap-3">
                    <span className="mono w-28 truncate text-[10px] uppercase tracking-wider text-slate-400">{g.id}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                      <motion.div className="h-full bg-gradient-to-r from-violet-500 to-cyan-400" animate={{ width: `${(gestureCounts[g.id] / max) * 100}%` }} />
                    </div>
                    <span className="mono w-6 text-right text-[11px] text-slate-300">{gestureCounts[g.id]}</span>
                  </div>
                );
              })}
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
