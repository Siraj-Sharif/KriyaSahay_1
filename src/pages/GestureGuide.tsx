import { AnimatePresence, motion } from "framer-motion";
import { useRef, useState } from "react";
import { RoboticHand } from "../components/3D/RoboticHand";
import { HandPoseSVG } from "../components/HandPoseSVG";
import { Corner, GlassCard, Icon } from "../components/ui";
import { GESTURE_DEFS, GESTURE_LIST, encodeCommand } from "../services/gestures";
import { useSystem } from "../services/SystemContext";
import { usePipelineState } from "../hooks/usePipelineState";
import type { FingerPose, Gesture } from "../services/types";
import { cn } from "../utils/cn";

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

export function GestureGuide() {
  const { frame, gestureCounts } = useSystem();
  const pipelineState = usePipelineState();
  const [selected, setSelected] = useState<Gesture>("GRIP");
  const poseRef = useRef<FingerPose>(GESTURE_DEFS.GRIP.pose);
  const def = GESTURE_DEFS[selected];
  poseRef.current = def.pose;

  const select = (g: Gesture) => {
    setSelected(g);
  };

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
      {/* Grid */}
      <div className="xl:col-span-2">
        <div className="mb-4 flex items-end justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Gesture Command Set</h2>
            <p className="text-sm text-slate-400">13 trained gestures recognised by the NeuroGrip classifier. Click a card to preview it on the 3D hand.</p>
          </div>
          <span className="mono text-[10px] uppercase tracking-[0.25em] text-slate-500">{GESTURE_LIST.length} commands</span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
          {GESTURE_LIST.map((g, i) => {
            const live = (pipelineState.handDetected && pipelineState.gesture === g.id) || (frame?.handDetected && frame.gesture === g.id);
            const active = selected === g.id;
            return (
              <motion.button
                key={g.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
                whileHover={{ y: -3 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => select(g.id)}
                className={cn(
                  "group relative flex flex-col items-center overflow-hidden rounded-2xl border p-4 text-left transition",
                  active ? "glass-strong border-cyan-400/40" : "glass hover:border-white/15",
                )}
              >
                {live && (
                  <span className="absolute right-3 top-3 flex items-center gap-1 rounded-full bg-emerald-500/15 px-2 py-0.5 ring-1 ring-emerald-400/40">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                    <span className="mono text-[9px] uppercase tracking-wider text-emerald-300">Live</span>
                  </span>
                )}
                <span className="absolute left-3 top-3 mono text-[9px] text-slate-600">{String(i + 1).padStart(2, "0")}</span>
                <div className="mt-3 rounded-xl bg-black/30 p-2">
                  <HandPoseSVG pose={g.pose} size={84} active={active || !!live} />
                </div>
                <div className="mt-3 w-full text-center">
                  <div className="mono text-[11px] font-semibold tracking-wider text-white">{g.id}</div>
                  <div className="mt-0.5 truncate text-[11px] text-slate-500">{g.action}</div>
                </div>
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-cyan-500/10 to-transparent opacity-0 transition group-hover:opacity-100" />
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Detail */}
      <div className="flex flex-col gap-4 xl:sticky xl:top-24 xl:self-start">
        <GlassCard strong className="relative min-h-[360px]">
          <Corner />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_60%,rgba(34,211,238,0.12),transparent_55%)]" />
          <div className="absolute left-5 top-4 z-10">
            <div className="mono text-[9px] uppercase tracking-[0.3em] text-slate-500">Preview pose</div>
            <AnimatePresence mode="wait">
              <motion.div key={selected} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mono text-lg font-bold text-cyan-200 text-glow">
                {selected}
              </motion.div>
            </AnimatePresence>
          </div>
          <span className="absolute right-5 top-4 z-10 text-3xl">{def.emoji}</span>
          <RoboticHand poseRef={poseRef} className="h-[360px] w-full" />
        </GlassCard>

        <GlassCard title={def.label} icon={<Icon.Hand />} right={<span className="mono text-[10px] text-slate-500">seen ×{gestureCounts[selected]}</span>}>
          <div className="flex flex-col gap-4 p-5">
            <p className="text-sm leading-relaxed text-slate-300">{def.description}</p>
            <div className="grid grid-cols-5 gap-2">
              {FINGERS.map((f, i) => (
                <div key={f} className="rounded-lg border border-white/5 bg-white/[0.02] p-2 text-center">
                  <div className="mono text-[8px] uppercase tracking-widest text-slate-500">{f}</div>
                  <div className="mono mt-1 text-xs font-semibold text-slate-100">{i === 0 && def.pose[5] > 0.5 ? "TOUCH" : def.pose[i] >= 0.9 ? "CURL" : def.pose[i] <= 0.1 ? "OPEN" : "HALF"}</div>
                  <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/5">
                    <motion.div className="h-full bg-cyan-400" animate={{ width: `${(i === 0 && def.pose[5] > 0.5 ? 0.5 : def.pose[i]) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between rounded-lg border border-white/5 bg-black/30 px-3 py-2">
              <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Serial packet</span>
              <span className="mono text-xs text-cyan-200">{encodeCommand(selected)}</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border border-white/5 bg-black/30 px-3 py-2">
              <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Mapped action</span>
              <span className="text-xs text-slate-200">{def.action}</span>
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
