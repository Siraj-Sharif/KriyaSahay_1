import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import type { VoiceState } from "../services/types";

/**
 * Animated AI avatar. Canvas renders a particle orb that reacts to voice state
 * and the audio level (0..1) supplied by the VoiceAdapter.
 */
export function AIFace({ state, levelRef }: { state: VoiceState; levelRef: React.RefObject<number> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const N = 160;
    const particles = Array.from({ length: N }, (_, i) => ({
      a: (i / N) * Math.PI * 2,
      r: 0.72 + Math.random() * 0.28,
      sp: 0.2 + Math.random() * 0.6,
      ph: Math.random() * Math.PI * 2,
    }));
    let raf = 0;
    let t = 0;
    let smooth = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      t += 0.016;
      const { width: W, height: H } = canvas.getBoundingClientRect();
      if (canvas.width !== W * 2) {
        canvas.width = W * 2;
        canvas.height = H * 2;
      }
      ctx.setTransform(2, 0, 0, 2, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2;
      const cy = H / 2;
      const R = Math.min(W, H) * 0.3;
      const s = stateRef.current;
      const lvl = levelRef.current ?? 0;
      smooth += (lvl - smooth) * 0.2;

      const speed = s === "processing" ? 3 : s === "speaking" ? 1.4 : s === "listening" ? 0.8 : 0.3;
      const hue1 = s === "processing" ? "139,92,246" : s === "speaking" ? "34,211,238" : s === "listening" ? "59,130,246" : "34,211,238";

      // glow core
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * (1.3 + smooth * 0.4));
      g.addColorStop(0, `rgba(${hue1},${0.35 + smooth * 0.3})`);
      g.addColorStop(0.5, `rgba(${hue1},0.08)`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);

      // rings
      for (let k = 0; k < 3; k++) {
        ctx.beginPath();
        const rr = R * (0.95 + k * 0.18 + Math.sin(t * speed + k) * 0.03 + smooth * 0.12 * (k + 1));
        ctx.arc(cx, cy, rr, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${hue1},${0.25 - k * 0.07})`;
        ctx.lineWidth = 1;
        ctx.setLineDash(k === 1 ? [4, 10] : []);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // particles
      particles.forEach((p) => {
        const ang = p.a + t * speed * p.sp * (s === "processing" ? 1.5 : 0.4);
        const wobble = Math.sin(t * 3 + p.ph) * 0.05 + smooth * 0.35 * Math.sin(t * 8 + p.ph);
        const rad = R * (p.r + wobble);
        const x = cx + Math.cos(ang) * rad;
        const y = cy + Math.sin(ang) * rad * (s === "listening" ? 0.9 : 1);
        ctx.beginPath();
        ctx.arc(x, y, 1.2 + smooth * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${hue1},${0.5 + Math.sin(t * 2 + p.ph) * 0.3})`;
        ctx.fill();
      });

      // face: eyes
      const blink = Math.abs(Math.sin(t * 0.6)) > 0.985 ? 0.1 : 1;
      const eyeY = cy - R * 0.18;
      const eyeW = R * 0.16;
      const eyeH = R * (s === "processing" ? 0.08 : 0.22) * blink;
      [-1, 1].forEach((side) => {
        const ex = cx + side * R * 0.32;
        ctx.beginPath();
        ctx.ellipse(ex, eyeY, eyeW, eyeH, 0, 0, Math.PI * 2);
        ctx.fillStyle = "#e0fbff";
        ctx.shadowColor = `rgb(${hue1})`;
        ctx.shadowBlur = 18;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // mouth / waveform
      const my = cy + R * 0.3;
      ctx.beginPath();
      ctx.strokeStyle = "#e0fbff";
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.shadowColor = `rgb(${hue1})`;
      ctx.shadowBlur = 14;
      const mw = R * 0.6;
      if (s === "speaking" || s === "listening") {
        const bars = 14;
        for (let i = 0; i < bars; i++) {
          const x = cx - mw / 2 + (i / (bars - 1)) * mw;
          const env = Math.sin((i / (bars - 1)) * Math.PI);
          const h = (s === "speaking" ? 6 + Math.abs(Math.sin(t * 14 + i * 0.9)) * 26 * env * (0.5 + smooth) : 3 + smooth * 18 * env * Math.abs(Math.sin(t * 9 + i)));
          ctx.moveTo(x, my - h / 2);
          ctx.lineTo(x, my + h / 2);
        }
      } else if (s === "processing") {
        for (let i = 0; i < 3; i++) {
          const x = cx - 16 + i * 16;
          const yy = my + Math.sin(t * 8 - i * 0.8) * 5;
          ctx.moveTo(x, yy);
          ctx.lineTo(x + 0.01, yy);
        }
      } else {
        ctx.moveTo(cx - mw / 2, my);
        ctx.quadraticCurveTo(cx, my + R * 0.12, cx + mw / 2, my);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [levelRef]);

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} className="h-full w-full" />
      <motion.div
        key={state}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="pointer-events-none absolute bottom-6 left-1/2 -translate-x-1/2 rounded-full border border-white/10 bg-black/50 px-4 py-1.5 backdrop-blur"
      >
        <span className="mono text-[10px] uppercase tracking-[0.3em] text-cyan-200">{state}</span>
      </motion.div>
    </div>
  );
}
