import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { usePipelineState } from "../hooks/usePipelineState";
import { useDesktopRuntime } from "../hooks/useDesktopRuntime";
import { HAND_CONNECTIONS, GESTURE_DEFS } from "../services/gestures";
import { Corner, Icon, StatusDot } from "./ui";

/**
 * Live CV panel. Shows real-time pipeline state from Python bridge.
 * Camera feed is owned by Python pipeline; we show landmarks overlay from real data.
 * Does NOT open a second camera stream.
 */
export function CameraPanel() {
  const pipelineState = usePipelineState();
  const runtime = useDesktopRuntime();
  const isDesktop = Boolean(window.neurogrip?.isDesktop);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // Receive real camera frames from Python pipeline via Electron bridge
  useEffect(() => {
    if (typeof window === "undefined" || !(window as any).neurogrip?.onCameraFrame) return;
    const unsub = (window as any).neurogrip.onCameraFrame((msg: { data: string }) => {
      if (msg && msg.data) {
        if (imgRef.current) {
          imgRef.current.src = "data:image/jpeg;base64," + msg.data;
        }
        setHasFrame(true);
      }
    });
    return () => unsub();
  }, []);

  const landmarksRef = useRef<{ x: number; y: number; z: number }[]>([]);

  // Phase 2: the overlay draws the landmarks the pipeline actually detected (normalised
  // image coordinates) instead of an always-empty array.
  useEffect(() => {
    landmarksRef.current = pipelineState.handDetected ? pipelineState.landmarks : [];
  }, [pipelineState.landmarks, pipelineState.handDetected]);

  // Landmark overlay render loop - uses real landmarks from pipeline if available
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { width: W, height: H } = canvas.getBoundingClientRect();
      if (canvas.width !== W * 2 || canvas.height !== H * 2) {
        canvas.width = W * 2;
        canvas.height = H * 2;
      }
      ctx.setTransform(2, 0, 0, 2, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // Draw landmarks if we have them (21 points from MediaPipe)
      const pts = landmarksRef.current;
      if (!pts || pts.length < 21) return;

      const points = pts.map((l) => [l.x * W, l.y * H] as const);

      // bounding box
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      const pad = 18;
      const bx = Math.min(...xs) - pad;
      const by = Math.min(...ys) - pad;
      const bw = Math.max(...xs) - bx + pad;
      const bh = Math.max(...ys) - by + pad;
      ctx.strokeStyle = "rgba(34,211,238,0.5)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);
      ctx.font = "600 11px JetBrains Mono, monospace";
      ctx.fillStyle = "rgba(34,211,238,0.95)";
      ctx.fillText(`${pipelineState.handedness?.toUpperCase() || "HAND"} · ${(pipelineState.confidence * 100).toFixed(0)}%`, bx, by - 6);

      // connections
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      for (const [a, b] of HAND_CONNECTIONS) {
        const grad = ctx.createLinearGradient(points[a][0], points[a][1], points[b][0], points[b][1]);
        grad.addColorStop(0, "rgba(34,211,238,0.9)");
        grad.addColorStop(1, "rgba(139,92,246,0.9)");
        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(points[a][0], points[a][1]);
        ctx.lineTo(points[b][0], points[b][1]);
        ctx.stroke();
      }
      // joints
      points.forEach(([x, y], i) => {
        const tip = [4, 8, 12, 16, 20].includes(i);
        ctx.beginPath();
        ctx.arc(x, y, tip ? 4.5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = tip ? "#ffffff" : "#22d3ee";
        ctx.shadowColor = "#22d3ee";
        ctx.shadowBlur = tip ? 12 : 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      });
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [pipelineState.handedness, pipelineState.confidence]);

  const detected = pipelineState.handDetected;
  const def = detected ? GESTURE_DEFS[pipelineState.gesture] : null;
  const pipelineFrameAvailable = pipelineState.pipelineState !== "STARTING";
  const snapshotCamera = runtime?.snapshot?.camera;
  const cameraEnabled = pipelineFrameAvailable ? pipelineState.cameraEnabled : Boolean(snapshotCamera?.enabled);
  const cameraStatus = pipelineFrameAvailable
    ? pipelineState.cameraStatus
    : !snapshotCamera ? "disconnected" : !snapshotCamera.enabled ? "stopped" : snapshotCamera.opened ? "connected" : "disconnected";
  const cameraError = pipelineFrameAvailable ? pipelineState.cameraError : snapshotCamera?.error ?? null;
  const cameraIndex = pipelineFrameAvailable ? pipelineState.cameraIndex : snapshotCamera?.index ?? 0;
  const cameraResolution = pipelineFrameAvailable ? pipelineState.cameraResolution : snapshotCamera?.resolution ?? "--";
  const cvReady = runtime?.cv === "ready";
  const cameraReady = isDesktop && runtime?.backend === "ready" && cvReady && cameraEnabled && cameraStatus === "connected";
  const overlayTitle = !isDesktop
    ? "Python camera preview unavailable"
    : runtime?.backend === "error" || runtime?.backend === "stopped"
      ? "Python backend unavailable"
      : runtime?.backend !== "ready"
        ? "Waiting for Python backend"
        : runtime?.cv === "degraded"
          ? "CV detector unavailable"
          : cameraStatus === "stopped" || !cameraEnabled
            ? "Capture paused"
            : cameraStatus === "error" || cameraStatus === "disconnected"
              ? "Camera unavailable"
              : "Waiting for live frames";
  const overlayDetail = !isDesktop
    ? "Browser preview mode does not connect a camera to the Python CV pipeline."
    : runtime?.cv === "degraded"
      ? runtime.message
      : cameraError || (cameraReady ? "Waiting for the next real frame from Python." : runtime?.message || "The preview appears only after Python sends a real camera frame.");

  useEffect(() => {
    if (cameraReady) return;
    setHasFrame(false);
    if (imgRef.current) imgRef.current.removeAttribute("src");
  }, [cameraReady]);

  return (
    <div className="relative flex h-full flex-col">
      {/* Header strip */}
      <div className="flex items-center justify-between border-b border-white/5 px-5 py-3">
        <div className="flex items-center gap-3">
          <Icon.Camera className="h-4 w-4 text-cyan-300/80" />
          <h3 className="mono text-[11px] font-medium uppercase tracking-[0.2em] text-slate-400">Live Vision Feed</h3>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-4 2xl:flex">
            <StatusDot status={cameraReady ? "ok" : "warn"} label={cameraReady ? `Camera ${cameraIndex}` : overlayTitle} />
            <StatusDot status={cameraReady && detected ? "ok" : "warn"} label={cameraReady ? (detected ? "Tracking" : "No hand") : "CV idle"} />
          </div>
        </div>
      </div>

      {/* Feed container */}
      <div className="relative aspect-video w-full overflow-hidden bg-[#05070c] scanline">
        {/* Real Python camera stream */}
        <img
          ref={imgRef}
          alt="Python OpenCV Feed"
          className="absolute inset-0 h-full w-full object-cover z-0"
          style={{ display: hasFrame ? "block" : "none" }}
        />

        <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(34,211,238,0.08),transparent_60%)] pointer-events-none" />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none z-10" />
        <Corner />

        {/* The overlay never invents a camera frame: it distinguishes boot, pause and device failure. */}
        {(!cameraReady || !hasFrame) && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-[#03050a]/92 px-6 text-center">
            <div className="relative flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/[0.03]">
              <Icon.Camera className={`h-7 w-7 ${cameraReady ? "animate-pulse text-cyan-400" : "text-slate-500"}`} />
            </div>
            <div className="relative">
              <div className="mono text-sm font-semibold uppercase tracking-[0.22em] text-slate-300">{overlayTitle}</div>
              <div className="mx-auto mt-2 max-w-[360px] text-xs leading-relaxed text-slate-500">{overlayDetail}</div>
              {cameraReady && !hasFrame && <div className="mono mt-3 text-[9px] uppercase tracking-wider text-cyan-300/70">Awaiting Python JPEG preview</div>}
            </div>
          </div>
        )}

        {/* HUD top-left */}
        <div className="absolute left-4 top-4 flex flex-col gap-1">
          <div className="flex items-center gap-2 rounded-md bg-black/50 px-2 py-1 backdrop-blur">
            <span className={`h-1.5 w-1.5 rounded-full ${detected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
            <span className="mono text-[10px] uppercase tracking-widest text-slate-200">
              {cameraReady ? (detected ? "TRACKING" : "NO HAND") : overlayTitle.toUpperCase()} · {cameraResolution}
            </span>
          </div>
          <span className="mono text-[10px] text-slate-500">{isDesktop ? `Python OpenCV · ${pipelineState.modelUsed}` : "Browser preview · no Python feed"}</span>
        </div>

        {/* HUD top-right metrics */}
        <div className="absolute right-4 top-4 flex gap-2">
          {[
            ["FPS", pipelineState.fps ?? "--"],
            ["LAT", pipelineState.latencyMs ? `${pipelineState.latencyMs}ms` : "--"],
            ["CONF", detected ? `${(pipelineState.confidence * 100).toFixed(0)}%` : "--"],
          ].map(([k, v]) => (
            <div key={k as string} className="rounded-md bg-black/50 px-2 py-1 text-right backdrop-blur">
              <div className="mono text-[9px] tracking-widest text-slate-500">{k}</div>
              <div className="mono text-xs font-semibold text-cyan-200">{v}</div>
            </div>
          ))}
        </div>

        {/* Gesture readout */}
        <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
          <AnimatePresence mode="wait">
            <motion.div
              key={detected ? pipelineState.gesture : "none"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="rounded-xl border border-cyan-400/20 bg-black/60 px-4 py-3 backdrop-blur-md"
            >
              <div className="mono text-[9px] uppercase tracking-[0.3em] text-slate-500">Detected gesture</div>
              <div className="mt-1 flex items-center gap-3">
                <span className="text-2xl">{detected && def ? def.emoji : "⌛"}</span>
                <div>
                  <div className="mono text-xl font-bold tracking-wide text-white text-glow">{detected ? pipelineState.gesture : "NO HAND"}</div>
                  <div className="text-[11px] text-slate-400">{detected && def ? def.action : "Place hand in frame"}</div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
          <div className="rounded-lg bg-black/50 px-3 py-2 text-right backdrop-blur">
            <div className="mono text-[9px] tracking-widest text-slate-500">HANDS</div>
            <div className="mono text-sm font-semibold text-slate-200">{pipelineState.handCount} / 2</div>
          </div>
        </div>
      </div>

      {/* Real pipeline notice: camera unavailable / invalid / permission errors */}
      {cameraError && (
        <div className="border-t border-rose-400/20 bg-rose-500/[0.07] px-5 py-2">
          <span className="mono text-[10px] uppercase tracking-wider text-rose-300">Camera</span>
          <span className="ml-2 text-[11px] text-rose-200">{cameraError}</span>
        </div>
      )}

      {/* Confidence bar */}
      <div className="px-5 py-3">
        <div className="flex items-center justify-between">
          <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">Classifier confidence</span>
          <span className="mono text-[11px] text-cyan-300">{detected ? `${(pipelineState.confidence * 100).toFixed(1)}%` : "--"}</span>
        </div>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-500 shadow-[0_0_12px_rgba(34,211,238,0.6)]"
            animate={{ width: `${detected ? pipelineState.confidence * 100 : 0}%` }}
            transition={{ type: "spring", stiffness: 120, damping: 20 }}
          />
        </div>
      </div>
    </div>
  );
}
