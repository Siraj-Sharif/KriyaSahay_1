import { useEffect, useState } from "react";
import type { EngineStatus, VoiceState } from "../services/types";
import { useDesktopRuntime } from "../hooks/useDesktopRuntime";
import { useMicDevices } from "../hooks/useMicDevices";
import { usePipelineState } from "../hooks/usePipelineState";
import { useSystem } from "../services/SystemContext";
import { setStopArmed } from "../services/pipelineControl";
import { Icon, StatusDot } from "./ui";
import { CameraSelector } from "./CameraSelector";
import { cn } from "../utils/cn";

type Tone = "ok" | "warn" | "off" | "busy";

function Tile({
  title,
  icon,
  status,
  detail,
  meta,
  action,
}: {
  title: string;
  icon: React.ReactNode;
  status: string;
  detail: string;
  meta?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="relative flex min-h-[146px] flex-col rounded-2xl border border-white/[0.07] bg-[#0a0e17]/85 p-4 shadow-[0_12px_30px_rgba(0,0,0,0.18)]">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-cyan-400/15 bg-cyan-400/[0.07] text-cyan-200">{icon}</span>
          <div className="min-w-0">
            <div className="mono text-[9px] uppercase tracking-[0.22em] text-slate-500">{title}</div>
            <div className="mt-1 truncate text-[13px] font-semibold text-slate-100" title={status}>{status}</div>
          </div>
        </div>
        <span className="shrink-0 pt-1"><StatusDot status={statusTone(status)} /></span>
      </div>
      <p className="mt-3 line-clamp-2 min-h-8 text-[11px] leading-relaxed text-slate-400">{detail}</p>
      {(meta || action) && (
        <div className="mt-auto flex items-end justify-between gap-2 pt-2">
          <div className="min-w-0">{meta}</div>
          {action}
        </div>
      )}
    </section>
  );
}

function statusTone(label: string): Tone {
  const value = label.toLowerCase();
  if (/ready|connected|armed|confirmed|tracking/.test(value)) return "ok";
  if (/starting|waiting|awaiting|loading|paused|demo|simulat|optional|permitted/.test(value)) return "busy";
  if (/degraded|disconnected|unavailable|not reported|not verified|no device|no tx|disarmed|error|not linked/.test(value)) return "warn";
  return "off";
}

const buttonClass = "rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[10px] font-medium text-slate-200 transition hover:border-cyan-300/40 hover:bg-cyan-400/10 disabled:cursor-not-allowed disabled:opacity-40";

export function SystemOverview() {
  const pipeline = usePipelineState();
  const runtime = useDesktopRuntime();
  const { backendControlled, setCameraOn, voice, cameraNotice } = useSystem();
  const microphones = useMicDevices();
  const isDesktop = Boolean(window.neurogrip?.isDesktop);
  const snapshot = runtime?.snapshot;
  const pipelineFrameAvailable = backendControlled && pipeline.pipelineState !== "STARTING";
  const pipelineReported = backendControlled && runtime?.backend === "ready" && runtime.bridge === "connected" && (pipelineFrameAvailable || Boolean(snapshot?.initialized && snapshot.running));
  const cameraIndex = pipelineFrameAvailable ? pipeline.cameraIndex : snapshot?.camera?.index ?? 0;
  const cameraResolution = pipelineFrameAvailable ? pipeline.cameraResolution : snapshot?.camera?.resolution ?? "--";
  const cameraError = pipelineFrameAvailable ? pipeline.cameraError : snapshot?.camera?.error ?? null;
  const cameraStatus = pipelineFrameAvailable
    ? pipeline.cameraStatus
    : !snapshot?.camera ? "disconnected" : !snapshot.camera.enabled ? "stopped" : snapshot.camera.opened ? "connected" : "disconnected";
  const cameraLive = pipelineReported && runtime?.cv === "ready" && cameraStatus === "connected";
  const serialMode = pipelineFrameAvailable ? pipeline.serialMode : snapshot?.serial?.mode ?? "disabled";
  const serialConnected = pipelineFrameAvailable ? pipeline.serialConnected : Boolean(snapshot?.serial?.connected);
  const serialPort = pipelineFrameAvailable ? pipeline.serialPort : snapshot?.serial?.port || "--";
  const serialBaud = pipelineFrameAvailable ? pipeline.baudRate : snapshot?.serial?.baud ?? 115200;
  const framesSent = pipelineFrameAvailable ? pipeline.framesSent : snapshot?.serial?.frames_sent ?? 0;
  const lastTx = pipelineFrameAvailable ? pipeline.lastTx : snapshot?.serial?.last_tx ?? "NONE";
  const isArmed = pipelineFrameAvailable ? pipeline.isArmed : Boolean(snapshot?.stop_armed);
  const safetyStatus = pipelineFrameAvailable ? pipeline.safetyStatus : snapshot?.stop_armed ? "STOP ARMED" : "STOP DISARMED";
  const serialLive = pipelineReported && serialMode === "real" && serialConnected;
  const coreReady = Boolean(runtime?.backend === "ready" && runtime?.cv === "ready" && cameraLive && serialLive);
  const fatal = runtime?.backend === "error" || runtime?.backend === "stopped" || runtime?.bridge === "error" || runtime?.cv === "error";

  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [engine, setEngine] = useState<EngineStatus>({
    phase: isDesktop ? "loading" : "idle",
    progress: 0,
    message: isDesktop ? "Waiting for voice engine" : "Browser preview",
  });
  const [busySafety, setBusySafety] = useState(false);
  const [busyCamera, setBusyCamera] = useState(false);
  const [busyRestart, setBusyRestart] = useState(false);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    const stopState = voice.subscribe((state) => setVoiceState(state));
    const stopEngine = voice.subscribeEngine(setEngine);
    return () => {
      stopState();
      stopEngine();
    };
  }, [voice]);

  const healthLabel = !isDesktop
    ? "Browser demo"
    : fatal
      ? "Runtime error"
      : !runtime || runtime.backend === "starting" || runtime.backend === "stopping" || !pipelineReported
        ? "Starting"
        : runtime.cv === "degraded"
          ? "Degraded · CV unavailable"
          : coreReady
            ? "System ready"
            : "Degraded · device missing";
  const healthTone: Tone = !isDesktop ? "busy" : fatal ? "off" : coreReady ? "ok" : runtime?.backend === "ready" ? "warn" : "busy";
  const healthDetail = !isDesktop
    ? "Browser preview only. Gesture and serial adapters are simulated here; no command can reach a robot."
    : fatal
      ? runtime?.message || "The Python backend or local bridge failed to start. Retry the backend after resolving the error."
      : !pipelineReported
        ? runtime?.message || "Waiting for the Electron bridge to confirm an initialized Python pipeline."
        : runtime?.cv === "degraded"
          ? runtime?.message || "CV detector unavailable; camera tracking is disabled."
          : coreReady
          ? "Python is live, the camera is open, and a real serial link is connected. Voice remains optional."
          : `Python is live, but ${[
              !cameraLive ? "camera capture" : "",
              !serialLive ? "a real serial device link" : "",
            ].filter(Boolean).join(" and ")} is not confirmed. Controls fail safely until the device is available.`;

  const setCamera = async () => {
    setBusyCamera(true);
    setNotice("");
    try {
      // A failed camera remains retryable; only an actually open camera is paused.
      await setCameraOn(cameraLive ? false : true);
    } finally {
      setBusyCamera(false);
    }
  };

  const toggleSafety = async () => {
    if (!pipelineReported) return;
    setBusySafety(true);
    setNotice("");
    try {
      const res = await setStopArmed(!isArmed);
      if (!res.ok) setNotice(res.error || "STOP safety request was rejected by Python.");
    } finally {
      setBusySafety(false);
    }
  };

  const restartBackend = async () => {
    if (!window.neurogrip?.restartBackend) return;
    setBusyRestart(true);
    setNotice("");
    try {
      const res = await window.neurogrip.restartBackend();
      if (!res.ok) setNotice(res.error || "Backend restart failed.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyRestart(false);
    }
  };

  const cameraLabel = !isDesktop
    ? "Simulation only"
    : !pipelineReported
      ? "Awaiting pipeline"
      : runtime?.cv !== "ready"
        ? "CV detector unavailable"
        : cameraLive
          ? `Camera ${cameraIndex} connected`
          : cameraStatus === "stopped"
            ? "Capture paused"
            : cameraStatus === "error"
              ? "Camera unavailable"
              : "Camera not opened";
  const cameraDetail = !isDesktop
    ? "No browser camera stream is connected to the Python CV pipeline."
    : runtime?.cv === "degraded"
      ? runtime.message
      : cameraLive
        ? `${cameraResolution} · ${pipelineFrameAvailable ? pipeline.fps || "--" : "--"} FPS · ${pipelineFrameAvailable ? pipeline.modelUsed : "CV model ready"}`
        : cameraError || (cameraStatus === "stopped" ? "Capture is stopped; manual and voice command paths remain available." : "Select a camera reported by Python, then retry capture.");

  const serialLabel = !isDesktop
    ? "Simulation only"
    : !pipelineReported
      ? "Awaiting pipeline"
      : serialMode === "mock"
        ? "Mock transport · no hardware"
        : serialLive
          ? "Serial link open"
          : serialMode === "disabled"
            ? "Serial disabled"
            : "Serial disconnected";
  const serialDetail = !isDesktop
    ? "The browser adapter cannot open a hardware port or transmit to an ESP32."
    : serialLive
      ? `${serialPort} @ ${serialBaud} baud · device identity is not reported by the current protocol.`
      : serialMode === "mock"
        ? "Mock mode is explicitly simulated and does not confirm a physical device."
        : "Choose a discovered port in Hardware and connect through the Python-owned serial interface.";

  const micReady = microphones.inputsFound > 0;
  const voiceLabel = voiceState !== "idle"
    ? `Voice ${voiceState}`
    : !isDesktop && voice.mode === "demo"
      ? "Voice demo"
      : engine.phase === "ready"
        ? micReady ? "Voice available" : "No microphone found"
        : engine.phase === "error"
          ? "Voice unavailable"
          : "Voice loading";
  const voiceDetail = !isDesktop && voice.mode === "demo"
    ? "Scripted browser voice demo; it is not connected to the robot backend."
    : `${microphones.inputsFound} microphone input${microphones.inputsFound === 1 ? "" : "s"} reported · ${voice.engine.name} · ${engine.phase === "ready" ? "engine ready" : engine.message || engine.phase}.`;

  const txConfirmed = serialLive && framesSent > 0 && lastTx !== "NONE";
  const txDetail = !isDesktop
    ? "No hardware TX is possible in browser preview."
    : serialMode === "mock"
      ? lastTx !== "NONE" ? `Simulated frame only: ${lastTx}` : "Mock transport; no physical TX is possible."
      : txConfirmed
        ? `${lastTx} · ${framesSent} frame${framesSent === 1 ? "" : "s"} written by Python this session.`
        : "No serial write has been confirmed in this session.";
  const txLabel = !isDesktop
    ? "No hardware TX"
    : serialMode === "mock"
      ? "Simulated TX"
      : txConfirmed
        ? "Serial write confirmed"
        : "No TX confirmed";
  const gateLabel = !pipelineReported
    ? "Frame gate: awaiting pipeline"
    : !pipelineFrameAvailable && runtime?.cv === "degraded"
      ? "CV gate: unavailable"
      : pipelineFrameAvailable && pipeline.isTxPermitted
        ? "Frame gate: permitted"
        : "Frame gate: no command";

  return (
    <section className="overflow-hidden rounded-[24px] border border-cyan-400/10 bg-[linear-gradient(135deg,rgba(16,22,36,0.98),rgba(7,10,17,0.96)_58%,rgba(20,14,34,0.94))] shadow-[0_22px_65px_rgba(0,0,0,0.3)]">
      <div className="relative overflow-hidden border-b border-white/[0.06] px-5 py-5 sm:px-6">
        <div className="absolute -right-12 -top-28 h-64 w-64 rounded-full bg-cyan-500/[0.08] blur-3xl" />
        <div className="relative flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mono flex items-center gap-2 text-[9px] uppercase tracking-[0.28em] text-cyan-300/80">
              <Icon.Grid className="h-3.5 w-3.5" /> System overview
            </div>
            <h2 className="mt-2 text-xl font-semibold tracking-tight text-white sm:text-2xl">Control room</h2>
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-400">
              Python owns the camera, CV pipeline, safety state, command validation and serial link. This view reports what the backend actually confirms.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={cn("rounded-full border px-3 py-1.5", healthTone === "ok" ? "border-emerald-400/25 bg-emerald-400/[0.08]" : healthTone === "off" ? "border-rose-400/25 bg-rose-400/[0.08]" : "border-amber-300/20 bg-amber-300/[0.06]") }>
              <StatusDot status={healthTone} label={healthLabel} />
            </span>
            {isDesktop && (runtime?.backend === "error" || runtime?.backend === "stopped") && (
              <button className={buttonClass} disabled={busyRestart} onClick={() => void restartBackend()}>
                {busyRestart ? "Restarting…" : "Retry backend"}
              </button>
            )}
          </div>
        </div>
        <p className={cn("relative mt-4 max-w-4xl rounded-xl border px-3.5 py-2.5 text-[11px] leading-relaxed", healthTone === "ok" ? "border-emerald-400/15 bg-emerald-400/[0.04] text-emerald-100/80" : healthTone === "off" ? "border-rose-400/15 bg-rose-400/[0.04] text-rose-100/80" : "border-amber-300/15 bg-amber-300/[0.035] text-amber-100/75")}>
          {healthDetail}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 p-4 sm:grid-cols-2 xl:grid-cols-3 sm:p-5">
        <Tile
          title="Bridge · Python"
          icon={<Icon.Serial className="h-4 w-4" />}
          status={!isDesktop ? "Not in browser" : runtime?.backend === "ready" ? runtime?.cv === "degraded" ? "Runtime ready · CV degraded" : "Pipeline ready" : runtime?.backend === "error" || runtime?.backend === "stopped" ? "Backend error" : runtime?.backend === "stopping" ? "Stopping" : "Starting Python"}
          detail={!isDesktop ? "Open the Electron desktop app for a supervised backend and live bridge status." : runtime?.message || "Waiting for Electron lifecycle status."}
          meta={<span className="mono text-[9px] uppercase tracking-wider text-slate-500">TCP bridge · {runtime?.bridge ?? (isDesktop ? "starting" : "offline")}</span>}
        />

        <Tile
          title="Camera · CV"
          icon={<Icon.Camera className="h-4 w-4" />}
          status={cameraLabel}
          detail={cameraDetail}
          meta={backendControlled && pipelineReported
            ? runtime?.cv === "ready"
              ? <CameraSelector />
              : <span className="mono text-[9px] text-slate-500">CV controls unavailable</span>
            : isDesktop && pipelineReported
              ? <span className="mono text-[9px] text-slate-500">Index {cameraIndex} · {cameraResolution}</span>
              : null}
          action={backendControlled && pipelineReported && runtime?.cv === "ready" && (
            <button className={buttonClass} disabled={busyCamera} onClick={() => void setCamera()}>
              {busyCamera ? "Working…" : cameraLive ? "Pause" : cameraStatus === "stopped" ? "Start" : "Retry"}
            </button>
          )}
        />

        <Tile
          title="Serial · device link"
          icon={<Icon.Chip className="h-4 w-4" />}
          status={serialLabel}
          detail={serialDetail}
          meta={<span className="mono text-[9px] text-slate-500">{isDesktop && serialPort !== "--" ? `${serialPort} · ${serialBaud} baud` : "No port assumed"}</span>}
          action={isDesktop && !serialLive && <span className="mono text-[9px] uppercase tracking-wider text-slate-500">Configure in Hardware</span>}
        />

        <Tile
          title="Voice · Microphone"
          icon={<Icon.Mic className="h-4 w-4" />}
          status={voiceLabel}
          detail={voiceDetail}
          meta={<span className="mono text-[9px] text-slate-500">{voice.modeLabel}</span>}
        />

        <Tile
          title="STOP safety"
          icon={<Icon.Hand className="h-4 w-4" />}
          status={!pipelineReported ? "Awaiting pipeline state" : isArmed ? "STOP armed" : "STOP disarmed"}
          detail={!pipelineReported
            ? "Arm state is not inferred; the button becomes available after Python reports its live state."
            : pipelineFrameAvailable && pipeline.isStopActive
              ? "The pipeline reports STOP active on the current frame."
              : pipelineFrameAvailable
                ? "Software STOP arming state is read directly from the running pipeline."
                : "The last backend snapshot reports the arming flag; no CV frame is currently available."}
          meta={pipelineReported ? <span className="mono text-[9px] text-slate-500">{safetyStatus}</span> : null}
          action={backendControlled && (
            <button className={buttonClass} disabled={!pipelineReported || busySafety} onClick={() => void toggleSafety()}>
              {busySafety ? "Updating…" : isArmed ? "Disarm" : "Arm STOP"}
            </button>
          )}
        />

        <Tile
          title="Actual TX · Protocol"
          icon={<Icon.Serial className="h-4 w-4" />}
          status={txLabel}
          detail={txDetail}
          meta={<span className="mono text-[9px] text-slate-500">{gateLabel} · {framesSent} frame{framesSent === 1 ? "" : "s"}</span>}
          action={pipelineFrameAvailable && pipeline.isTxPermitted && <span className="mono rounded border border-cyan-400/20 bg-cyan-400/[0.06] px-2 py-1 text-[9px] uppercase tracking-wider text-cyan-200">Backend gate</span>}
        />
      </div>

      {(notice || cameraNotice) && <div className="border-t border-amber-300/10 px-5 py-2 text-[11px] text-amber-200">{notice || cameraNotice}</div>}
    </section>
  );
}
