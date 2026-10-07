import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createMockSerial, createMockVision, createMockVoice, type SerialAdapter, type VisionAdapter, type VoiceAdapter } from "./adapters";
import { createDesktopVision } from "./desktopVision";
import { createDesktopSerial } from "./desktopSerial";
import type { CameraInfo, ConfidencePoint, Gesture, SerialState, VisionFrame } from "./types";
import { GESTURE_DEFS } from "./gestures";
import { createLocalWhisperVoice, isDesktopSttAvailable } from "./localWhisperVoice";
import { createWebSpeechVoice, isSpeechRecognitionSupported } from "./webSpeechVoice";
import { useCameraDevices, type CameraDevice } from "../hooks/useCameraDevices";
import { useGestureCommand } from "../hooks/useGestureCommand";
import { usePipelineState } from "../hooks/usePipelineState";
import {
  isPipelineControlAvailable,
  selectCamera as selectBackendCamera,
  setCameraEnabled,
  setVoiceState,
  sendVoiceTranscript,
  voiceTranscriptFrom,
} from "./pipelineControl";

interface SystemContextValue {
  frame: VisionFrame | null;
  /** High-rate frame ref for canvas / 3D consumers (no re-render). */
  frameRef: React.RefObject<VisionFrame | null>;
  camera: CameraInfo;
  serial: SerialState | null;
  history: ConfidencePoint[];
  vision: VisionAdapter;
  serialAdapter: SerialAdapter;
  voice: VoiceAdapter;
  overrideGesture: Gesture | null;
  setOverrideGesture: (g: Gesture | null) => void;
  gestureCounts: Record<Gesture, number>;
  cameraOn: boolean;
  setCameraOn: (on: boolean) => void;
  /** All detected video inputs + the one currently chosen. */
  cameraDevices: CameraDevice[];
  selectedCamera: CameraDevice;
  selectCamera: (id: string) => void;
  refreshCameras: () => Promise<void>;
  /** Called by the preview once a stream opens, to report its real resolution. */
  reportCameraResolution: (res: string | null) => void;
  /** Last real camera error reported by the pipeline (null when healthy). */
  cameraNotice: string | null;
  /** `true` when camera + serial actions are handled by the running Python pipeline. */
  backendControlled: boolean;
}

const SystemContext = createContext<SystemContextValue | null>(null);

export function SystemProvider({ children }: { children: ReactNode }) {
  // Detect if running in Electron desktop app
  const isDesktop = typeof window !== "undefined" && !!(window as any).neurogrip?.isDesktop;

  const vision = useMemo(() => {
    if (isDesktop) {
      return createDesktopVision();
    }
    return createMockVision();
  }, [isDesktop]);

  const serialAdapter = useMemo(() => {
    if (isDesktop) {
      return createDesktopSerial();
    }
    return createMockSerial();
  }, [isDesktop]);
  const [frame, setFrame] = useState<VisionFrame | null>(null);
  const frameRef = useRef<VisionFrame | null>(null);
  const [serial, setSerial] = useState<SerialState | null>(null);
  const [history, setHistory] = useState<ConfidencePoint[]>([]);
  const [gestureCounts, setCounts] = useState<Record<Gesture, number>>(
    () => Object.fromEntries(Object.keys(GESTURE_DEFS).map((k) => [k, 0])) as Record<Gesture, number>,
  );
  const pipelineState = usePipelineState();
  // Camera, serial, commands and voice are owned by the running pipeline in desktop mode.
  const backendControlled = isPipelineControlAvailable();
  const [localCameraOn, setLocalCameraOn] = useState(true);
  const [cameraNotice, setCameraNotice] = useState<string | null>(null);
  const { devices: cameraDevices, selected: selectedCamera, select, refresh: refreshCameras } = useCameraDevices();
  const [resolution, setResolution] = useState<string | null>(null);

  /**
   * The one owner of command emission (detector frames, manual overrides and voice
   * commands all funnel through it). Its re-entrancy + de-duplication guard is what stops
   * `desktopVision.overrideGesture` from producing a second serial write.
   */
  const gestureCommand = useGestureCommand({ vision, serial: serialAdapter });
  // Destructured so effect/memo dependencies stay referentially stable across emissions.
  const {
    emitFromFrame,
    setOverride: setOverrideGesture,
    previewOverride,
    reset: resetCommandHistory,
  } = gestureCommand;
  const overrideGesture = gestureCommand.override;

  const selectCamera = (id: string) => {
    if (id === selectedCamera.id) return;
    if (backendControlled) {
      const index = Number(id);
      if (!Number.isFinite(index)) {
        setCameraNotice(`Cannot select camera "${id}": the pipeline addresses cameras by index.`);
        return;
      }
      // Real switch: the pipeline closes the old device and opens this one.
      void selectBackendCamera(index).then((res) => {
        setCameraNotice(res.ok ? null : res.error ?? "Camera switch rejected by the pipeline.");
      });
    }
    select(id);
    setResolution(null);
    vision.setSource(id);
  };

  const setCameraOn = (on: boolean) => {
    vision.setEnabled(on);
    if (!on) {
      // Powering the camera off clears the emission history, so switching it back on and
      // re-showing the same gesture transmits again (matches pre-Phase-1 behaviour).
      resetCommandHistory();
    }
    if (backendControlled) {
      // Start/stop actual capture in the Python pipeline. The local flag only mirrors the
      // backend; the authoritative value comes back as `camera_info.enabled`.
      void setCameraEnabled(on).then((res) => {
        setCameraNotice(res.ok ? null : res.error ?? "Camera command rejected by the pipeline.");
      });
    }
    setLocalCameraOn(on);
  };

  // Authority comes from the pipeline: a camera switched off (or failing) elsewhere is
  // reflected here, and camera selection follows backend-originated changes too.
  const cameraOn = backendControlled ? pipelineState.cameraEnabled : localCameraOn;

  useEffect(() => {
    if (!backendControlled) return;
    const active = String(pipelineState.cameraIndex);
    if (active !== selectedCamera.id) select(active);
  }, [backendControlled, pipelineState.cameraIndex, selectedCamera.id, select]);

  // Surface real pipeline camera errors (unavailable / invalid / permission).
  useEffect(() => {
    const notice = pipelineState.cameraError ?? pipelineState.errors[0] ?? null;
    if (notice) setCameraNotice(notice);
  }, [pipelineState.cameraError, pipelineState.errors]);

  // latest values for spoken status reports (read lazily by the voice adapter)
  const statusRef = useRef<() => string>(() => "Status unavailable.");
  statusRef.current = () => {
    const f = frameRef.current;
    const cam = cameraOn ? `Camera on, ${selectedCamera.label}.` : "Camera is off.";
    const vis = f && f.handDetected ? `Detecting ${f.gesture.replace(/_/g, " ").toLowerCase()} at ${Math.round(f.confidence * 100)} percent confidence, ${f.fps} frames per second.` : "No hand detected.";
    const ser = serial?.connected ? `Serial connected on ${serial.port} at ${serial.baudRate} baud.` : "Serial is disconnected.";
    return `${cam} ${vis} ${ser}`;
  };

  const voice = useMemo(() => {
    // Desktop: the pipeline has already transmitted the interpreted command itself, so the
    // renderer must ONLY preview it — emitting here would duplicate the frame on the wire.
    // Browser/mock mode keeps the original behaviour of previewing *and* transmitting once.
    const onCommand = backendControlled ? previewOverride : (g: Gesture | null) => setOverrideGesture(g);
    // Phase 2: the pipeline interprets the transcript and transmits any resulting command
    // through the same serial owner as camera detections. The adapter only shows the result.
    const interpret = backendControlled
      ? async (transcript: string) => {
          const res = await sendVoiceTranscript(transcript, "whisper-base.en");
          const parsed = voiceTranscriptFrom(res);
          if (!parsed) {
            const error = res.error ?? "Voice command failed.";
            return { kind: "unknown" as const, gesture: null, reply: error, command: null, error };
          }
          return {
            kind: (parsed.intent === "gesture" || parsed.intent === "release" || parsed.intent === "status" || parsed.intent === "help"
              ? parsed.intent
              : "unknown") as "gesture" | "release" | "status" | "help" | "unknown",
            gesture: (parsed.gesture as Gesture | null) ?? null,
            reply: parsed.reply,
            command: parsed.command ?? null,
            error: res.ok ? null : res.error ?? null,
          };
        }
      : undefined;
    const opts = { onCommand, getStatus: () => statusRef.current(), interpret };
    // desktop app → on-device Whisper · Chrome/Edge → browser speech API · otherwise scripted demo
    if (isDesktopSttAvailable()) return createLocalWhisperVoice(opts);
    if (isSpeechRecognitionSupported()) return createWebSpeechVoice(opts);
    return createMockVoice(onCommand);
  }, [setOverrideGesture, previewOverride, backendControlled]);

  // Report the voice engine's real state to the pipeline, so `voice_info` — and everything
  // the UI shows from it — is backend-owned rather than a second copy in React.
  useEffect(() => {
    if (!backendControlled) return;
    return voice.subscribe((state) => {
      void setVoiceState(state, voice.engine.name);
    });
  }, [voice, backendControlled]);

  useEffect(() => {
    let lastUi = 0;
    let lastHist = 0;
    const unsub = vision.subscribe((f) => {
      frameRef.current = f;
      const now = f.timestamp;
      if (now - lastUi > 120) {
        lastUi = now;
        setFrame(f);
      }
      if (now - lastHist > 250) {
        lastHist = now;
        setHistory((h) => [...h.slice(-119), { t: now, confidence: Math.round(f.confidence * 100), fps: f.fps }]);
      }
      // Detected frames may only produce a hardware command when the renderer *is* the
      // owner (browser/mock mode). In desktop mode the running pipeline stabilizes,
      // validates and transmits the detection itself — re-emitting the raw frame here would
      // (a) put a second identical frame on the wire and (b) dispatch gestures the backend
      // has not stabilized yet, bypassing the stabilizer/validator window.
      if (!backendControlled) emitFromFrame(f);
    });
    const unsubSerial = serialAdapter.subscribe(setSerial);
    return () => {
      unsub();
      unsubSerial();
    };
  }, [vision, serialAdapter, emitFromFrame, backendControlled]);

  /**
   * Gesture tally: counted once per change of the *effective* gesture (camera detection or
   * manual override). Previously this was incremented from two independent paths, so a
   * single manual override counted twice.
   */
  const countedRef = useRef<Gesture | null>(null);
  useEffect(() => {
    const active: Gesture | null = overrideGesture ?? (frame?.handDetected ? frame.gesture : null);
    if (!active || active === countedRef.current) return;
    countedRef.current = active;
    setCounts((c) => ({ ...c, [active]: (c[active] || 0) + 1 }));
  }, [overrideGesture, frame]);

  // Real camera descriptor. In desktop mode every field comes from the pipeline's
  // `camera_info`; the browser demo keeps its local device values.
  const value: SystemContextValue = {
    frame,
    frameRef,
    camera: {
      name: isDesktop ? `NeuroGrip Camera #${pipelineState.cameraIndex} (Python OpenCV)` : selectedCamera.label,
      kind: isDesktop ? "external" : selectedCamera.kind,
      resolution: isDesktop ? pipelineState.cameraResolution : (resolution ?? vision.getCamera().resolution),
      status: cameraOn
        ? (pipelineState.cameraStatus === "connected" ? "connected" : "initializing")
        : "disconnected",
      deviceId: isDesktop ? `python-opencv-${pipelineState.cameraIndex}` : selectedCamera.id,
    },
    serial,
    history,
    vision,
    serialAdapter,
    voice,
    overrideGesture,
    setOverrideGesture,
    gestureCounts,
    cameraOn,
    setCameraOn,
    cameraDevices,
    selectedCamera,
    selectCamera,
    refreshCameras,
    reportCameraResolution: setResolution,
    cameraNotice,
    backendControlled,
  };

  return <SystemContext.Provider value={value}>{children}</SystemContext.Provider>;
}

export function useSystem() {
  const ctx = useContext(SystemContext);
  if (!ctx) throw new Error("useSystem must be used within SystemProvider");
  return ctx;
}
