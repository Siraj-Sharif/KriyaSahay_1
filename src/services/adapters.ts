/**
 * Integration adapters.
 *
 * These interfaces define the contracts that the live computer-vision pipeline,
 * the serial bridge, and the voice assistant module will implement. The UI only
 * talks to these interfaces — swap `createMock*` for real implementations
 * (WebSocket / WebSerial / WebRTC) without touching any component code.
 */
import { GESTURES, type CameraInfo, type EngineStatus, type FingerPose, type Gesture, type SerialState, type VisionFrame, type VoiceMessage, type VoiceState } from "./types";
import { GESTURE_DEFS, encodeCommand, landmarksForPose } from "./gestures";
import { cancelSpeech, speakText } from "./tts";

export type Unsubscribe = () => void;

export interface VisionAdapter {
  subscribe(cb: (frame: VisionFrame) => void): Unsubscribe;
  getCamera(): CameraInfo;
  /** Manually force a gesture (used by the Gesture Guide / voice commands). */
  overrideGesture(g: Gesture | null): void;
  /** Power the camera / CV pipeline on or off. When off, frames report no hand and 0 fps. */
  setEnabled(enabled: boolean): void;
  /** Tell the CV pipeline which camera to open (browser deviceId / OS camera index). */
  setSource(deviceId: string): void;
}

/** Where a command came from — telemetry only; every source uses the same pipeline path. */
export type CommandSourceTag = "detector" | "override" | "voice";

export interface SerialAdapter {
  subscribe(cb: (state: SerialState) => void): Unsubscribe;
  /**
   * Ask the pipeline to validate, encode and transmit one command.
   *
   * `source` is an optional telemetry label (additive Phase 2 change). It never changes
   * *where* the command goes: every source is transmitted by the running pipeline.
   */
  send(command: Gesture, source?: CommandSourceTag): Promise<void>;
  connect(): Promise<void>;
  disconnect(): Promise<void>;
}

/**
 * Result of interpreting one transcript.
 *
 * Produced by the running Python pipeline in the desktop app (`voice.transcript`), or by
 * the local `intents.ts` parser in the browser demo.
 */
export interface VoiceInterpretation {
  kind: "gesture" | "release" | "status" | "help" | "unknown";
  gesture: Gesture | null;
  reply: string;
  /** Wire command name the pipeline accepted, if any (e.g. "CALL"). */
  command?: string | null;
  /** Real backend error, e.g. the serial link is unavailable. */
  error?: string | null;
}

/**
 * Optional transcript interpreter.
 *
 * When supplied, the voice adapter does not decide commands itself: it hands the transcript
 * to the running pipeline, which validates and transmits through the same serial owner as
 * camera and manual commands.
 */
export type VoiceInterpreter = (transcript: string) => Promise<VoiceInterpretation | null> | VoiceInterpretation | null;

export interface VoiceAdapter {
  subscribe(cb: (state: VoiceState, messages: VoiceMessage[], level: number) => void): Unsubscribe;
  startListening(): void;
  stopListening(): void;
  /** Run a typed phrase through the same pipeline as speech. */
  sendText(text: string): void;
  /** "live" = real speech recognition, "demo" = scripted mock. */
  mode: "live" | "demo";
  modeLabel: string;
  /** Which microphone to capture from ("" = system default). */
  setInputDevice(deviceId: string): void;
  /** Model download / load progress for the recognition engine. */
  subscribeEngine(cb: (s: EngineStatus) => void): Unsubscribe;
  /** Describes the active recognition engine so the UI can be honest about its limits. */
  engine: { name: string; offline: boolean; note: string };
}

/* ------------------------------------------------------------------ */
/* Mock Vision                                                         */
/* ------------------------------------------------------------------ */

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function createMockVision(): VisionAdapter {
  const listeners = new Set<(f: VisionFrame) => void>();
  let override: Gesture | null = null;
  let enabled = true;
  let current: Gesture = "STOP";
  let confidence = 0.9;
  let handDetected = true;
  let pose: number[] = [...GESTURE_DEFS[current].pose];
  let frameTimer: number | undefined;
  let gestureTimer: number | undefined;

  const camera: CameraInfo = {
    name: "FaceTime HD Camera",
    kind: "built-in",
    resolution: "1280 × 720",
    status: "connected",
    deviceId: "cam-0",
  };

  const pickGesture = () => {
    if (override) return;
    if (Math.random() < 0.08) {
      handDetected = false;
      return;
    }
    handDetected = true;
    current = GESTURES[Math.floor(Math.random() * GESTURES.length)];
  };

  const start = () => {
    gestureTimer = window.setInterval(pickGesture, 3200);
    frameTimer = window.setInterval(() => {
      if (!enabled) {
        const off: VisionFrame = {
          timestamp: Date.now(),
          gesture: current,
          confidence: 0,
          fps: 0,
          latencyMs: 0,
          handDetected: false,
          handCount: 0,
          handedness: null,
          model: "Pipeline suspended",
          landmarks: [],
        };
        listeners.forEach((l) => l(off));
        return;
      }
      const target = override ?? current;
      const targetPose = GESTURE_DEFS[target].pose;
      pose = pose.map((p, i) => lerp(p, targetPose[i], 0.18));
      const detected = override ? true : handDetected;
      const targetConf = detected ? 0.82 + Math.random() * 0.17 : 0;
      confidence = lerp(confidence, targetConf, 0.25);
      const frame: VisionFrame = {
        timestamp: Date.now(),
        gesture: target,
        confidence: detected ? Math.min(0.995, confidence) : 0,
        fps: Math.round(29 + Math.random() * 2),
        latencyMs: Math.round(95 + Math.random() * 45),
        handDetected: detected,
        handCount: detected ? 1 : 0,
        handedness: detected ? "Right" : null,
        model: "MediaPipe Hands → NeuroGrip MLP v2",
        landmarks: detected ? landmarksForPose(pose as FingerPose, 0.006) : [],
      };
      listeners.forEach((l) => l(frame));
    }, 1000 / 30);
  };
  const stop = () => {
    window.clearInterval(frameTimer);
    window.clearInterval(gestureTimer);
  };

  return {
    subscribe(cb) {
      listeners.add(cb);
      if (listeners.size === 1) start();
      return () => {
        listeners.delete(cb);
        if (listeners.size === 0) stop();
      };
    },
    getCamera: () => camera,
    setSource(deviceId) {
      camera.deviceId = deviceId;
    },
    setEnabled(v) {
      enabled = v;
      camera.status = v ? "connected" : "disconnected";
    },
    overrideGesture(g) {
      override = g;
      if (g) current = g;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Mock Serial                                                         */
/* ------------------------------------------------------------------ */

export function createMockSerial(): SerialAdapter {
  const listeners = new Set<(s: SerialState) => void>();
  let state: SerialState = {
    connected: true,
    port: "COM3",
    baudRate: 115200,
    tx: "idle",
    lastCommand: encodeCommand("STOP"),
    packetsSent: 0,
    lastAckMs: 4,
  };
  const emit = () => listeners.forEach((l) => l({ ...state }));

  return {
    subscribe(cb) {
      listeners.add(cb);
      cb({ ...state });
      return () => listeners.delete(cb);
    },
    async send(command) {
      if (!state.connected) return;
      state = { ...state, tx: "transmitting", lastCommand: encodeCommand(command) };
      emit();
      await new Promise((r) => setTimeout(r, 60 + Math.random() * 60));
      state = {
        ...state,
        tx: "idle",
        packetsSent: state.packetsSent + 1,
        lastAckMs: Math.round(2 + Math.random() * 6),
      };
      emit();
    },
    async connect() {
      await new Promise((r) => setTimeout(r, 600));
      state = { ...state, connected: true };
      emit();
    },
    async disconnect() {
      state = { ...state, connected: false, tx: "idle" };
      emit();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Mock Voice                                                          */
/* ------------------------------------------------------------------ */

const VOICE_SCRIPTS: { heard: string; reply: string; command?: Gesture }[] = [
  { heard: "NeuroGrip, close the hand.", reply: "Closing the hand. Sending CLOSED_FIST to actuator bus.", command: "CLOSED_FIST" },
  { heard: "Grip the object.", reply: "Engaging adaptive grasp. GRIP command transmitted.", command: "GRIP" },
  { heard: "Open the hand and stop.", reply: "Releasing all actuators. STOP pose active.", command: "STOP" },
  { heard: "Give me a thumbs up.", reply: "Acknowledged. THUMBS_UP pose dispatched.", command: "THUMBS_UP" },
  { heard: "What's the system status?", reply: "All systems nominal. Serial link on COM3 at 115200 baud, vision pipeline running at 30 frames per second." },
  { heard: "Show me the OK sign.", reply: "Executing precision OK gesture.", command: "OK" },
];

export function createMockVoice(onCommand?: (g: Gesture | null) => void): VoiceAdapter {
  const listeners = new Set<(s: VoiceState, m: VoiceMessage[], level: number) => void>();
  let state: VoiceState = "idle";
  let messages: VoiceMessage[] = [
    {
      id: "m0",
      role: "assistant",
      text: "NeuroGrip voice interface online. Tap the microphone and speak a command.",
      timestamp: Date.now(),
    },
  ];
  let level = 0;
  let levelTimer: number | undefined;
  let scriptIdx = 0;
  let sequence = 0;

  const emit = () => listeners.forEach((l) => l(state, messages, level));
  const setState = (s: VoiceState) => {
    state = s;
    emit();
  };

  const startLevels = () => {
    window.clearInterval(levelTimer);
    levelTimer = window.setInterval(() => {
      level = state === "idle" ? 0 : 0.3 + Math.random() * 0.7;
      emit();
    }, 80);
  };

  const run = async () => {
    const mySeq = ++sequence;
    const script = VOICE_SCRIPTS[scriptIdx++ % VOICE_SCRIPTS.length];
    setState("listening");
    startLevels();
    await new Promise((r) => setTimeout(r, 2400));
    if (mySeq !== sequence) return;
    messages = [...messages, { id: `u${Date.now()}`, role: "user", text: script.heard, timestamp: Date.now() }];
    setState("processing");
    await new Promise((r) => setTimeout(r, 1400));
    if (mySeq !== sequence) return;
    messages = [
      ...messages,
      { id: `a${Date.now()}`, role: "assistant", text: script.reply, command: script.command, timestamp: Date.now() },
    ];
    if (script.command) onCommand?.(script.command);
    setState("speaking");
    await Promise.all([speakText(script.reply, () => mySeq === sequence), new Promise((r) => setTimeout(r, 1800))]);
    if (mySeq !== sequence) return;
    window.clearInterval(levelTimer);
    level = 0;
    setState("idle");
  };

  return {
    mode: "demo",
    modeLabel: "Demo mode · speech recognition unavailable in this browser",
    setInputDevice() {},
    subscribeEngine(cb) {
      cb({ phase: "ready", progress: 100, message: "Demo mode" });
      return () => undefined;
    },
    engine: { name: "Scripted demo", offline: true, note: "No real recognition in this browser — use Chrome or Edge, or type a command." },
    sendText() {
      if (state === "idle") void run();
    },
    subscribe(cb) {
      listeners.add(cb);
      cb(state, messages, level);
      return () => {
        listeners.delete(cb);
        if (listeners.size === 0) window.clearInterval(levelTimer);
      };
    },
    startListening() {
      if (state === "idle") void run();
    },
    stopListening() {
      sequence++;
      cancelSpeech();
      window.clearInterval(levelTimer);
      level = 0;
      setState("idle");
    },
  };
}
