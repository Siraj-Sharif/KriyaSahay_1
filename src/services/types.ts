export const GESTURES = [
  "CALL",
  "CLOSED_FIST",
  "FOUR_FINGERS",
  "GRIP",
  "THUMBS_UP",
  "PINKY",
  "MIDDLE_FINGER",
  "OK",
  "INDEX_FINGER",
  "INDEX_PINKY",
  "STOP",
  "TWO_FINGERS",
  "THREE_FINGERS",
] as const;

export type Gesture = (typeof GESTURES)[number];

/**
 * Finger curl values 0 (fully extended) → 1 (fully curled).
 * Order: thumb, index, middle, ring, pinky, thumbOpposition.
 * thumbOpposition (0..1) swings the thumb across so its tip touches the index tip.
 */
export type FingerPose = [number, number, number, number, number, number];

export interface Landmark {
  x: number; // normalized 0..1
  y: number;
  z: number;
}

export interface VisionFrame {
  timestamp: number;
  gesture: Gesture;
  confidence: number; // 0..1
  fps: number;
  latencyMs: number;
  handDetected: boolean;
  handCount: number;
  handedness: "Left" | "Right" | null;
  model: string;
  landmarks: Landmark[]; // 21 MediaPipe-style landmarks
}

export interface CameraInfo {
  name: string;
  kind: "built-in" | "external";
  resolution: string;
  status: "connected" | "disconnected" | "initializing";
  deviceId: string;
}

export interface SerialState {
  connected: boolean;
  port: string;
  baudRate: number;
  tx: "idle" | "transmitting" | "error";
  lastCommand: string;
  packetsSent: number;
  lastAckMs: number;
}

export interface ConfidencePoint {
  t: number;
  confidence: number;
  fps: number;
}

export type VoiceState = "idle" | "listening" | "processing" | "speaking";

export interface VoiceMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  command?: Gesture;
  timestamp: number;
}

/** State of the speech-recognition engine (model download / loading / ready). */
export interface EngineStatus {
  phase: "idle" | "loading" | "downloading" | "ready" | "error";
  progress: number; // 0..100
  message: string;
}
