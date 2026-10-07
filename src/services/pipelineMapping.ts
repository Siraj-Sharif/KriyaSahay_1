/**
 * Single home for every raw-pipeline → UI-state mapping.
 *
 * Before Phase 1 this logic existed in three places (`usePipelineState`, `desktopVision`
 * and `desktopSerial`), which meant the same wire value could be interpreted differently
 * depending on which component you looked at. All raw→UI translation now lives here so
 * `usePipelineState` is the one authority for live pipeline state.
 *
 * This module is pure and has no IPC / React dependencies — it is safe to import anywhere.
 */
import { GESTURE_DEFS } from "./gestures";
import { GESTURES, type Gesture } from "./types";

/** Gesture names the pipeline may report that are *not* transmittable gestures. */
export const NON_GESTURE_COMMANDS = [
  "NO_COMMAND",
  "NO_HAND",
  "AMBIGUOUS",
  "ERROR",
  "INVALID",
  "UNKNOWN",
  "REST",
  "NONE",
] as const;

/** Safe state used whenever the pipeline is not reporting a real gesture. */
export const SAFE_GESTURE: Gesture = "STOP";

/**
 * Marker written into synthetic (manual-override) frames.
 *
 * Frames carrying this tag are explicit operator previews, never detections, so the
 * command-emission path must ignore them — that distinction is what makes the
 * `desktopVision.overrideGesture` double-send structurally impossible.
 */
export const OVERRIDE_MODEL_TAG = "MANUAL_OVERRIDE";

/** One detected hand as the bridge sends it: `[x, y, z]` normalised image coords. */
export type RawLandmarkPoint = [number, number, number];

/** Truthful camera state reported by the pipeline (Phase 2). */
export interface RawCameraInfo {
  index: number;
  enabled: boolean;
  opened: boolean;
  state: string;
  backend: string;
  resolution: string;
  target_resolution: string;
  error: string | null;
}

/** Truthful serial state reported by the pipeline (Phase 2). */
export interface RawSerialInfo {
  mode: "real" | "mock" | "disabled";
  port: string;
  baud: number;
  connected: boolean;
  state: string;
  frames_sent: number;
  last_tx: string;
}

/** Voice state recorded inside the pipeline (Phase 2). */
export interface RawVoiceInfo {
  state: string;
  transcript: string;
  command: string | null;
  reply: string;
  engine: string;
}

/** Raw `VisualizationState` payload pushed by the Python bridge over Electron IPC. */
export interface RawPipelineState {
  command: string;
  confidence: number;
  fps: number;
  latency_ms: number;
  hand_count: string;
  pipeline_state: string;
  model_used: string;
  raw_prediction: string;
  routing_str: string;
  transport_status: string;
  last_tx: string;
  is_stop_active: boolean;
  is_ambiguous: boolean;
  message: string;
  safety_status: string;
  handedness: string;
  landmarks_count: number;
  /** Not in the current bridge payload, but tolerated for forward compatibility. */
  is_stabilized?: boolean;
  /**
   * Backend-owned STOP arming flag. Sent by every bridge snapshot since Phase 2 (the field
   * exists on `VisualizationState`); optional here only so an older bridge cannot break the UI.
   */
  is_armed?: boolean;
  is_tx_permitted?: boolean;
  final_command?: string;
  stabilizer_state?: string;
  /* ── Phase 2 runtime sections (additive; absent on older bridges) ── */
  camera_info?: RawCameraInfo;
  serial_info?: RawSerialInfo;
  voice_info?: RawVoiceInfo;
  errors?: string[];
  manual_tx_count?: number;
  /** `[[[x, y, z], …21 points], …one entry per hand]`. */
  landmarks?: RawLandmarkPoint[][];
}

const GESTURE_SET: ReadonlySet<string> = new Set<string>(GESTURES as readonly string[]);

/**
 * Map a raw `command` string to the UI `Gesture` union.
 *
 * Unknown / non-gesture states collapse to the safe STOP pose so the UI never renders a
 * gesture the hardware cannot receive. (CALL and OK are real taxonomy members and pass
 * straight through.)
 */
export function mapCommandToGesture(cmd: string | null | undefined): Gesture {
  if (!cmd) return SAFE_GESTURE;
  const upper = String(cmd).trim().toUpperCase();
  if (GESTURE_SET.has(upper)) return upper as Gesture;
  return SAFE_GESTURE;
}

/** `true` when the raw command is a real, transmittable gesture. */
export function isTransmittableGesture(cmd: string | null | undefined): boolean {
  if (!cmd) return false;
  return GESTURE_SET.has(String(cmd).trim().toUpperCase());
}

/** Normalise the bridge's handedness string; anything unknown becomes `null`. */
export function mapHandedness(h: string | null | undefined): "Left" | "Right" | null {
  if (!h) return null;
  const lower = String(h).trim().toLowerCase();
  if (lower === "left") return "Left";
  if (lower === "right") return "Right";
  return null;
}

/** `hand_count` arrives as a string ("0" | "1" | "2" | "2+"). */
export function parseHandCount(handCount: string | number | null | undefined): number {
  if (typeof handCount === "number") return handCount;
  const parsed = Number.parseInt(String(handCount ?? "0"), 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Pull the COM/tty port out of a transport status such as `"CONNECTED (COM3)"`.
 * Returns `"--"` when the status carries no port.
 */
export function extractSerialPort(transportStatus: string | null | undefined): string {
  if (!transportStatus) return "--";
  const match = String(transportStatus).match(/\b(COM\d+|tty(?:USB|ACM)\d+)\b/i);
  return match ? match[1] : "--";
}

/** `true` for `"CONNECTED"` / `"CONNECTED (COM3)"`, `false` for `"DISABLED"` / `"MOCK"`. */
export function isTransportConnected(transportStatus: string | null | undefined): boolean {
  if (!transportStatus) return false;
  return String(transportStatus).toUpperCase().includes("CONNECTED");
}

/**
 * Baud rate to display for the current transport status.
 *
 * Replaces a dead ternary that read `connected ? 115200 : 115200`; the bridge does not
 * currently report a baud rate, so this stays 115200 until it does — but the value is now
 * derived in one place instead of being copy-pasted.
 */
export const DEFAULT_BAUD_RATE = 115200;

export function resolveBaudRate(_transportStatus?: string | null): number {
  // Reserved for when the bridge starts reporting the negotiated baud rate.
  return DEFAULT_BAUD_RATE;
}

/**
 * Landmarks from the bridge payload.
 *
 * Phase 2: the bridge ships the real MediaPipe points, so the camera overlay draws the
 * hand the pipeline actually detected instead of staying empty.
 * Returns the first hand as `{x, y, z}` (the overlay renders a single hand).
 */
export function parseLandmarks(raw?: RawLandmarkPoint[][] | null): { x: number; y: number; z: number }[] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  const hand = raw[0];
  if (!Array.isArray(hand)) return [];
  return hand
    .filter((p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]))
    .map((p) => ({ x: Number(p[0]), y: Number(p[1]), z: Number(p[2] ?? 0) }));
}

/** `true` when the payload carries a usable landmark set for the overlay. */
export function hasLandmarks(raw?: RawLandmarkPoint[][] | null): boolean {
  return parseLandmarks(raw).length >= 21;
}

/** All gestures as a typed array (mirrors `Object.keys(GESTURE_DEFS)`). */
export const ALL_GESTURES = Object.keys(GESTURE_DEFS) as Gesture[];
