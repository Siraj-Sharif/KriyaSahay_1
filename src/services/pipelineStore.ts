/**
 * `pipelineStore` — the single source of truth for interpreted live pipeline state.
 *
 * Before Phase 1, `usePipelineState()` was instantiated ~8 times across the UI. Each
 * instance opened its own React subscription and ran its own copy of the parsing/mapping
 * code, so the same raw frame produced eight independent state machines that could (and
 * did) disagree. The store collapses that into one module-level reducer over the single
 * IPC subscription owned by `pipelineSource`, exposed to React through
 * `useSyncExternalStore`.
 *
 * Consumers therefore get the *same object reference* until a genuinely new frame arrives,
 * which keeps re-render behaviour identical to before while removing the duplication.
 */
import type { CameraInfo, Gesture } from "./types";
import {
  DEFAULT_BAUD_RATE,
  extractSerialPort,
  mapCommandToGesture,
  mapHandedness,
  isTransportConnected,
  parseHandCount,
  parseLandmarks,
  resolveBaudRate,
  type RawCameraInfo,
  type RawPipelineState,
  type RawVoiceInfo,
} from "./pipelineMapping";
import { subscribePipelineState } from "./pipelineSource";

export interface PipelineState {
  // Core gesture state
  command: string; // Raw command from Python (may include NO_COMMAND, AMBIGUOUS, etc.)
  gesture: Gesture; // Mapped to our Gesture type
  confidence: number; // 0-1
  isStopActive: boolean; // STOP state active
  isAmbiguous: boolean; // 2+ hands detected
  handDetected: boolean; // Exactly 1 hand
  handCount: number; // 0, 1, or 2+
  handedness: "Left" | "Right" | null;

  // Performance metrics
  fps: number;
  latencyMs: number;

  // Pipeline info
  pipelineState: string; // TRACKING, NO_HAND, AMBIGUOUS, STABLE_CMD, etc.
  modelUsed: string; // HYBRID, HAGRID, EXTRA_TREES, RULE_BASED, N/A
  rawPrediction: string; // Raw model prediction
  routingStr: string; // e.g., "HYBRID -> GRIP"
  safetyStatus: string; // STOP ARMED, STOP DISARMED, etc.
  /** Backend-owned STOP arming flag (`is_armed`) — never a local React guess. */
  isArmed: boolean;
  /** `true` only when the pipeline actually transmitted a frame for this update. */
  isTxPermitted: boolean;
  /** Command the pipeline reports as final for this frame (may be NO_COMMAND). */
  finalCommand: string;

  // Serial/hardware
  transportStatus: string; // CONNECTED, DISCONNECTED, DISABLED, MOCK
  lastTx: string; // Last protocol frame sent (e.g., "NG1|GRIP")
  serialPort: string; // Extracted from transport_status if available
  baudRate: number; // Default 115200 if connected

  // Camera — derived from the pipeline's real `camera_info`, never hardcoded
  cameraStatus: string; // connected, stopped, disconnected, error
  cameraResolution: string;
  cameraIndex: number; // Real cv2/OpenCV device index the pipeline holds
  cameraEnabled: boolean; // UI-requested capture state (backend-owned)
  cameraError: string | null;
  landmarksCount: number; // Number of landmarks (0 or 21)
  /** Real landmark points from the pipeline (first hand), for the CV overlay. */
  landmarks: { x: number; y: number; z: number }[];

  // Serial — derived from the pipeline's real `serial_info`
  serialConnected: boolean; // true only when the pipeline holds a serial link
  serialMode: "real" | "mock" | "disabled";
  framesSent: number;

  // Voice — recorded and echoed by the pipeline
  voice: RawVoiceInfo;

  /** Real error strings reported by the pipeline (camera / serial / command failures). */
  errors: string[];

  // Raw message
  message: string; // Diagnostic message from pipeline
}

/** Shown until the first frame from the Python bridge arrives. */
export const DEFAULT_PIPELINE_STATE: PipelineState = {
  command: "NO_COMMAND",
  gesture: "STOP",
  confidence: 0,
  isStopActive: false,
  isAmbiguous: false,
  handDetected: false,
  handCount: 0,
  handedness: null,
  fps: 0,
  latencyMs: 0,
  pipelineState: "STARTING",
  modelUsed: "N/A",
  rawPrediction: "N/A",
  routingStr: "N/A",
  safetyStatus: "ARMED",
  isArmed: false,
  isTxPermitted: false,
  finalCommand: "NO_COMMAND",
  transportStatus: "DISCONNECTED",
  lastTx: "NONE",
  serialPort: "--",
  baudRate: DEFAULT_BAUD_RATE,
  cameraStatus: "disconnected",
  cameraResolution: "1280 × 720",
  cameraIndex: 0,
  cameraEnabled: false,
  cameraError: null,
  landmarksCount: 0,
  landmarks: [],
  serialConnected: false,
  serialMode: "disabled",
  framesSent: 0,
  voice: { state: "idle", transcript: "", command: null, reply: "", engine: "N/A" },
  errors: [],
  message: "Waiting for pipeline...",
};

const CAMERA_STATUS_CONNECTED = "connected";
const CAMERA_STATUS_STOPPED = "stopped";
const DEFAULT_CAMERA_RESOLUTION = "1280 × 720";

/** Truthful camera status from the pipeline's own camera state. */
function deriveCameraStatus(info?: RawCameraInfo): string {
  if (!info) return CAMERA_STATUS_CONNECTED; // legacy payloads carried a live camera
  if (info.error) return "error";
  if (!info.enabled) return CAMERA_STATUS_STOPPED;
  return info.opened ? CAMERA_STATUS_CONNECTED : "disconnected";
}

/** The one and only raw → `PipelineState` mapping in the frontend. */
export function derivePipelineState(raw: RawPipelineState): PipelineState {
  const handCount = parseHandCount(raw.hand_count);
  const handDetected = raw.hand_count === "1" && !raw.is_ambiguous;
  const gesture = mapCommandToGesture(raw.command);
  const cameraInfo = raw.camera_info;
  const serialInfo = raw.serial_info;

  return {
    command: raw.command,
    gesture,
    confidence: handDetected ? raw.confidence : 0,
    isStopActive: raw.is_stop_active,
    isAmbiguous: raw.is_ambiguous,
    handDetected,
    handCount,
    handedness: handDetected ? mapHandedness(raw.handedness) : null,
    fps: Math.round(raw.fps),
    latencyMs: Math.round(raw.latency_ms),
    pipelineState: raw.pipeline_state,
    modelUsed: raw.model_used,
    rawPrediction: raw.raw_prediction,
    routingStr: raw.routing_str,
    safetyStatus: raw.safety_status,
    // Both of these come straight from the pipeline snapshot: the UI reflects the
    // backend's real arming/transmission state instead of inferring it locally.
    isArmed: Boolean(raw.is_armed),
    isTxPermitted: Boolean(raw.is_tx_permitted),
    finalCommand: raw.final_command ?? raw.command,
    transportStatus: raw.transport_status,
    // `serial_info` is the pipeline's own serial-owner view of the last transmitted frame;
    // the top-level `last_tx` is the same value and is used as a fallback.
    lastTx:
      serialInfo?.last_tx && serialInfo.last_tx !== "NONE" ? serialInfo.last_tx : raw.last_tx || "NONE",
    // The pipeline's `serial_info` is authoritative; the transport string is only a fallback
    // for older payloads (it cannot distinguish a mock link from real hardware).
    serialPort: serialInfo?.port || extractSerialPort(raw.transport_status),
    baudRate: serialInfo?.baud || resolveBaudRate(raw.transport_status),
    serialConnected: serialInfo ? Boolean(serialInfo.connected) : isTransportConnected(raw.transport_status),
    serialMode: serialInfo?.mode ?? (isTransportConnected(raw.transport_status) ? "real" : "disabled"),
    framesSent: serialInfo?.frames_sent ?? 0,

    cameraStatus: deriveCameraStatus(cameraInfo),
    cameraResolution: cameraInfo?.resolution || DEFAULT_CAMERA_RESOLUTION,
    cameraIndex: cameraInfo?.index ?? 0,
    cameraEnabled: cameraInfo ? Boolean(cameraInfo.enabled) : true,
    cameraError: cameraInfo?.error ?? null,
    landmarksCount: raw.landmarks_count,
    landmarks: parseLandmarks(raw.landmarks),

    voice: raw.voice_info ?? DEFAULT_PIPELINE_STATE.voice,
    errors: Array.isArray(raw.errors) ? raw.errors : [],

    message: raw.message,
  };
}

/* ------------------------------------------------------------------ */
/* Store                                                               */
/* ------------------------------------------------------------------ */

type StoreListener = () => void;

const storeListeners = new Set<StoreListener>();
let snapshot: PipelineState = DEFAULT_PIPELINE_STATE;
let detachRaw: (() => void) | null = null;

function publish(next: PipelineState): void {
  snapshot = next;
  storeListeners.forEach((l) => {
    try {
      l();
    } catch (err) {
      console.error("[pipelineStore] listener failed", err);
    }
  });
}

function attachRaw(): void {
  if (detachRaw) return;
  detachRaw = subscribePipelineState((raw) => publish(derivePipelineState(raw)));
}

function detachRawIfIdle(): void {
  if (storeListeners.size > 0 || !detachRaw) return;
  detachRaw();
  detachRaw = null;
  // Drop back to the pre-connection defaults so a remount cannot show stale telemetry.
  snapshot = DEFAULT_PIPELINE_STATE;
}

/** Subscribe to derived pipeline state. Ref-counted: one raw IPC subscription overall. */
export function subscribeDerivedPipelineState(cb: StoreListener): () => void {
  storeListeners.add(cb);
  attachRaw();
  return () => {
    storeListeners.delete(cb);
    detachRawIfIdle();
  };
}

/**
 * Current derived snapshot. Must return a cached reference — `useSyncExternalStore`
 * compares by identity, so building a fresh object here would loop forever.
 */
export function getPipelineSnapshot(): PipelineState {
  return snapshot;
}

/** Camera descriptor derived from the same single source of truth. */
export function getPipelineCamera(): CameraInfo {
  return {
    name: "NeuroGrip Camera (Python OpenCV)",
    kind: "external",
    resolution: snapshot.cameraResolution || DEFAULT_CAMERA_RESOLUTION,
    status: snapshot.cameraStatus === CAMERA_STATUS_CONNECTED ? "connected" : "disconnected",
    deviceId: "python-opencv-0",
  };
}

/** Subscriber count — diagnostics only. */
export function pipelineStoreSubscriberCount(): number {
  return storeListeners.size;
}
