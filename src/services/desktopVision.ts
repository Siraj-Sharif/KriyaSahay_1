/**
 * Desktop Vision Adapter - connects to Python NeuroGrip pipeline via Electron IPC.
 * Receives real-time VisualizationState from the TCP bridge.
 *
 * Phase 1 changes (deliberately minimal):
 *   - it no longer opens its own `onPipelineState` subscription — it consumes the single
 *     app-wide one from `pipelineSource`;
 *   - it no longer carries a second copy of the raw→UI mapping — it uses the shared
 *     helpers from `pipelineMapping`;
 *   - `overrideGesture()` no longer emits an *untagged* synthetic frame that re-entered the
 *     command-emission path and produced a second serial write. The preview frame is now
 *     tagged with `OVERRIDE_MODEL_TAG`, so the emitter ignores it while the explicit
 *     override emission still transmits exactly once.
 *
 * Real pipeline frames are untouched by an active override (this adapter never masked them),
 * so live telemetry keeps flowing to the UI exactly as before. `createMockVision` does mask
 * them — that pre-existing difference is documented, not silently changed.
 *
 * The `VisionAdapter` contract is unchanged.
 */
import type { Unsubscribe, VisionAdapter } from "./adapters";
import type { CameraInfo, VisionFrame } from "./types";
import {
  OVERRIDE_MODEL_TAG,
  mapCommandToGesture,
  mapHandedness,
  parseHandCount,
  parseLandmarks,
  type RawPipelineState,
} from "./pipelineMapping";
import { subscribePipelineState } from "./pipelineSource";

export function createDesktopVision(): VisionAdapter {
  const listeners = new Set<(frame: VisionFrame) => void>();
  let enabled = true;
  let detachRaw: Unsubscribe | null = null;

  // Default camera info - will be updated if we get serial/transport info
  const camera: CameraInfo = {
    name: "NeuroGrip Camera",
    kind: "external",
    resolution: "1280 × 720",
    status: "connected",
    deviceId: "neurogrip-camera",
  };

  /** Translate one raw bridge payload into the frame shape the UI consumes. */
  const toFrame = (state: RawPipelineState): VisionFrame => {
    const handDetected = state.hand_count === "1" && !state.is_ambiguous;
    return {
      timestamp: Date.now(),
      gesture: mapCommandToGesture(state.command),
      confidence: handDetected ? state.confidence : 0,
      fps: Math.round(state.fps),
      latencyMs: Math.round(state.latency_ms),
      handDetected,
      handCount: parseHandCount(state.hand_count),
      handedness: handDetected ? mapHandedness(state.handedness) : null,
      model: state.model_used,
      landmarks: handDetected ? parseLandmarks(state.landmarks) : [],
    };
  };

  const emitFrame = (frame: VisionFrame) => {
    listeners.forEach((l) => {
      try {
        l(frame);
      } catch (err) {
        console.error("[desktopVision] listener failed", err);
      }
    });
  };

  /** Attach to the shared IPC stream only while somebody is actually listening. */
  const ensureSubscribed = () => {
    if (detachRaw) return;
    detachRaw = subscribePipelineState((state) => {
      if (!enabled) return;
      emitFrame(toFrame(state));
    });
  };

  const releaseIfIdle = () => {
    if (listeners.size > 0 || !detachRaw) return;
    detachRaw();
    detachRaw = null;
  };

  return {
    subscribe(cb) {
      listeners.add(cb);
      ensureSubscribed();
      return () => {
        listeners.delete(cb);
        releaseIfIdle();
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
    /**
     * Manually force a gesture preview.
     *
     * Emits exactly one *tagged* frame so the UI can show the pose immediately. The tag is
     * what stops the command-emission path from treating this preview as a detection and
     * sending a second frame; the real transmission is performed once by
     * `useGestureCommand.setOverride()`. Clearing the override simply stops injecting the
     * preview — the live pipeline is already on screen.
     */
    overrideGesture(g) {
      if (!g) return;
      emitFrame({
        timestamp: Date.now(),
        gesture: g,
        confidence: 1.0,
        fps: 30,
        latencyMs: 1,
        handDetected: true,
        handCount: 1,
        handedness: "Right",
        model: OVERRIDE_MODEL_TAG,
        landmarks: [],
      });
    },
  };
}
