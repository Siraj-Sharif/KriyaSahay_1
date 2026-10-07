/**
 * `useGestureCommand()` — the single owner of gesture-command emission.
 *
 * Phase 1 background: commands reached the hardware from several unrelated places
 * (`SystemContext.setOverrideGesture`, the `vision.subscribe` callback, and the voice
 * adapter callback), and `desktopVision.overrideGesture()` emitted an untagged synthetic
 * frame into the very subscriber that emitted commands. The result was a double send for
 * every manual override and no single place to answer "what did we transmit?".
 *
 * The override preview frame is now tagged `MANUAL_OVERRIDE`, so it is never mistaken for a
 * detection, and the explicit override emission transmits exactly once.
 *
 * Every emission now goes through `emit()` here. The policy (re-entrancy guard, duplicate
 * window, detector change-only rule) lives in `createCommandEmitter` so it is unit-testable
 * without React; this hook only supplies React state around it.
 *
 * Transmission is delegated to the `SerialAdapter`, which in desktop mode is a thin wrapper
 * around the Electron bridge → Python pipeline. Nothing here talks to the hardware
 * directly, so the running Python pipeline remains the source of truth.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SerialAdapter, VisionAdapter } from "../services/adapters";
import type { Gesture } from "../services/types";
import { OVERRIDE_MODEL_TAG } from "../services/pipelineMapping";
import {
  createCommandEmitter,
  DEFAULT_DEDUPE_MS,
  type CommandEmitter,
  type CommandSource,
  type CommandStats,
} from "../services/commandEmitter";

export type { CommandSource, CommandStats } from "../services/commandEmitter";
/** Re-exported so callers have one import site for the synthetic-frame marker. */
export { OVERRIDE_MODEL_TAG };

export interface GestureCommandOptions {
  vision: VisionAdapter;
  serial: SerialAdapter;
  /** Fires once per accepted emission, after the adapter has been asked to send. */
  onSent?: (gesture: Gesture, source: CommandSource) => void;
  dedupeMs?: number;
}

export interface GestureCommandApi {
  /** The one path a gesture command may take to the hardware. */
  emit: (gesture: Gesture, source?: CommandSource) => boolean;
  /** Convenience wrapper for frames coming from the vision pipeline. */
  emitFromFrame: (frame: { gesture: Gesture; handDetected: boolean; confidence: number; model?: string }) => boolean;
  /** Set/clear the manual override and transmit it exactly once. */
  setOverride: (gesture: Gesture | null) => void;
  /**
   * Set/clear the manual override **without transmitting**.
   *
   * Used when the authoritative owner has already transmitted the command (desktop voice:
   * the pipeline interprets `voice.transcript` and writes the frame itself). Calling
   * `setOverride` there would put a second identical frame on the wire.
   */
  previewOverride: (gesture: Gesture | null) => void;
  /** Currently active manual override, if any. */
  override: Gesture | null;
  /** `true` while a transmission is in flight. */
  isSending: boolean;
  /** Last gesture accepted for transmission. */
  lastEmitted: Gesture | null;
  /** How many commands were actually handed to the serial adapter. */
  sentCount: number;
  /** Clear suppression history (e.g. after the camera is powered off). */
  reset: () => void;
  /**
   * Diagnostics for the suppression rules.
   *
   * Deliberately a getter, not React state: `emitFromFrame` runs on every pipeline frame
   * (~30 fps) and suppression counts change constantly, so publishing them as state would
   * re-render the whole app at frame rate.
   */
  getStats: () => CommandStats;
}

export function useGestureCommand({ vision, serial, onSent, dedupeMs = DEFAULT_DEDUPE_MS }: GestureCommandOptions): GestureCommandApi {
  const [override, setOverrideState] = useState<Gesture | null>(null);
  const [isSending, setSending] = useState(false);
  const [lastEmitted, setLastEmitted] = useState<Gesture | null>(null);
  const [sentCount, setSentCount] = useState(0);

  const sendingRef = useRef(0);

  // Fresh callbacks without changing the emitter's identity.
  const onSentRef = useRef(onSent);
  useEffect(() => {
    onSentRef.current = onSent;
  }, [onSent]);

  const serialRef = useRef(serial);
  serialRef.current = serial;
  const onErrorRef = useRef<(err: unknown) => void>((err) => console.error("[useGestureCommand] serial send failed", err));

  /**
   * The emitter instance is created once and kept in a ref: it owns the re-entrancy and
   * de-duplication state, which must survive re-renders.
   */
  const emitterRef = useRef<CommandEmitter | null>(null);
  if (emitterRef.current === null) {
    emitterRef.current = createCommandEmitter({
      dedupeMs,
      send: (gesture, source) => {
        // Track the real in-flight window so the UI shows "transmitting" for the duration
        // of the adapter round-trip rather than for a single render.
        sendingRef.current += 1;
        setSending(true);

        let pending: Promise<void>;
        try {
          pending = Promise.resolve(serialRef.current.send(gesture, source));
        } catch (err) {
          sendingRef.current = Math.max(0, sendingRef.current - 1);
          if (sendingRef.current === 0) setSending(false);
          throw err;
        }

        const settle = () => {
          sendingRef.current = Math.max(0, sendingRef.current - 1);
          if (sendingRef.current === 0) setSending(false);
        };
        void pending.then(settle, settle);
        return pending;
      },
      onSent: (gesture, source) => {
        setLastEmitted(gesture);
        setSentCount((c) => c + 1);
        onSentRef.current?.(gesture, source);
      },
      onError: (err) => onErrorRef.current(err),
    });
  }
  const emitter = emitterRef.current;

  /**
   * Run the guarded emission policy.
   *
   * Note there is no `setState` on the suppressed path — only accepted transmissions touch
   * React state, so a steady camera feed cannot trigger re-renders.
   */
  const emit = useCallback<GestureCommandApi["emit"]>(
    (gesture, source = "detector") => emitter.emit(gesture, source).accepted,
    [emitter],
  );

  /**
   * Accept a frame from the vision pipeline.
   *
   * Synthetic override frames are ignored outright — the override already transmitted
   * itself in `setOverride`, and honouring the frame as well is exactly the double send
   * this hook exists to prevent. (The emitter's re-entrancy guard is the second line of
   * defence if a frame ever arrives without the marker.)
   */
  const emitFromFrame = useCallback<GestureCommandApi["emitFromFrame"]>(
    (frame) => {
      if (frame.model === OVERRIDE_MODEL_TAG) return false;
      if (!frame.handDetected || frame.confidence <= 0.7) return false;
      return emit(frame.gesture, "detector");
    },
    [emit],
  );

  /** Visual only: updates the 3D hand / camera preview. Never a command. */
  const previewOverride = useCallback(
    (gesture: Gesture | null) => {
      setOverrideState(gesture);
      vision.overrideGesture(gesture);
    },
    [vision],
  );

  const setOverride = useCallback(
    (gesture: Gesture | null) => {
      previewOverride(gesture);
      if (gesture) emit(gesture, "override");
    },
    [previewOverride, emit],
  );

  /** Forget suppression history so the next detection transmits again. */
  const reset = useCallback(() => {
    emitter.reset();
    setLastEmitted(null);
  }, [emitter]);

  const getStats = useCallback(() => emitter.stats(), [emitter]);

  return useMemo(
    () => ({
      emit,
      emitFromFrame,
      setOverride,
      previewOverride,
      override,
      isSending,
      lastEmitted,
      sentCount,
      reset,
      getStats,
    }),
    [emit, emitFromFrame, setOverride, previewOverride, override, isSending, lastEmitted, sentCount, reset, getStats],
  );
}
