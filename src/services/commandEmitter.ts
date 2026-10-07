/**
 * `createCommandEmitter` — the framework-free core of `useGestureCommand`.
 *
 * All of the "one action must produce exactly one hardware frame" policy lives here so it
 * can be reasoned about (and verified) without a DOM or a React renderer:
 *
 *   1. **Re-entrancy guard** — if an adapter synchronously fans a frame back out while we
 *      are still inside an emission, the echo of the same gesture is refused. This is the
 *      second line of defence for the `desktopVision.overrideGesture → synthetic frame →
 *      subscriber → emit` loop that used to double-send every manual override; the first is
 *      that the preview frame carries `OVERRIDE_MODEL_TAG` and is skipped on sight.
 *   2. **Duplicate window** — the same gesture within `dedupeMs` is refused.
 *   3. **Detector change-only** — pipeline frames only transmit when the gesture actually
 *      changes, so a steady camera feed cannot spam the actuator bus.
 *
 * The emitter never talks to hardware itself; it delegates to the injected `send`, which in
 * the app is `SerialAdapter.send` → Electron bridge → the running Python pipeline.
 */
import type { Gesture } from "./types";
import { isTransmittableGesture } from "./pipelineMapping";

export type CommandSource = "detector" | "override" | "voice";

export type EmitReason = "accepted" | "rejected" | "reentrant" | "duplicate" | "unchanged";

export interface EmitResult {
  accepted: boolean;
  reason: EmitReason;
}

export interface CommandStats {
  dispatched: number;
  suppressedReentrant: number;
  suppressedDuplicate: number;
  rejected: number;
}

export interface CommandEmitterOptions {
  send: (gesture: Gesture, source: CommandSource) => void | Promise<void>;
  onSent?: (gesture: Gesture, source: CommandSource) => void;
  /** Minimum gap before the same gesture may be transmitted again. */
  dedupeMs?: number;
  /** Injectable clock (tests). */
  now?: () => number;
  /** Injectable validator (tests / future policy changes). */
  isTransmittable?: (gesture: unknown) => boolean;
  onError?: (err: unknown) => void;
}

export interface CommandEmitter {
  emit: (gesture: string, source?: CommandSource) => EmitResult;
  /** Last gesture accepted for transmission. */
  lastEmitted: () => Gesture | null;
  stats: () => CommandStats;
  /** Clear all suppression state (e.g. when the camera is powered off). */
  reset: () => void;
}

export const DEFAULT_DEDUPE_MS = 300;

export function createCommandEmitter(opts: CommandEmitterOptions): CommandEmitter {
  const dedupeMs = opts.dedupeMs ?? DEFAULT_DEDUPE_MS;
  const now = opts.now ?? (() => Date.now());
  const isTransmittable = opts.isTransmittable ?? isTransmittableGesture;

  let depth = 0;
  let last: { gesture: Gesture; at: number } | null = null;
  let lastEmitted: Gesture | null = null;
  const counters: CommandStats = { dispatched: 0, suppressedReentrant: 0, suppressedDuplicate: 0, rejected: 0 };

  const emit = (gesture: string, source: CommandSource = "detector"): EmitResult => {
    const name = String(gesture ?? "").trim().toUpperCase();

    if (!isTransmittable(name)) {
      counters.rejected += 1;
      return { accepted: false, reason: "rejected" };
    }

    // Safe: `isTransmittable` validated `name` against the locked 13-gesture taxonomy.
    const valid: Gesture = name as Gesture;
    const at = now();

    // 1. Echo of the emission we are currently inside.
    if (depth > 0 && last?.gesture === valid) {
      counters.suppressedDuplicate += 1;
      return { accepted: false, reason: "reentrant" };
    }

    // 2. Detector frames only transmit on a genuine change. Checked before the duplicate
    //    window so a steady camera feed reports the more informative "unchanged" reason.
    if (source === "detector" && lastEmitted === valid) {
      counters.suppressedDuplicate += 1;
      return { accepted: false, reason: "unchanged" };
    }

    // 3. Same gesture again inside the de-duplication window.
    if (last && last.gesture === valid && at - last.at < dedupeMs) {
      counters.suppressedDuplicate += 1;
      return { accepted: false, reason: "duplicate" };
    }

    last = { gesture: valid, at };
    lastEmitted = valid;
    counters.dispatched += 1;

    // Enter the guard *before* invoking `send`: adapters may fan out synchronously, which
    // would otherwise re-enter this function while the transmission is still in flight.
    depth += 1;
    try {
      try {
        const result = opts.send(valid, source);
        if (result && typeof (result as Promise<void>).then === "function") {
          (result as Promise<void>).catch((err) => (opts.onError ?? console.error)(err));
        }
      } catch (err) {
        (opts.onError ?? console.error)(err);
      }
      opts.onSent?.(valid, source);
    } finally {
      depth -= 1;
    }

    return { accepted: true, reason: "accepted" };
  };

  return {
    emit,
    lastEmitted: () => lastEmitted,
    stats: () => ({ ...counters }),
    reset: () => {
      depth = 0;
      last = null;
      lastEmitted = null;
    },
  };
}
