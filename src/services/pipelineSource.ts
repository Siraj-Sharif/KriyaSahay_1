/**
 * Single owner of the `neurogrip.onPipelineState` IPC subscription.
 *
 * Phase 1 background: `usePipelineState` (instantiated ~8×), `desktopVision` and
 * `desktopSerial` each opened their *own* subscription to the same Electron channel, so a
 * single pipeline frame was parsed, mapped and fanned out three separate times.
 *
 * This module attach/detaches exactly one underlying subscription (reference-counted) and
 * broadcasts the raw payload to every consumer. It deliberately holds no derived state —
 * `usePipelineState` remains the single source of truth for *interpreted* pipeline state.
 */
import type { Unsubscribe } from "./adapters";
import type { RawPipelineState } from "./pipelineMapping";

type RawListener = (state: RawPipelineState) => void;

const listeners = new Set<RawListener>();
let bridgeUnsubscribe: Unsubscribe | null = null;
let lastState: RawPipelineState | null = null;

/** `true` when the Electron preload bridge exposes pipeline state. */
export function isPipelineSourceAvailable(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof (window as { neurogrip?: { onPipelineState?: unknown } }).neurogrip?.onPipelineState === "function"
  );
}

function attach(): void {
  if (bridgeUnsubscribe || !isPipelineSourceAvailable()) return;

  const bridge = (window as { neurogrip?: { onPipelineState?: (cb: (s: RawPipelineState) => void) => Unsubscribe } }).neurogrip;
  if (!bridge?.onPipelineState) return;

  // One IPC subscription for the whole app, for as long as anyone is listening.
  bridgeUnsubscribe = bridge.onPipelineState((state: RawPipelineState) => {
    lastState = state;
    listeners.forEach((l) => {
      try {
        l(state);
      } catch (err) {
        // A single bad consumer must not kill the fan-out for everyone else.
        console.error("[pipelineSource] listener failed", err);
      }
    });
  });
}

function detach(): void {
  if (!bridgeUnsubscribe) return;
  bridgeUnsubscribe();
  bridgeUnsubscribe = null;
  lastState = null;
}

/**
 * Subscribe to raw pipeline state. The first subscriber opens the IPC channel, the last
 * one to leave closes it — so React StrictMode double-mounting cannot leak listeners.
 */
export function subscribePipelineState(cb: RawListener): Unsubscribe {
  listeners.add(cb);
  attach();
  // Late subscribers immediately get the most recent frame, matching the previous
  // per-adapter behaviour of not waiting a full frame interval for the first value.
  if (lastState) cb(lastState);

  return () => {
    listeners.delete(cb);
    if (listeners.size === 0) detach();
  };
}

/** Most recent raw payload, or `null` before the first frame / after the last detach. */
export function getLastPipelineState(): RawPipelineState | null {
  return lastState;
}

/** Number of active raw subscribers (diagnostics / tests). */
export function pipelineSourceSubscriberCount(): number {
  return listeners.size;
}
