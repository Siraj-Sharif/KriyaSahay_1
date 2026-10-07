/**
 * React hook for consuming real-time pipeline state from the Python bridge via Electron IPC.
 *
 * `usePipelineState` is the **single source of truth** for live pipeline state: every
 * consumer reads the same snapshot from `pipelineStore`, which is fed by the single IPC
 * subscription owned by `pipelineSource`. Mounting this hook in eight components now costs
 * one subscription instead of eight, and there is exactly one place where raw bridge values
 * are interpreted (`services/pipelineStore.ts`).
 *
 * Public API and return shape are unchanged from before Phase 1.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { Gesture } from "../services/types";
import { GESTURE_DEFS } from "../services/gestures";
import {
  getPipelineSnapshot,
  subscribeDerivedPipelineState,
  type PipelineState,
} from "../services/pipelineStore";

export type { PipelineState } from "../services/pipelineStore";
export { DEFAULT_PIPELINE_STATE as DEFAULT_STATE } from "../services/pipelineStore";

const GESTURES = Object.keys(GESTURE_DEFS) as Gesture[];

/**
 * Single source of truth for live pipeline state.
 *
 * Reads a shared snapshot via `useSyncExternalStore`: all call sites receive the same
 * object reference, so React skips the re-render unless a new frame actually arrived.
 */
export function usePipelineState(): PipelineState {
  return useSyncExternalStore(
    subscribeDerivedPipelineState,
    getPipelineSnapshot,
    getPipelineSnapshot,
  );
}

/**
 * Hook for confidence history - maintains rolling buffer of real confidence values
 */
export function useConfidenceHistory(maxPoints = 120) {
  const [history, setHistory] = useState<Array<{ t: number; confidence: number; fps: number }>>([]);
  const pipelineState = usePipelineState();
  const lastTimeRef = useRef(0);

  useEffect(() => {
    if (pipelineState.pipelineState === "STARTING" && pipelineState.fps === 0 && !pipelineState.handDetected) {
      return;
    }

    const now = Date.now();
    if (now - lastTimeRef.current < 250) return;
    lastTimeRef.current = now;

    const confVal = pipelineState.handDetected ? Math.round(pipelineState.confidence * 100) : 0;
    setHistory((prev) => {
      const next = [
        ...prev,
        {
          t: now,
          confidence: confVal,
          fps: pipelineState.fps || 0,
        },
      ];
      return next.slice(-maxPoints);
    });
  }, [pipelineState.handDetected, pipelineState.confidence, pipelineState.fps, pipelineState.pipelineState, maxPoints]);

  return history;
}

/**
 * Hook for gesture counts from real pipeline
 */
export function useGestureCounts() {
  const [counts, setCounts] = useState<Record<Gesture, number>>(
    () => Object.fromEntries(GESTURES.map(g => [g, 0])) as Record<Gesture, number>
  );
  const pipelineState = usePipelineState();
  const lastSentRef = useRef<Gesture | null>(null);

  useEffect(() => {
    if (pipelineState.handDetected && pipelineState.confidence > 0.7 && pipelineState.gesture !== lastSentRef.current) {
      lastSentRef.current = pipelineState.gesture;
      setCounts(prev => ({ ...prev, [pipelineState.gesture]: prev[pipelineState.gesture] + 1 }));
    }
  }, [pipelineState.handDetected, pipelineState.confidence, pipelineState.gesture]);

  return counts;
}
