import type { EngineStatus } from "../services/types";

/** Bridge exposed by electron/preload.cjs. Undefined when running in a normal browser. */
export interface NeuroGripBridge {
  isDesktop: true;
  platform: string;
  stt: {
    warmup(): Promise<EngineStatus>;
    getStatus(): Promise<EngineStatus>;
    transcribe(pcm: Float32Array): Promise<{ text: string; ms: number; error?: string }>;
    onStatus(cb: (s: EngineStatus) => void): () => void;
  };
  onPipelineState(cb: (s: Record<string, unknown>) => void): () => void;
  /**
   * Phase 2 control channel: acts on the *running* Python pipeline (camera, serial,
   * commands, STOP, voice). Never opens a second process or serial port.
   */
  control(action: string, payload?: Record<string, unknown>): Promise<ControlEnvelope>;
  onControlResult(cb: (result: ControlEnvelope) => void): () => void;
  onCameraFrame(cb: (msg: { data: string; ts?: number }) => void): () => void;
  serialPorts(): Promise<string[]>;
  serialConnect(port?: string, baud?: number): Promise<{ ok: boolean; port?: string; baud?: number; error?: string }>;
  serialDisconnect(): Promise<{ ok: boolean; port?: string; baud?: number; error?: string }>;
  serialSend(command: string, source?: "detector" | "override" | "voice" | "manual"): Promise<{ ok: boolean; command?: string; frame?: string; output?: string; error?: string }>;
  onBridgeStatus(cb: (status: { connected: boolean }) => void): () => void;
}

/** Result of one control request (`{"type":"control_result", ...}` on the wire). */
export interface ControlEnvelope {
  ok: boolean;
  id?: string;
  data: Record<string, unknown>;
  error: string | null;
}

declare global {
  interface Window {
    neurogrip?: NeuroGripBridge;
  }
}
