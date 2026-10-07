/**
 * `pipelineControl` — the renderer's control channel to the **running Python pipeline**.
 *
 * Phase 2: before this existed, the UI changed camera/serial state by spawning throwaway
 * `python -c` processes (a second serial owner) or by only mutating local React state. Every
 * function here sends a control request over the single persistent bridge connection, so the
 * pipeline stays the one authority for camera, serial, commands, STOP and voice.
 *
 * The action names are the contract with `pc/src/neurogrip/bridge/control.py::Action`;
 * `pc/tests/test_pipeline_control_contract.py` fails if the two lists ever drift apart.
 *
 * Every call resolves — `{ ok: false, error }` when no pipeline is connected — so callers
 * never need a try/catch and the UI can show a real failure instead of hanging.
 */
import type { ControlEnvelope } from "../types/desktop";

export const CONTROL_ACTIONS = {
  CAMERA_LIST: "camera.list",
  CAMERA_SELECT: "camera.select",
  CAMERA_SET_ENABLED: "camera.set_enabled",
  SERIAL_LIST_PORTS: "serial.list_ports",
  SERIAL_CONNECT: "serial.connect",
  SERIAL_DISCONNECT: "serial.disconnect",
  COMMAND_SEND: "command.send",
  STOP_SET_ARMED: "stop.set_armed",
  VOICE_SET_STATE: "voice.set_state",
  VOICE_TRANSCRIPT: "voice.transcript",
  PIPELINE_SNAPSHOT: "pipeline.snapshot",
  APP_SHUTDOWN: "app.shutdown",
} as const;

export type ControlAction = (typeof CONTROL_ACTIONS)[keyof typeof CONTROL_ACTIONS];

/** All actions, in contract order (used by the cross-language contract test). */
export const ALL_CONTROL_ACTIONS: readonly string[] = Object.values(CONTROL_ACTIONS);

/** A camera the *pipeline* can actually open. */
export interface CameraDescriptor {
  index: number;
  label: string;
  available: boolean;
  active: boolean;
}

/** Truthful camera state reported by the pipeline. */
export interface CameraInfoPayload {
  index: number;
  enabled: boolean;
  opened: boolean;
  state: string;
  backend: string;
  resolution: string;
  target_resolution: string;
  error: string | null;
}

/** Truthful serial state reported by the pipeline. */
export interface SerialInfoPayload {
  mode: "real" | "mock" | "disabled";
  port: string;
  baud: number;
  connected: boolean;
  state: string;
  frames_sent: number;
  last_tx: string;
}

/** Voice state recorded inside the pipeline. */
export interface VoiceInfoPayload {
  state: string;
  transcript: string;
  command: string | null;
  reply: string;
  engine: string;
}

/** Result of a voice transcript interpreted by the pipeline. */
export interface VoiceTranscriptPayload {
  transcript: string;
  intent: string;
  gesture: string | null;
  reply: string;
  command?: string | null;
  frame?: string | null;
  sent?: boolean;
  mode?: string;
  voice?: VoiceInfoPayload;
}

type Bridge = {
  control?: (action: string, payload?: Record<string, unknown>) => Promise<ControlEnvelope>;
};

function getBridge(): Bridge | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as unknown as { neurogrip?: Bridge }).neurogrip;
}

/** `true` when the desktop control channel is available (Electron preload). */
export function isPipelineControlAvailable(): boolean {
  return typeof getBridge()?.control === "function";
}

/**
 * Send one control request to the running pipeline.
 *
 * Never throws: a missing bridge, a disconnected pipeline and a rejected action all come
 * back as `{ ok: false, error }`.
 */
export async function controlRequest(
  action: ControlAction | string,
  payload: Record<string, unknown> = {},
): Promise<ControlEnvelope> {
  const bridge = getBridge();
  if (!bridge?.control) {
    return { ok: false, data: {}, error: "Desktop control channel unavailable (run the Electron app)." };
  }
  try {
    const res = await bridge.control(action, payload);
    if (!res || typeof res.ok !== "boolean") {
      return { ok: false, data: {}, error: "Malformed control response from the desktop bridge." };
    }
    return res;
  } catch (err) {
    return { ok: false, data: {}, error: err instanceof Error ? err.message : String(err) };
  }
}

/* ------------------------------------------------------------------ */
/* Typed helpers                                                       */
/* ------------------------------------------------------------------ */

export const listCameras = () => controlRequest(CONTROL_ACTIONS.CAMERA_LIST);

export const selectCamera = (index: number) =>
  controlRequest(CONTROL_ACTIONS.CAMERA_SELECT, { index });

export const setCameraEnabled = (enabled: boolean) =>
  controlRequest(CONTROL_ACTIONS.CAMERA_SET_ENABLED, { enabled });

export const listSerialPorts = () => controlRequest(CONTROL_ACTIONS.SERIAL_LIST_PORTS);

export const connectSerial = (port?: string, baud?: number, mode: "real" | "mock" = "real") =>
  controlRequest(CONTROL_ACTIONS.SERIAL_CONNECT, { port, baud, mode });

export const disconnectSerial = () => controlRequest(CONTROL_ACTIONS.SERIAL_DISCONNECT);

/** The one authoritative path for manual (button) commands. */
export const sendCommand = (command: string, source = "manual") =>
  controlRequest(CONTROL_ACTIONS.COMMAND_SEND, { command, source });

export const setStopArmed = (armed: boolean) => controlRequest(CONTROL_ACTIONS.STOP_SET_ARMED, { armed });

export const setVoiceState = (state: string, engine?: string) =>
  controlRequest(CONTROL_ACTIONS.VOICE_SET_STATE, { state, engine });

/** Ask the pipeline to interpret a transcript and transmit any resulting command. */
export const sendVoiceTranscript = (text: string, engine?: string) =>
  controlRequest(CONTROL_ACTIONS.VOICE_TRANSCRIPT, { text, engine });

export const getPipelineSnapshot = () => controlRequest(CONTROL_ACTIONS.PIPELINE_SNAPSHOT);

/* ------------------------------------------------------------------ */
/* Reading typed data out of an envelope                               */
/* ------------------------------------------------------------------ */

export function camerasFrom(res: ControlEnvelope): CameraDescriptor[] {
  const cameras = (res.data as { cameras?: unknown }).cameras;
  return Array.isArray(cameras) ? (cameras as CameraDescriptor[]) : [];
}

export function serialInfoFrom(res: ControlEnvelope): SerialInfoPayload | null {
  const serial = (res.data as { serial?: unknown }).serial;
  return serial && typeof serial === "object" ? (serial as SerialInfoPayload) : null;
}

export function cameraInfoFrom(res: ControlEnvelope): CameraInfoPayload | null {
  const camera = (res.data as { camera?: unknown }).camera;
  return camera && typeof camera === "object" ? (camera as CameraInfoPayload) : null;
}

/** The pipeline's own arming state echoed back by `stop.set_armed` (null when absent). */
export function stopArmedFrom(res: ControlEnvelope): boolean | null {
  const armed = (res.data as { armed?: unknown }).armed;
  return typeof armed === "boolean" ? armed : null;
}

export function voiceTranscriptFrom(res: ControlEnvelope): VoiceTranscriptPayload | null {
  const data = res.data as Partial<VoiceTranscriptPayload>;
  return typeof data?.intent === "string" ? (data as VoiceTranscriptPayload) : null;
}
