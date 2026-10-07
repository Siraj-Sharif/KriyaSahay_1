/**
 * Desktop Serial Adapter — talks to the Python pipeline's serial transport through the
 * Electron bridge.
 *
 * Phase 1: it no longer opens its own `onPipelineState` subscription and no longer keeps a
 * second copy of the transport parsing.
 *
 * Phase 2: `send()` reaches the hardware through `serialSend` → the persistent control
 * channel → the pipeline's own serial owner (the old path spawned a `python -c` process
 * that opened the same COM port behind the pipeline's back). Connection state, port, baud
 * and the transmitted-frame counter now come from the pipeline's `serial_info`.
 */
import type { CommandSourceTag, SerialAdapter, Unsubscribe } from "./adapters";
import type { Gesture, SerialState } from "./types";
import { encodeCommand } from "./gestures";
import { extractSerialPort, isTransportConnected, type RawPipelineState } from "./pipelineMapping";
import { subscribePipelineState } from "./pipelineSource";

export function createDesktopSerial(): SerialAdapter {
  const listeners = new Set<(s: SerialState) => void>();
  let detachRaw: Unsubscribe | null = null;
  let state: SerialState = {
    connected: false,
    port: "--",
    baudRate: 115200,
    tx: "idle",
    lastCommand: "NONE",
    packetsSent: 0,
    lastAckMs: 0,
  };

  const emit = () => {
    listeners.forEach((l) => l({ ...state }));
  };

  /**
   * Fold one pipeline frame into the serial view state (only emits when it changed).
   *
   * Phase 2: `serial_info` from the pipeline is authoritative — it is the only source that
   * can distinguish a real COM link from a mock one, and it carries the pipeline's own
   * transmitted-frame counter. The transport string remains a fallback for old payloads.
   */
  const applyPipelineState = (data: RawPipelineState) => {
    const info = data.serial_info;
    const isConnected = info ? Boolean(info.connected) : isTransportConnected(data.transport_status);
    const parsedPort = info?.port || extractSerialPort(data.transport_status);
    const baudRate = info?.baud || state.baudRate;

    let changed = false;
    if (state.connected !== isConnected) {
      state.connected = isConnected;
      changed = true;
    }
    if (parsedPort && parsedPort !== "--" && state.port !== parsedPort) {
      state.port = parsedPort;
      changed = true;
    }
    if (baudRate && state.baudRate !== baudRate) {
      state.baudRate = baudRate;
      changed = true;
    }
    if (info && state.packetsSent !== info.frames_sent) {
      // Real count of frames the pipeline wrote, not a locally guessed one.
      state.packetsSent = info.frames_sent;
      changed = true;
    }
    const lastTx = info?.last_tx || data.last_tx || "NONE";
    if (lastTx !== "NONE" && state.lastCommand !== lastTx) {
      state.lastCommand = lastTx;
      changed = true;
    }

    if (changed) emit();
  };

  const ensureSubscribed = () => {
    if (detachRaw) return;
    detachRaw = subscribePipelineState((data) => {
      applyPipelineState(data);
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
      cb({ ...state });
      return () => {
        listeners.delete(cb);
        releaseIfIdle();
      };
    },
    async send(command: Gesture, source?: CommandSourceTag) {
      const cmdStr = encodeCommand(command);
      state = { ...state, tx: "transmitting", lastCommand: cmdStr };
      emit();

      if (typeof window !== "undefined" && (window as any).neurogrip?.serialSend) {
        try {
          const res = await (window as any).neurogrip.serialSend(command, source);
          state = {
            ...state,
            // Always reflect the pipeline's own report; the next state snapshot also
            // carries `serial_info.frames_sent`, so the counter cannot drift.
            tx: res && res.ok ? "idle" : "error",
            lastCommand: (res && res.frame) || cmdStr,
          };
        } catch {
          state = { ...state, tx: "error" };
        }
      } else {
        state = { ...state, tx: "idle" };
      }
      emit();
    },
    async connect(port?: string, baud?: number) {
      const targetPort = port || state.port;
      const targetBaud = baud || state.baudRate;
      if (typeof window !== "undefined" && (window as any).neurogrip?.serialConnect) {
        try {
          const res = await (window as any).neurogrip.serialConnect(targetPort, targetBaud);
          state = {
            ...state,
            connected: Boolean(res && res.ok),
            port: (res && res.port) || targetPort,
            baudRate: (res && res.baud) || targetBaud,
          };
        } catch {
          state = { ...state, connected: false };
        }
        emit();
      }
    },
    async disconnect() {
      if (typeof window !== "undefined" && (window as any).neurogrip?.serialDisconnect) {
        await (window as any).neurogrip.serialDisconnect();
      }
      state = { ...state, connected: false, tx: "idle" };
      emit();
    },
  };
}
