/**
 * Electron ↔ NeuroGrip pipeline bridge (no Electron dependency — unit-testable).
 *
 * The Python pipeline connects to this TCP server as a client and keeps one connection
 * open for the whole session. That single connection carries:
 *
 *   Python → Electron   state snapshots, JPEG preview frames, control results
 *   Electron → Python   control requests (camera / serial / command / STOP / voice)
 *
 * Framing is one JSON object per line, UTF-8, "\n" terminated — identical to the Python
 * side (`neurogrip/bridge/tcp_bridge.py`).
 *
 * All hardware access goes through this channel: the renderer can only ever act on the
 * running pipeline, never on a second process that owns the camera or the serial port.
 */
"use strict";

const net = require("net");

/** How long to wait for a control_result before failing the caller. */
const DEFAULT_CONTROL_TIMEOUT_MS = 6000;

/**
 * @param {object} options
 * @param {number} [options.port]           TCP port to listen on (default 8765)
 * @param {string} [options.host]           interface to bind (default 127.0.0.1)
 * @param {number} [options.controlTimeoutMs]
 * @param {(state: object) => void} [options.onState]         state snapshot received
 * @param {(msg: object) => void} [options.onFrame]           preview frame received
 * @param {(connected: boolean) => void} [options.onStatus]   pipeline connect/disconnect
 * @param {(result: object) => void} [options.onControlResult] late/unsolicited result
 * @param {(message: string) => void} [options.log]
 */
function createPipelineBridge(options) {
  const opts = options || {};
  const port = Number(opts.port || 8765);
  const host = opts.host || "127.0.0.1";
  const controlTimeoutMs = Number(opts.controlTimeoutMs || DEFAULT_CONTROL_TIMEOUT_MS);
  const log = opts.log || ((m) => console.log(`[TCP Bridge] ${m}`));
  const logError = opts.logError || ((m) => console.error(`[TCP Bridge] ${m}`));

  let server = null;
  let socket = null;
  let buffer = "";
  let seq = 0;
  /** id → { resolve, timer } for in-flight control requests. */
  const pending = new Map();

  function notifyStatus(connected) {
    if (opts.onStatus) opts.onStatus(Boolean(connected));
  }

  function handleLine(line) {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch (e) {
      logError(`JSON parse error: ${e.message}`);
      return;
    }

    if (msg.type === "frame" && msg.data) {
      if (opts.onFrame) opts.onFrame(msg);
      return;
    }

    if (msg.type === "control_result") {
      const key = String(msg.id);
      const entry = pending.get(key);
      if (entry) {
        pending.delete(key);
        clearTimeout(entry.timer);
        entry.resolve(msg);
      } else if (opts.onControlResult) {
        // Late reply after a timeout: surface it instead of silently dropping it.
        opts.onControlResult(msg);
      }
      return;
    }

    if (opts.onState) opts.onState(msg);
  }

  function failPending(reason) {
    for (const [id, entry] of pending) {
      clearTimeout(entry.timer);
      entry.resolve({ ok: false, data: {}, error: reason });
      pending.delete(id);
    }
  }

  function dropSocket(reason) {
    if (socket) log(reason);
    socket = null;
    buffer = "";
    failPending("NeuroGrip pipeline disconnected.");
    notifyStatus(false);
  }

  function handleConnection(conn) {
    log("pipeline connected");
    // An accepted socket never emits "connect" — report the real state right here, or the
    // UI's backend indicator could never turn on.
    socket = conn;
    conn.setNoDelay(true);
    notifyStatus(true);

    conn.on("data", (data) => {
      buffer += data.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (line.trim()) handleLine(line);
      }
    });

    conn.on("error", (err) => {
      logError(`socket error: ${err.message}`);
      if (socket === conn) dropSocket("pipeline socket error");
    });

    conn.on("close", () => {
      log("pipeline disconnected");
      if (socket === conn) dropSocket("pipeline disconnected");
    });
  }

  function start() {
    if (server) return;
    server = net.createServer(handleConnection);
    server.on("error", (err) => logError(`server error: ${err.message}`));
    server.listen(port, host, () => log(`listening on ${host}:${port}`));
  }

  function stop() {
    if (server) {
      server.close();
      server = null;
    }
    if (socket) {
      socket.destroy();
      socket = null;
    }
    failPending("NeuroGrip pipeline disconnected.");
  }

  /** `true` while a pipeline is connected. */
  function isConnected() {
    return Boolean(socket && !socket.destroyed);
  }

  /**
   * Send one control request to the running pipeline.
   * Always resolves — with `{ok:false,error}` when no pipeline is connected or it does not
   * answer in time — so the renderer can never hang on a dead backend.
   *
   * @returns {Promise<{ok: boolean, id?: string, data?: object, error?: string|null}>}
   */
  function request(action, payload) {
    return new Promise((resolve) => {
      if (!isConnected()) {
        resolve({ ok: false, data: {}, error: "NeuroGrip pipeline is not connected (port " + port + ")." });
        return;
      }

      const id = `ui-${++seq}`;
      const timer = setTimeout(() => {
        pending.delete(id);
        resolve({ ok: false, id, data: {}, error: `Pipeline did not answer '${action}' in time.` });
      }, controlTimeoutMs);

      pending.set(id, { resolve, timer });
      try {
        socket.write(JSON.stringify({ type: "control", id, action, payload: payload || {} }) + "\n");
      } catch (e) {
        clearTimeout(timer);
        pending.delete(id);
        resolve({ ok: false, id, data: {}, error: e.message });
      }
    });
  }

  /** The port actually bound (useful when constructed with `port: 0` in tests). */
  function boundPort() {
    const address = server && server.address();
    return address && typeof address === "object" ? address.port : port;
  }

  return { start, stop, request, isConnected, boundPort, port, host };
}

module.exports = { createPipelineBridge, DEFAULT_CONTROL_TIMEOUT_MS };
