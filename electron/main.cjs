const { app, BrowserWindow, ipcMain, session } = require("electron");
const path = require("path");
const stt = require("./stt.cjs");
const { createPipelineBridge } = require("./bridge.cjs");
const { createBackendSupervisor } = require("./backend.cjs");

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) app.quit();
app.setAppUserModelId("com.kriyasahay.desktop");

let win = null;
let isShuttingDown = false;
let backendRestarting = false;
let allowQuit = false;
let startupStateChecked = false;
let snapshotCheckPending = false;
let runtimeGeneration = 0;
let snapshotRetryTimer = null;
const BRIDGE_PORT = 8765;
const runtimeStatus = {
  bridge: "starting",
  backend: "starting",
  cv: "starting",
  message: "Starting the local control bridge…",
  pid: null,
  snapshot: null,
};
const INITIAL_PIPELINE_STATE = {
  command: "NO_COMMAND",
  confidence: 0,
  fps: 0,
  latency_ms: 0,
  hand_count: "0",
  pipeline_state: "STARTING",
  model_used: "N/A",
  raw_prediction: "N/A",
  routing_str: "N/A",
  transport_status: "DISABLED",
  last_tx: "NONE",
  is_stop_active: false,
  is_ambiguous: false,
  message: "Waiting for pipeline…",
  safety_status: "STOP DISARMED",
  handedness: "UNKNOWN",
  landmarks_count: 0,
  landmarks: [],
  is_armed: false,
  is_tx_permitted: false,
  final_command: "NO_COMMAND",
  camera_info: {
    index: 0,
    enabled: false,
    opened: false,
    state: "DISCONNECTED",
    backend: "opencv",
    resolution: "--",
    target_resolution: "--",
    error: null,
  },
  serial_info: {
    mode: "disabled",
    port: "",
    baud: 115200,
    connected: false,
    state: "DISCONNECTED",
    frames_sent: 0,
    last_tx: "NONE",
  },
};

function getRuntimeStatus() {
  return { ...runtimeStatus };
}

function sendToWindow(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function updateRuntimeStatus(patch) {
  Object.assign(runtimeStatus, patch);
  sendToWindow("neurogrip:runtime-status", getRuntimeStatus());
}

function normalizePipelineSnapshot(data) {
  const cvReady = data.cv_ready === true;
  return {
    initialized: data.initialized === true,
    running: data.running === true,
    cv_ready: cvReady,
    stop_armed: Boolean(data.stop_armed),
    state: String(data.state || "UNKNOWN"),
    camera: data.camera || undefined,
    serial: data.serial || undefined,
    voice: data.voice || undefined,
  };
}

function scheduleSnapshotCheck(delayMs = 180) {
  if (isShuttingDown || startupStateChecked || snapshotCheckPending || !bridge.isConnected()) return;
  clearTimeout(snapshotRetryTimer);
  snapshotRetryTimer = setTimeout(() => {
    snapshotRetryTimer = null;
    void checkPipelineSnapshot();
  }, delayMs);
}

async function checkPipelineSnapshot() {
  if (isShuttingDown || startupStateChecked || snapshotCheckPending || !bridge.isConnected()) return;
  snapshotCheckPending = true;
  const generation = runtimeGeneration;
  const res = await bridge.request("pipeline.snapshot", {});
  if (generation !== runtimeGeneration || isShuttingDown) return;
  snapshotCheckPending = false;

  const data = res && res.data ? res.data : {};
  if (res && res.ok && data.initialized === true && data.running === true) {
    startupStateChecked = true;
    const snapshot = normalizePipelineSnapshot(data);
    const cvReady = snapshot.cv_ready;
    updateRuntimeStatus({
      backend: "ready",
      cv: cvReady ? "ready" : "degraded",
      snapshot,
      message: cvReady
        ? "Python pipeline initialized; CV detector is ready."
        : "Python controls are live, but the CV detector did not initialize. Install/repair its model to resume camera tracking; serial and voice controls remain available.",
    });
    return;
  }

  if (res && res.ok && (data.initialized !== true || data.running !== true)) {
    // The control handler can register just before initialize() flips its lifecycle flags.
    scheduleSnapshotCheck(220);
    return;
  }

  if (res && res.error && /no handler registered/i.test(res.error)) {
    // The Python socket can connect before initialize() registers snapshot controls.
    scheduleSnapshotCheck(220);
    return;
  }

  if (bridge.isConnected()) {
    updateRuntimeStatus({
      backend: "error",
      cv: "error",
      snapshot: null,
      message: res?.error || "Python pipeline did not report an initialized, running state.",
    });
  }
}

/** Single owner of the pipeline connection: one socket, one pipeline, one serial port. */
const bridge = createPipelineBridge({
  port: BRIDGE_PORT,
  onState: (state) => {
    sendToWindow("neurogrip:state", state);
    if (!startupStateChecked && !snapshotCheckPending && state && typeof state.pipeline_state === "string") {
      void checkPipelineSnapshot();
    }
  },
  onFrame: (msg) => sendToWindow("neurogrip:frame", msg),
  onStatus: (connected) => {
    sendToWindow("neurogrip:bridge-status", { connected });
    // Clear the previous process's last frame so a restart/disconnect cannot look ready.
    sendToWindow("neurogrip:state", INITIAL_PIPELINE_STATE);
    runtimeGeneration += 1;
    snapshotCheckPending = false;
    startupStateChecked = false;
    clearTimeout(snapshotRetryTimer);
    snapshotRetryTimer = null;

    if (connected) {
      updateRuntimeStatus({
        bridge: "connected",
        backend: "starting",
        cv: "starting",
        snapshot: null,
        message: "Python connected; confirming initialized pipeline and CV readiness…",
      });
      scheduleSnapshotCheck(80);
    } else if (!isShuttingDown) {
      const alreadyFailed = runtimeStatus.backend === "error";
      updateRuntimeStatus({
        bridge: "disconnected",
        backend: alreadyFailed ? "error" : "starting",
        cv: alreadyFailed ? "error" : "starting",
        snapshot: null,
        message: alreadyFailed ? runtimeStatus.message : "Waiting for the Python pipeline to reconnect…",
      });
    }
  },
  onError: (error) => {
    if (!isShuttingDown) {
      updateRuntimeStatus({ bridge: "error", backend: "error", cv: "error", snapshot: null, message: `TCP bridge error: ${error.message}` });
    }
  },
  onControlResult: (msg) => sendToWindow("neurogrip:control-result", msg),
  log: (message) => console.log(`[BRIDGE] ${message}`),
  logError: (message) => console.error(`[BRIDGE] ${message}`),
});

const projectRoot = process.env.NEUROGRIP_PROJECT_ROOT || path.resolve(__dirname, "..");
const backend = createBackendSupervisor({
  projectRoot,
  onStatus: (status) => {
    if (isShuttingDown) return;
    if (status.state === "error") {
      startupStateChecked = true;
      updateRuntimeStatus({ backend: "error", cv: "error", snapshot: null, message: status.message, pid: null });
    } else if (status.state === "stopped") {
      updateRuntimeStatus(
        backendRestarting
          ? { backend: "starting", cv: "starting", snapshot: null, message: "Restarting Python backend…", pid: null }
          : {
              backend: "error",
              cv: "error",
              snapshot: null,
              message: "Python backend stopped unexpectedly. Use Retry backend to start it again.",
              pid: null,
            },
      );
    } else if (status.state === "starting" || status.state === "running") {
      if (runtimeStatus.backend !== "ready") {
        updateRuntimeStatus({ backend: "starting", cv: "starting", snapshot: null, message: status.message, pid: status.pid || null });
      }
    } else if (status.state === "stopping") {
      updateRuntimeStatus({ backend: "stopping", cv: "starting", message: status.message });
    }
  },
  onOutput: (stream, text) => {
    const prefix = stream === "stderr" ? "[PYTHON]" : "[PYTHON]";
    const trimmed = text.replace(/\s+$/, "");
    if (trimmed) (stream === "stderr" ? console.error : console.log)(`${prefix} ${trimmed}`);
  },
});

const SNAPSHOT_REFRESH_ACTIONS = new Set([
  "camera.select",
  "camera.set_enabled",
  "serial.connect",
  "serial.disconnect",
  "command.send",
  "stop.set_armed",
  "voice.transcript",
]);
let snapshotRefreshTimer = null;

async function refreshRuntimeSnapshot() {
  if (!startupStateChecked || isShuttingDown || !bridge.isConnected()) return;
  const generation = runtimeGeneration;
  const res = await bridge.request("pipeline.snapshot", {});
  if (generation !== runtimeGeneration || !res?.ok || !res.data?.initialized || !res.data?.running) return;
  const snapshot = normalizePipelineSnapshot(res.data);
  const cvReady = snapshot.cv_ready;
  updateRuntimeStatus({
    backend: "ready",
    cv: cvReady ? "ready" : "degraded",
    snapshot,
    message: cvReady
      ? "Python pipeline initialized; CV detector is ready."
      : runtimeStatus.message,
  });
}

function requestControl(action, payload) {
  return bridge.request(action, payload).then((res) => {
    if (res?.ok && SNAPSHOT_REFRESH_ACTIONS.has(action)) {
      clearTimeout(snapshotRefreshTimer);
      snapshotRefreshTimer = setTimeout(() => {
        snapshotRefreshTimer = null;
        void refreshRuntimeSnapshot();
      }, 80);
    }
    return res;
  });
}

/** Permissions the UI legitimately needs: camera, microphone, fullscreen. */
const ALLOWED = new Set(["media", "audioCapture", "videoCapture", "mediaKeySystem", "fullscreen", "clipboard-sanitized-write"]);

function setupPermissions() {
  const ses = session.defaultSession;
  ses.setPermissionRequestHandler((wc, permission, callback) => {
    callback(Boolean(win && wc === win.webContents && ALLOWED.has(permission)));
  });
  // Needed so enumerateDevices() returns real camera / microphone names.
  ses.setPermissionCheckHandler((wc, permission) => Boolean(win && wc === win.webContents && ALLOWED.has(permission)));
}

function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#05070b",
    title: "Kriya Sahay",
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });

  win.once("ready-to-show", () => win.show());
  win.setMenuBarVisibility(false);
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (event) => event.preventDefault());
  win.webContents.on("render-process-gone", (_event, details) => {
    console.error(`[ELECTRON] Renderer process exited: ${details.reason} (code ${details.exitCode})`);
  });
  win.webContents.on("did-fail-load", (_event, code, description) => {
    console.error(`[UI] Renderer failed to load (${code}): ${description}`);
  });
  win.on("closed", () => {
    win = null;
  });

  void win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

ipcMain.handle("neurogrip:control", async (_event, action, payload) => requestControl(action, payload));
ipcMain.handle("neurogrip:runtime-status", () => getRuntimeStatus());
ipcMain.handle("neurogrip:backend-restart", async () => {
  if (isShuttingDown) return { ok: false, error: "Application is shutting down." };
  backendRestarting = true;
  startupStateChecked = false;
  updateRuntimeStatus({ backend: "starting", cv: "starting", snapshot: null, message: "Preparing a clean Python backend restart…", pid: null });

  if (!bridge.isListening()) {
    try {
      await bridge.start();
      updateRuntimeStatus({ bridge: "listening", backend: "starting", cv: "starting", message: "Local bridge ready; restarting Python…" });
    } catch (error) {
      const message = `Could not restart the TCP bridge: ${error.message || error}`;
      backendRestarting = false;
      updateRuntimeStatus({ bridge: "error", backend: "error", cv: "error", snapshot: null, message });
      return { ok: false, error: message };
    }
  }

  if (backend.isRunning()) {
    const stopped = await backend.stop({ requestShutdown: () => requestControl("app.shutdown", {}) });
    if (!stopped.stopped || backend.isRunning()) {
      const message = "The existing Python process did not stop; refusing to start a second pipeline owner.";
      backendRestarting = false;
      updateRuntimeStatus({ backend: "error", cv: "error", snapshot: null, message });
      return { ok: false, error: message };
    }
  }

  const result = await backend.start();
  backendRestarting = false;
  if (!result.ok) {
    updateRuntimeStatus({ backend: "error", cv: "error", snapshot: null, message: result.error || "Python backend could not restart." });
  }
  return { ok: Boolean(result && result.ok), error: result && result.error ? result.error : undefined };
});

/**
 * Legacy renderer helpers remain thin wrappers over the one Python control channel.
 * None of these handlers opens a serial port or writes to hardware in Electron.
 */
ipcMain.handle("serial:ports", async () => {
  const res = await requestControl("serial.list_ports", {});
  if (!res.ok) return [];
  const ports = (res.data && res.data.ports) || [];
  return ports.map((port) => (typeof port === "string" ? port : port.device)).filter(Boolean);
});

ipcMain.handle("serial:connect", async (_event, port, baud) => {
  const res = await requestControl("serial.connect", {
    port: port || undefined,
    baud: baud ? Number(baud) : undefined,
    mode: "real",
  });
  const serial = (res.data && res.data.serial) || {};
  return {
    ok: Boolean(res.ok),
    port: serial.port,
    baud: serial.baud,
    error: res.ok ? undefined : res.error || "Port unavailable",
  };
});

ipcMain.handle("serial:disconnect", async () => {
  const res = await requestControl("serial.disconnect", {});
  const serial = (res.data && res.data.serial) || {};
  return { ok: Boolean(res.ok), port: serial.port, baud: serial.baud, error: res.error || undefined };
});

ipcMain.handle("serial:send", async (_event, command, source) => {
  const res = await requestControl("command.send", { command, source: source || "manual" });
  const data = res.data || {};
  return {
    ok: Boolean(res.ok),
    command: data.command || command,
    frame: data.frame || undefined,
    output: data.frame ? `${data.frame}\n` : undefined,
    error: res.ok ? undefined : res.error || "Serial send failed",
  };
});

ipcMain.handle("stt:status", () => stt.getStatus());
ipcMain.handle("stt:warmup", () => stt.warmup());
ipcMain.handle("stt:transcribe", async (_event, pcm) => {
  try {
    return await stt.transcribe(pcm);
  } catch (error) {
    return { text: "", ms: 0, error: String((error && error.message) || error) };
  }
});

async function startRuntime() {
  if (isShuttingDown) return;
  updateRuntimeStatus({ bridge: "starting", backend: "starting", cv: "starting", snapshot: null, message: "Starting local control bridge…", pid: null });
  try {
    await bridge.start();
    if (isShuttingDown) return;
    updateRuntimeStatus({ bridge: "listening", backend: "starting", cv: "starting", snapshot: null, message: "Bridge is ready; starting Python backend…" });
    const result = await backend.start();
    if (!result.ok && !isShuttingDown) {
      updateRuntimeStatus({ backend: "error", cv: "error", snapshot: null, message: result.error || "Python backend could not start.", pid: null });
    }
  } catch (error) {
    if (!isShuttingDown) {
      updateRuntimeStatus({ bridge: "error", backend: "error", message: `Could not start local bridge: ${error.message || error}` });
    }
  }
}

let shutdownPromise = null;
async function shutdownRuntime() {
  if (shutdownPromise) return shutdownPromise;
  isShuttingDown = true;
  shutdownPromise = (async () => {
    try {
      await backend.stop({
        requestShutdown: () => requestControl("app.shutdown", {}),
        requestTimeoutMs: 1500,
        gracefulTimeoutMs: 3500,
        terminateTimeoutMs: 1000,
      });
    } finally {
      bridge.stop();
    }
  })();
  return shutdownPromise;
}

app.whenReady().then(() => {
  if (!hasSingleInstanceLock) return;
  setupPermissions();
  stt.onStatus((status) => sendToWindow("stt:status", status));
  createWindow();
  void startRuntime();
  // Whisper's ensure() promise is shared; warming at most once keeps the first utterance fast.
  void stt.warmup();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", (event) => {
  if (allowQuit) return;
  event.preventDefault();
  if (shutdownPromise) return;
  void shutdownRuntime().finally(() => {
    allowQuit = true;
    app.quit();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("second-instance", () => {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});
