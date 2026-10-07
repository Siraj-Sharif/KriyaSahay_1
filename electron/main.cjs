const { app, BrowserWindow, ipcMain, session } = require("electron");
const path = require("path");
const stt = require("./stt.cjs");
const { createPipelineBridge } = require("./bridge.cjs");

if (!app.requestSingleInstanceLock()) {
  app.quit();
}
app.setAppUserModelId("com.neurogrip.app");

let win = null;
const BRIDGE_PORT = 8765;

/** Single owner of the pipeline connection: one socket, one pipeline, one serial port. */
const bridge = createPipelineBridge({
  port: BRIDGE_PORT,
  onState: (state) => sendToWindow("neurogrip:state", state),
  onFrame: (msg) => sendToWindow("neurogrip:frame", msg),
  onStatus: (connected) => sendToWindow("neurogrip:bridge-status", { connected }),
  onControlResult: (msg) => sendToWindow("neurogrip:control-result", msg),
});

function sendToWindow(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

/** Send one control request to the running pipeline and resolve with its result. */
function requestControl(action, payload) {
  return bridge.request(action, payload);
}

/** Permissions the UI legitimately needs: camera, microphone, fullscreen. */
const ALLOWED = new Set(["media", "audioCapture", "videoCapture", "mediaKeySystem", "fullscreen", "clipboard-sanitized-write"]);

function setupPermissions() {
  const ses = session.defaultSession;
  ses.setPermissionRequestHandler((wc, permission, callback) => {
    callback(Boolean(win && wc === win.webContents && ALLOWED.has(permission)));
  });
  // needed so enumerateDevices() returns real camera / microphone names
  ses.setPermissionCheckHandler((wc, permission) => Boolean(win && wc === win.webContents && ALLOWED.has(permission)));
}

function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#030407",
    title: "NeuroGrip",
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false, // keep vision / 3D loops running when unfocused
    },
  });

  win.once("ready-to-show", () => win.show());
  win.setMenuBarVisibility(false);
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e) => e.preventDefault());
  win.on("closed", () => {
    win = null;
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

/**
 * Phase 2: the renderer reaches the pipeline through these channels only.
 *
 * The old implementations spawned a throwaway `python -c` process that opened the SAME
 * COM port the pipeline already owned — a second serial writer that could not see the
 * pipeline's state. They are now thin wrappers over the persistent control channel.
 */

ipcMain.handle("neurogrip:control", async (_e, action, payload) => requestControl(action, payload));

ipcMain.handle("serial:ports", async () => {
  const res = await requestControl("serial.list_ports", {});
  if (!res.ok) return [];
  const ports = (res.data && res.data.ports) || [];
  // keep the historical `string[]` shape for existing callers
  return ports.map((p) => (typeof p === "string" ? p : p.device)).filter(Boolean);
});

ipcMain.handle("serial:connect", async (_e, port, baud) => {
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

ipcMain.handle("serial:send", async (_e, cmd, source) => {
  const res = await requestControl("command.send", {
    command: cmd,
    source: source || "manual",
  });
  const data = res.data || {};
  return {
    ok: Boolean(res.ok),
    command: data.command || cmd,
    frame: data.frame || undefined,
    output: data.frame ? `${data.frame}\n` : undefined,
    error: res.ok ? undefined : res.error || "Serial send failed",
  };
});

ipcMain.handle("stt:status", () => stt.getStatus());
ipcMain.handle("stt:warmup", () => stt.warmup());
ipcMain.handle("stt:transcribe", async (_e, pcm) => {
  try {
    return await stt.transcribe(pcm);
  } catch (err) {
    return { text: "", ms: 0, error: String((err && err.message) || err) };
  }
});

app.whenReady().then(() => {
  setupPermissions();
  bridge.start();
  stt.onStatus((s) => {
    if (win && !win.isDestroyed()) win.webContents.send("stt:status", s);
  });
  createWindow();
  // start loading the voice model right away so it's ready by the time you tap the mic
  void stt.warmup();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  bridge.stop();
  if (process.platform !== "darwin") app.quit();
});

app.on("second-instance", () => {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});
