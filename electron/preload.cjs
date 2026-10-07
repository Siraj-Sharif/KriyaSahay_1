const { contextBridge, ipcRenderer } = require("electron");

/**
 * The ONLY surface the web UI can use to reach the desktop side.
 * (contextIsolation is on and node integration is off.)
 */
contextBridge.exposeInMainWorld("neurogrip", {
  isDesktop: true,
  platform: process.platform,
  stt: {
    warmup: () => ipcRenderer.invoke("stt:warmup"),
    getStatus: () => ipcRenderer.invoke("stt:status"),
    transcribe: (pcm) => ipcRenderer.invoke("stt:transcribe", pcm),
    onStatus: (cb) => {
      const handler = (_e, s) => cb(s);
      ipcRenderer.on("stt:status", handler);
      return () => ipcRenderer.removeListener("stt:status", handler);
    },
  },
  /** Current truthful Electron / Python readiness snapshot. */
  getRuntimeStatus: () => ipcRenderer.invoke("neurogrip:runtime-status"),
  restartBackend: () => ipcRenderer.invoke("neurogrip:backend-restart"),
  onRuntimeStatus: (cb) => {
    const handler = (_e, status) => cb(status);
    ipcRenderer.on("neurogrip:runtime-status", handler);
    return () => ipcRenderer.removeListener("neurogrip:runtime-status", handler);
  },
  // Serial compatibility handlers (all delegate to the pipeline control channel).
  serialPorts: () => ipcRenderer.invoke("serial:ports"),
  serialConnect: (port, baud) => ipcRenderer.invoke("serial:connect", port, baud),
  serialDisconnect: () => ipcRenderer.invoke("serial:disconnect"),
  serialSend: (cmd, source) => ipcRenderer.invoke("serial:send", cmd, source),
  /**
   * Phase 2: send a control request to the running Python pipeline.
   *
   * `action` is one of the actions in `pc/src/neurogrip/bridge/control.py` (kept in sync
   * with `src/services/pipelineControl.ts`). Always resolves to
   * `{ ok, data, error }` — it never throws and never hangs.
   */
  control: (action, payload) => ipcRenderer.invoke("neurogrip:control", action, payload),
  /** Control results the renderer did not correlate (late replies / backend pushes). */
  onControlResult: (cb) => {
    const handler = (_e, result) => cb(result);
    ipcRenderer.on("neurogrip:control-result", handler);
    return () => ipcRenderer.removeListener("neurogrip:control-result", handler);
  },
  // Bridge connection status (Python connected / disconnected)
  onBridgeStatus: (cb) => {
    const handler = (_e, status) => cb(status);
    ipcRenderer.on("neurogrip:bridge-status", handler);
    return () => ipcRenderer.removeListener("neurogrip:bridge-status", handler);
  },
  // Real-time pipeline state from Python bridge
  onPipelineState: (cb) => {
    const handler = (_e, state) => cb(state);
    ipcRenderer.on("neurogrip:state", handler);
    return () => ipcRenderer.removeListener("neurogrip:state", handler);
  },
  // Real camera frame from existing Python camera
  onCameraFrame: (cb) => {
    const handler = (_e, frameMsg) => cb(frameMsg);
    ipcRenderer.on("neurogrip:frame", handler);
    return () => ipcRenderer.removeListener("neurogrip:frame", handler);
  },
});
