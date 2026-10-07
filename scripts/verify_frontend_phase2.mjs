/**
 * Frontend Phase 2 verification harness — the control channel.
 *
 * Phase 2 makes the UI actually control the running Python pipeline instead of faking it.
 * This harness bundles the *real* modules and drives them against a simulated Electron
 * preload bridge that records every control request, then checks:
 *
 *   1. `pipelineControl` talks to the pipeline over the persistent control channel
 *      (no `python -c`, no second serial owner) and tolerates a missing backend.
 *   2. Camera and serial UI state come from the pipeline's own `camera_info` / `serial_info`
 *      — never from hardcoded values or string guessing.
 *   3. Manual, override, voice and detector commands all reach hardware through the same
 *      single path (`command.send`), exactly once per action, with the source label.
 *   4. The camera switch/enable actions are real backend calls.
 *
 * Run:  node scripts/verify_frontend_phase2.mjs
 */
import { build } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = resolve(import.meta.dirname, "..");
const outDir = mkdtempSync(join(tmpdir(), "neurogrip-fe2-"));

let failures = 0;
let checks = 0;

function check(label, condition, detail = "") {
  checks += 1;
  if (condition) {
    console.log(`  \u2713 ${label}`);
  } else {
    failures += 1;
    console.log(`  \u2717 ${label}${detail ? ` — ${detail}` : ""}`);
  }
}

function section(title) {
  console.log(`\n${title}`);
}

const root = ROOT.replace(/\\/g, "/");
const entry = `
export * as control from "${root}/src/services/pipelineControl.ts";
export { subscribeDerivedPipelineState, getPipelineSnapshot, derivePipelineState } from "${root}/src/services/pipelineStore.ts";
export { createDesktopSerial } from "${root}/src/services/desktopSerial.ts";
export { createDesktopVision } from "${root}/src/services/desktopVision.ts";
export { createCommandEmitter } from "${root}/src/services/commandEmitter.ts";
export { OVERRIDE_MODEL_TAG } from "${root}/src/services/pipelineMapping.ts";
`;

const entryPath = join(outDir, "entry.ts");
const bundlePath = join(outDir, "bundle.mjs");
writeFileSync(entryPath, entry);

await build({
  entryPoints: [entryPath],
  outfile: bundlePath,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  logLevel: "warning",
  absWorkingDir: ROOT,
});

/* ------------------------------------------------------------------ */
/* Simulated Electron preload bridge                                   */
/* ------------------------------------------------------------------ */

const ipc = {
  controlCalls: [],       // [{ action, payload }]
  stateListeners: new Set(),
  frameListeners: new Set(),
  serialSendCalls: [],    // [{ command, source }]
  connected: true,
  /** Scripted replies keyed by action. */
  replies: {},
};

globalThis.window = {
  neurogrip: {
    isDesktop: true,
    onPipelineState(cb) {
      ipc.stateListeners.add(cb);
      return () => ipc.stateListeners.delete(cb);
    },
    onCameraFrame(cb) {
      ipc.frameListeners.add(cb);
      return () => ipc.frameListeners.delete(cb);
    },
    control(action, payload) {
      ipc.controlCalls.push({ action, payload });
      if (!ipc.connected) {
        return Promise.resolve({ ok: false, data: {}, error: "NeuroGrip pipeline is not connected (port 8765)." });
      }
      const scripted = ipc.replies[action];
      return Promise.resolve(
        scripted
          ? scripted
          : { ok: true, data: {}, error: null },
      );
    },
    serialSend(command, source) {
      ipc.serialSendCalls.push({ command, source });
      return Promise.resolve({ ok: true, command, frame: `NG1|${command}` });
    },
  },
};

const mod = await import(pathToFileURL(bundlePath).href);
const { control } = mod;

/** Push a raw VisualizationState exactly as the Python bridge would (Phase 2 payload). */
function pushState(overrides = {}) {
  const state = {
    command: "NO_COMMAND",
    confidence: 0,
    fps: 30,
    latency_ms: 12,
    hand_count: "0",
    pipeline_state: "NO_HAND",
    model_used: "N/A",
    raw_prediction: "N/A",
    routing_str: "N/A",
    transport_status: "DISCONNECTED (COM3)",
    last_tx: "NONE",
    is_stop_active: false,
    is_ambiguous: false,
    message: "",
    safety_status: "STOP DISARMED",
    handedness: "",
    landmarks_count: 0,
    camera_info: {
      index: 2,
      enabled: true,
      opened: true,
      state: "OPEN",
      backend: "opencv",
      resolution: "640 × 480",
      target_resolution: "1280 × 720",
      error: null,
    },
    serial_info: {
      mode: "real",
      port: "COM9",
      baud: 230400,
      connected: true,
      state: "CONNECTED",
      frames_sent: 7,
      last_tx: "NG1|CALL",
    },
    voice_info: { state: "listening", transcript: "call", command: "CALL", reply: "Calling.", engine: "whisper-base.en" },
    errors: [],
    manual_tx_count: 3,
    landmarks: Array.from({ length: 1 }, () => Array.from({ length: 21 }, (_, i) => [i / 21, i / 42, 0])),
    ...overrides,
  };
  ipc.stateListeners.forEach((l) => l(state));
}

/* ------------------------------------------------------------------ */
/* 1. Control channel contract                                         */
/* ------------------------------------------------------------------ */

section("1. Control channel (renderer → running pipeline)");

check(
  "every action is namespaced and unique",
  new Set(control.ALL_CONTROL_ACTIONS).size === control.ALL_CONTROL_ACTIONS.length &&
    control.ALL_CONTROL_ACTIONS.every((a) => a.includes(".")),
  control.ALL_CONTROL_ACTIONS.join(", "),
);
check("12 control actions are exposed", control.ALL_CONTROL_ACTIONS.length === 12, String(control.ALL_CONTROL_ACTIONS.length));

await control.selectCamera(1);
check("camera.select sends its payload", ipc.controlCalls.at(-1).action === "camera.select" && ipc.controlCalls.at(-1).payload.index === 1);

await control.setCameraEnabled(false);
check(
  "camera.set_enabled sends a boolean",
  ipc.controlCalls.at(-1).action === "camera.set_enabled" && ipc.controlCalls.at(-1).payload.enabled === false,
);

await control.connectSerial("COM9", 230400, "real");
check(
  "serial.connect carries port + baud + mode",
  ipc.controlCalls.at(-1).action === "serial.connect" &&
    ipc.controlCalls.at(-1).payload.port === "COM9" &&
    ipc.controlCalls.at(-1).payload.baud === 230400,
);

await control.disconnectSerial();
check("serial.disconnect is a real action", ipc.controlCalls.at(-1).action === "serial.disconnect");

await control.setStopArmed(true);
check("stop.set_armed sends the explicit armed flag", ipc.controlCalls.at(-1).payload.armed === true);

await control.setVoiceState("listening", "whisper");
check(
  "voice.set_state reports the UI voice state",
  ipc.controlCalls.at(-1).action === "voice.set_state" && ipc.controlCalls.at(-1).payload.state === "listening",
);

ipc.replies["voice.transcript"] = {
  ok: true,
  data: {
    transcript: "call",
    intent: "gesture",
    gesture: "CALL",
    reply: "Calling.",
    command: "CALL",
    frame: "NG1|CALL",
    sent: true,
  },
  error: null,
};
const voiceRes = await control.sendVoiceTranscript("call", "whisper");
const voiceData = control.voiceTranscriptFrom(voiceRes);
check(
  "voice.transcript returns the pipeline's own decision",
  ipc.controlCalls.at(-1).action === "voice.transcript" && voiceData?.intent === "gesture" && voiceData?.command === "CALL",
);

ipc.replies["pipeline.snapshot"] = {
  ok: true,
  data: { camera: { index: 1 }, serial: { connected: true }, voice: {}, running: true, initialized: true, stop_armed: false, state: "TRACKING", runtime: {} },
  error: null,
};
const snap = await control.getPipelineSnapshot();
check("pipeline.snapshot reports backend truth", snap.ok && snap.data.running === true);

// A disconnected pipeline must fail cleanly (no hang, no throw, no fake success).
ipc.connected = false;
const offline = await control.setCameraEnabled(true);
check("offline backend → ok:false with a real error", offline.ok === false && /not connected/i.test(offline.error ?? ""));
ipc.connected = true;

// Non-desktop (plain browser) usage must also degrade cleanly.
const savedBridge = globalThis.window.neurogrip;
delete globalThis.window.neurogrip;
const noBridge = await control.listCameras();
check("no desktop bridge → ok:false, no throw", noBridge.ok === false);
globalThis.window.neurogrip = savedBridge;

/* ------------------------------------------------------------------ */
/* 2. Real camera + serial state in the derived store                  */
/* ------------------------------------------------------------------ */

section("2. Pipeline state is the source of truth (no hardcoded values)");

let snapshot = mod.getPipelineSnapshot();
const unsubStore = mod.subscribeDerivedPipelineState(() => {
  snapshot = mod.getPipelineSnapshot();
});
pushState();

check("camera index comes from camera_info", snapshot.cameraIndex === 2, String(snapshot.cameraIndex));
check("camera resolution comes from camera_info", snapshot.cameraResolution === "640 × 480", snapshot.cameraResolution);
check("camera status reflects a live capture", snapshot.cameraStatus === "connected", snapshot.cameraStatus);
check("serial port comes from serial_info", snapshot.serialPort === "COM9", snapshot.serialPort);
check("baud rate comes from serial_info", snapshot.baudRate === 230400, String(snapshot.baudRate));
check("serial connection is a real boolean", snapshot.serialConnected === true);
check("frames sent is the pipeline's own counter", snapshot.framesSent === 7, String(snapshot.framesSent));
check("last TX is the real wire frame", snapshot.lastTx === "NG1|CALL", snapshot.lastTx);
check("safety status is reported by the backend", snapshot.safetyStatus === "STOP DISARMED", snapshot.safetyStatus);
check("landmarks reach the UI for the overlay", snapshot.landmarks.length === 21, String(snapshot.landmarks.length));
check("voice state is backend-owned", snapshot.voice.command === "CALL" && snapshot.voice.state === "listening");

pushState({
  camera_info: { index: 2, enabled: false, opened: false, state: "DISCONNECTED", backend: "opencv", resolution: "640 × 480", target_resolution: "1280 × 720", error: null },
});
check("camera off is reported as stopped", snapshot.cameraEnabled === false && snapshot.cameraStatus === "stopped", snapshot.cameraStatus);

pushState({
  camera_info: { index: 0, enabled: false, opened: false, state: "DISCONNECTED", backend: "opencv", resolution: "—", target_resolution: "1280 × 720", error: "camera 0 unavailable" },
  errors: ["camera 0 unavailable"],
});
check("camera errors surface with their real message", snapshot.cameraError === "camera 0 unavailable", String(snapshot.cameraError));
check("pipeline errors reach the UI", snapshot.errors[0] === "camera 0 unavailable");

pushState({
  transport_status: "MOCK",
  serial_info: { mode: "mock", port: "COM3", baud: 115200, connected: true, state: "MOCK", frames_sent: 0, last_tx: "NONE" },
});
check("a mock link is not mistaken for real hardware", snapshot.serialMode === "mock" && snapshot.serialConnected === true);

/* ------------------------------------------------------------------ */
/* 3. One command path, one serial owner                               */
/* ------------------------------------------------------------------ */

section("3. Every command source funnels into the pipeline");

const serial = mod.createDesktopSerial();
const serialStates = [];
const unsubSerial = serial.subscribe((s) => serialStates.push(s));

pushState({
  command: "CALL",
  hand_count: "1",
  confidence: 0.97,
  pipeline_state: "STABLE_CMD",
  serial_info: { mode: "real", port: "COM9", baud: 230400, connected: true, state: "CONNECTED", frames_sent: 9, last_tx: "NG1|OK" },
});
check("serial adapter picks up pipeline connection", serialStates.at(-1).connected === true);
check("serial adapter shows the pipeline's port", serialStates.at(-1).port === "COM9", serialStates.at(-1).port);
check("serial adapter shows the pipeline's frame count", serialStates.at(-1).packetsSent === 9, String(serialStates.at(-1).packetsSent));
check("serial adapter shows the real last frame", serialStates.at(-1).lastCommand === "NG1|OK", serialStates.at(-1).lastCommand);

// The emitter is the only caller of `serial.send`; one accepted emission → one wire command.
const emitter = mod.createCommandEmitter({ send: (g, src) => serial.send(g, src) });
const first = emitter.emit("CALL", "override");
await new Promise((r) => setTimeout(r, 10));
check("override emission is accepted", first.accepted === true);
check("override reaches the pipeline once", ipc.serialSendCalls.length === 1, JSON.stringify(ipc.serialSendCalls));
check("the source label is preserved", ipc.serialSendCalls[0].source === "override", String(ipc.serialSendCalls[0].source));

const identical = emitter.emit("CALL", "detector");
check("an identical detector frame is refused (unchanged)", identical.accepted === false && identical.reason === "unchanged", identical.reason);

emitter.emit("OK", "voice");
await new Promise((r) => setTimeout(r, 10));
check("voice emission reaches the same path", ipc.serialSendCalls.length === 2 && ipc.serialSendCalls[1].source === "voice");

const duplicated = emitter.emit("OK", "voice");
check("a repeat in the de-duplication window is refused", duplicated.accepted === false && duplicated.reason === "duplicate", duplicated.reason);

const rejected = emitter.emit("REST", "override");
check("non-transmittable names never reach hardware", rejected.accepted === false && rejected.reason === "rejected");

/* ------------------------------------------------------------------ */
/* 4. Detection path is tagged, not double-sent                        */
/* ------------------------------------------------------------------ */

section("4. Override preview never becomes a second command");

const vision = mod.createDesktopVision();
const frames = [];
const unsubVision = vision.subscribe((f) => frames.push(f));

vision.overrideGesture("CALL");
check("override preview frame is tagged", frames.at(-1).model === mod.OVERRIDE_MODEL_TAG, frames.at(-1).model);
check("override preview carries the gesture", frames.at(-1).gesture === "CALL");

pushState({ command: "OK", hand_count: "1", confidence: 0.9 });
check("real detection frames stay untagged", frames.at(-1).model !== mod.OVERRIDE_MODEL_TAG && frames.at(-1).gesture === "OK");
check("real landmarks are attached to the frame", frames.at(-1).landmarks.length === 21, String(frames.at(-1).landmarks.length));

/* ------------------------------------------------------------------ */
/* Teardown + result                                                   */
/* ------------------------------------------------------------------ */

unsubStore();
unsubSerial();
unsubVision();

console.log(`\n${checks - failures}/${checks} checks passed.`);
if (failures > 0) {
  console.log(`${failures} check(s) FAILED.`);
  process.exit(1);
}
