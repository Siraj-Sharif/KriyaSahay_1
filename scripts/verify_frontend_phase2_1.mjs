/**
 * Frontend Phase 2.1 verification harness — closed functional gaps.
 *
 * Phase 2.1 makes three frontend-side guarantees that the earlier harnesses did not cover.
 * This bundles the *real* modules and checks them against a simulated Electron preload
 * bridge, plus two source-level guards for wiring that only exists inside React:
 *
 *   1. STOP ARM/DISARM is a real backend control: `stop.set_armed` is requested from the UI,
 *      and the arming state displayed is the pipeline's own `is_armed`.
 *   2. The derived store carries `isArmed` / `isTxPermitted` / `finalCommand` straight from
 *      the backend payload (defaulting to *not armed* when an older bridge omits them).
 *   3. The renderer never duplicates a command the pipeline already transmitted:
 *      desktop voice commands are preview-only, and desktop detector frames are not
 *      re-emitted at all (the pipeline owns CV transmission).
 *
 * Run:  node scripts/verify_frontend_phase2_1.mjs
 */
import { build } from "esbuild";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = resolve(import.meta.dirname, "..");
const outDir = mkdtempSync(join(tmpdir(), "neurogrip-fe21-"));
const root = ROOT.replace(/\\/g, "/");

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

/* ------------------------------------------------------------------ */
/* Bundle the real modules                                             */
/* ------------------------------------------------------------------ */

const entry = `
export * as control from "${root}/src/services/pipelineControl.ts";
export { CONTROL_ACTIONS, ALL_CONTROL_ACTIONS } from "${root}/src/services/pipelineControl.ts";
export {
  derivePipelineState,
  DEFAULT_PIPELINE_STATE,
  subscribeDerivedPipelineState,
  getPipelineSnapshot,
  pipelineStoreSubscriberCount,
} from "${root}/src/services/pipelineStore.ts";
export { createDesktopSerial } from "${root}/src/services/desktopSerial.ts";
export { createCommandEmitter } from "${root}/src/services/commandEmitter.ts";
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
  controlCalls: [],
  serialSendCalls: [],
  connected: true,
  replies: {},
};

globalThis.window = {
  neurogrip: {
    isDesktop: true,
    onPipelineState() {
      return () => {};
    },
    onCameraFrame() {
      return () => {};
    },
    control(action, payload) {
      ipc.controlCalls.push({ action, payload });
      if (!ipc.connected) {
        return Promise.resolve({ ok: false, data: {}, error: "NeuroGrip pipeline is not connected (port 8765)." });
      }
      const scripted = ipc.replies[action];
      return Promise.resolve(scripted ? scripted : { ok: true, data: {}, error: null });
    },
    serialSend(command, source) {
      ipc.serialSendCalls.push({ command, source });
      return Promise.resolve({ ok: true, command, frame: `NG1|${command}` });
    },
  },
};

const mod = await import(pathToFileURL(bundlePath).href);
const { control, CONTROL_ACTIONS, derivePipelineState, DEFAULT_PIPELINE_STATE, createDesktopSerial } = mod;

/* ------------------------------------------------------------------ */
/* 1. STOP arming is a real backend control                            */
/* ------------------------------------------------------------------ */

section("1. STOP ARM/DISARM goes to the pipeline (no local-only flag)");

ipc.controlCalls.length = 0;
ipc.replies[CONTROL_ACTIONS.STOP_SET_ARMED] = { ok: true, data: { armed: true, serial: {} }, error: null };
const armedRes = await control.setStopArmed(true);

check("setStopArmed(true) sends exactly one control request", ipc.controlCalls.length === 1);
check(
  "…with action stop.set_armed",
  ipc.controlCalls[0]?.action === "stop.set_armed",
  ipc.controlCalls[0]?.action,
);
check(
  "…and the requested state as {armed:true}",
  ipc.controlCalls[0]?.payload?.armed === true,
  JSON.stringify(ipc.controlCalls[0]?.payload),
);
check("the response is read back from the pipeline", control.stopArmedFrom(armedRes) === true);

ipc.replies[CONTROL_ACTIONS.STOP_SET_ARMED] = { ok: true, data: { armed: false, serial: {} }, error: null };
const disarmedRes = await control.setStopArmed(false);
check("setStopArmed(false) requests {armed:false}", ipc.controlCalls[1]?.payload?.armed === false);
check("…and reports the backend's false state", control.stopArmedFrom(disarmedRes) === false);

ipc.replies[CONTROL_ACTIONS.STOP_SET_ARMED] = { ok: false, data: {}, error: "stop.set_armed requires a boolean 'armed'" };
const rejected = await control.setStopArmed(false);
check("a rejected arming request keeps ok:false (no fake success)", rejected.ok === false);
check("…and yields no arming state", control.stopArmedFrom(rejected) === null);
check("…with the pipeline's real error text", /boolean/.test(rejected.error || ""), rejected.error);

delete ipc.replies[CONTROL_ACTIONS.STOP_SET_ARMED];
const savedBridge = globalThis.window.neurogrip;
delete globalThis.window.neurogrip;
const noBridge = await control.setStopArmed(true);
check("no desktop bridge → ok:false, never a silent success", noBridge.ok === false && noBridge.data?.armed === undefined);
globalThis.window.neurogrip = savedBridge;

/* ------------------------------------------------------------------ */
/* 2. The store carries the backend's safety/transmission state        */
/* ------------------------------------------------------------------ */

section("2. Derived state carries the backend's arming + TX truth");

const armedState = derivePipelineState({
  command: "STOP",
  confidence: 0.9,
  fps: 28,
  latency_ms: 11,
  hand_count: "1",
  pipeline_state: "STABLE_CMD",
  model_used: "RULE_BASED",
  raw_prediction: "stop",
  routing_str: "RULE_BASED -> STOP",
  transport_status: "MOCK",
  last_tx: "NG1|STOP",
  is_stop_active: true,
  is_ambiguous: false,
  message: "",
  safety_status: "STOP ARMED",
  handedness: "RIGHT",
  landmarks_count: 1,
  is_armed: true,
  is_tx_permitted: true,
  final_command: "STOP",
});

check("isArmed comes from raw.is_armed", armedState.isArmed === true);
check("isTxPermitted comes from raw.is_tx_permitted", armedState.isTxPermitted === true);
check("finalCommand comes from raw.final_command", armedState.finalCommand === "STOP");

const legacyState = derivePipelineState({
  command: "NO_COMMAND",
  confidence: 0,
  fps: 0,
  latency_ms: 0,
  hand_count: "0",
  pipeline_state: "NO_HAND",
  model_used: "N/A",
  raw_prediction: "N/A",
  routing_str: "N/A",
  transport_status: "DISCONNECTED",
  last_tx: "NONE",
  is_stop_active: false,
  is_ambiguous: false,
  message: "",
  safety_status: "NORMAL",
  handedness: "",
  landmarks_count: 0,
});

check("an older payload without is_armed reports NOT armed", legacyState.isArmed === false);
check("…and NOT transmitting", legacyState.isTxPermitted === false);
check("DEFAULT_PIPELINE_STATE claims no arming", DEFAULT_PIPELINE_STATE.isArmed === false && DEFAULT_PIPELINE_STATE.isTxPermitted === false);

/* ------------------------------------------------------------------ */
/* 3. The renderer cannot duplicate a command the pipeline sent        */
/* ------------------------------------------------------------------ */

section("3. No renderer-side duplicate of a pipeline transmission");

const serial = createDesktopSerial();
ipc.serialSendCalls.length = 0;
await serial.send("CALL", "override");
check("manual override still transmits exactly one frame", ipc.serialSendCalls.length === 1);
check("…tagged with its source", ipc.serialSendCalls[0]?.source === "override");

// The voice path in desktop mode must preview only: the pipeline already wrote the frame for
// `voice.transcript`, so an extra adapter send here would be a second identical transmission.
const emitter = mod.createCommandEmitter({
  send: (gesture, source) => serial.send(gesture, source),
});
const before = ipc.serialSendCalls.length;
const accepted = emitter.emit("CALL", "voice");
check("the emitter itself still supports a voice transmission (browser path)", accepted.accepted === true);
check("…which produced one send", ipc.serialSendCalls.length === before + 1);

const contextSource = readFileSync(join(ROOT, "src/services/SystemContext.tsx"), "utf8");
check(
  "desktop voice commands are preview-only (pipeline already transmitted)",
  /backendControlled\s*\?\s*previewOverride/.test(contextSource),
);
check(
  "desktop detector frames are never re-emitted (pipeline owns CV transmission)",
  /if\s*\(!backendControlled\)\s*emitFromFrame\(f\)/.test(contextSource),
);

const hookSource = readFileSync(join(ROOT, "src/hooks/useGestureCommand.ts"), "utf8");
const previewBlock = hookSource.match(/const previewOverride = useCallback\(([\s\S]*?)\n  \);/)?.[1] ?? "";
check(
  "previewOverride exists, previews visually and never calls emit()",
  previewBlock.length > 0 && /overrideGesture/.test(previewBlock) && !/emit\(/.test(previewBlock),
  `block length ${previewBlock.length}`,
);
check(
  "setOverride still previews and then transmits exactly once",
  /const setOverride = useCallback\(\s*\(gesture: Gesture \| null\) => \{\s*previewOverride\(gesture\);\s*if \(gesture\) emit\(gesture, "override"\);/.test(
    hookSource,
  ),
);

/* ------------------------------------------------------------------ */
/* 4. The ARM/DISARM control is bound to backend state                 */
/* ------------------------------------------------------------------ */

section("4. The safety control reflects the pipeline, not React state");

const hardwareSource = readFileSync(join(ROOT, "src/pages/Hardware.tsx"), "utf8");
check("Hardware.tsx imports the control helper", /setStopArmed\s+as\s+requestStopArmed/.test(hardwareSource));
check("the control label follows the backend's isArmed", /pipelineState\.isArmed\s*\?/.test(hardwareSource));
check("the request asks for the opposite of the backend state", /requestStopArmed\(!pipelineState\.isArmed\)/.test(hardwareSource));
check("a rejected request surfaces the real error", /setSafetyNote\(res\.error/.test(hardwareSource));
check(
  "no local isArmed useState shadows the backend",
  !/useState[^;]*[Aa]rmed/.test(hardwareSource),
);
check(
  "pipeline errors are rendered from the state payload",
  /pipelineState\.errors\.length/.test(hardwareSource) && /pipelineState\.errors\[pipelineState\.errors\.length - 1\]/.test(hardwareSource),
);

/* ------------------------------------------------------------------ */
/* Report                                                              */
/* ------------------------------------------------------------------ */

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) process.exitCode = 1;
