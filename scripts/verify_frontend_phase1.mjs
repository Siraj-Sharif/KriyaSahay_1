/**
 * Frontend Phase 1 verification harness.
 *
 * Bundles the *real* frontend modules with esbuild and drives them against a simulated
 * Electron preload bridge. Checks the four claims Phase 1 makes about the frontend:
 *
 *   1. Exactly ONE `onPipelineState` IPC subscription exists, regardless of how many
 *      consumers (previously `usePipelineState` ×8 + `desktopVision` + `desktopSerial`).
 *   2. CALL and OK survive the raw → UI mapping unchanged (and the raw command reaches the
 *      adapter as the taxonomy name, so the serial bridge receives "CALL" / "OK").
 *   3. `desktopVision.overrideGesture` no longer double-sends: the synthetic frame is
 *      tagged and the emitter refuses the echo.
 *   4. The re-entrancy / duplicate / detector-change-only rules behave as documented.
 *
 * Run:  node scripts/verify_frontend_phase1.mjs
 */
import { build } from "esbuild";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = resolve(import.meta.dirname, "..");
const outDir = mkdtempSync(join(tmpdir(), "neurogrip-fe-"));

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
export { mapCommandToGesture, isTransmittableGesture, OVERRIDE_MODEL_TAG, extractSerialPort, isTransportConnected, mapHandedness, parseHandCount } from "${ROOT.replace(/\\/g, "/")}/src/services/pipelineMapping.ts";
export { subscribePipelineState, pipelineSourceSubscriberCount } from "${ROOT.replace(/\\/g, "/")}/src/services/pipelineSource.ts";
export { subscribeDerivedPipelineState, getPipelineSnapshot, DEFAULT_PIPELINE_STATE, derivePipelineState, pipelineStoreSubscriberCount } from "${ROOT.replace(/\\/g, "/")}/src/services/pipelineStore.ts";
export { createCommandEmitter, DEFAULT_DEDUPE_MS } from "${ROOT.replace(/\\/g, "/")}/src/services/commandEmitter.ts";
export { createDesktopVision } from "${ROOT.replace(/\\/g, "/")}/src/services/desktopVision.ts";
export { createDesktopSerial } from "${ROOT.replace(/\\/g, "/")}/src/services/desktopSerial.ts";
`;

const entryPath = join(outDir, "entry.ts");
const bundlePath = join(outDir, "bundle.mjs");
await import("node:fs").then((fs) => fs.writeFileSync(entryPath, entry));

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
  onPipelineStateCalls: 0,
  unsubscribes: 0,
  listeners: new Set(),
  sent: [],
};

globalThis.window = {
  neurogrip: {
    isDesktop: true,
    onPipelineState(cb) {
      ipc.onPipelineStateCalls += 1;
      ipc.listeners.add(cb);
      return () => {
        ipc.unsubscribes += 1;
        ipc.listeners.delete(cb);
      };
    },
    serialSend(command) {
      ipc.sent.push(command);
      return Promise.resolve({ ok: true, command });
    },
  },
};

const mod = await import(pathToFileURL(bundlePath).href);

/** Push a raw VisualizationState exactly as the Python bridge would. */
function pushFrame(overrides = {}) {
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
    transport_status: "MOCK",
    last_tx: "NONE",
    is_stop_active: false,
    is_ambiguous: false,
    message: "",
    safety_status: "ARMED",
    handedness: "",
    landmarks_count: 0,
    ...overrides,
  };
  ipc.listeners.forEach((l) => l(state));
}

/* ------------------------------------------------------------------ */
/* 1. Mapping                                                          */
/* ------------------------------------------------------------------ */

section("1. Raw → UI mapping (CALL / OK must pass through)");
check("CALL maps to CALL", mod.mapCommandToGesture("CALL") === "CALL");
check("OK maps to OK", mod.mapCommandToGesture("OK") === "OK");
check("lowercase ' ok ' maps to OK", mod.mapCommandToGesture(" ok ") === "OK");
check("CALL is transmittable", mod.isTransmittableGesture("CALL") === true);
check("OK is transmittable", mod.isTransmittableGesture("OK") === true);
for (const nonGesture of ["NO_COMMAND", "NO_HAND", "AMBIGUOUS", "REST", "NONE", ""]) {
  check(`'${nonGesture}' is not transmittable`, mod.isTransmittableGesture(nonGesture) === false);
}
check("NO_COMMAND falls back to the safe STOP pose", mod.mapCommandToGesture("NO_COMMAND") === "STOP");
check("port parsed from 'CONNECTED (COM3)'", mod.extractSerialPort("CONNECTED (COM3)") === "COM3");
check("'CONNECTED' is connected", mod.isTransportConnected("CONNECTED") === true);
check("'MOCK' is not connected", mod.isTransportConnected("MOCK") === false);

/* ------------------------------------------------------------------ */
/* 2. Single IPC subscription                                          */
/* ------------------------------------------------------------------ */

section("2. One IPC subscription for the whole app");
check("no IPC subscription before any consumer exists", ipc.onPipelineStateCalls === 0, `saw ${ipc.onPipelineStateCalls}`);

const vision = mod.createDesktopVision();
const serial = mod.createDesktopSerial();
const frames = [];
const unsubFrames = vision.subscribe((f) => frames.push(f));
const unsubStoreA = mod.subscribeDerivedPipelineState(() => {});
const unsubStoreB = mod.subscribeDerivedPipelineState(() => {});
const unsubSerial = serial.subscribe(() => {});

check("exactly one onPipelineState registration", ipc.onPipelineStateCalls === 1, `saw ${ipc.onPipelineStateCalls}`);
check(
  "vision + serial + store share the single registration",
  mod.pipelineSourceSubscriberCount() === 3,
  `raw consumers = ${mod.pipelineSourceSubscriberCount()}`,
);
// Mounting many components must not add IPC consumers: the store is the only one.
const extraStoreUnsubs = Array.from({ length: 8 }, () => mod.subscribeDerivedPipelineState(() => {}));
check(
  "8 extra usePipelineState consumers add zero IPC consumers",
  mod.pipelineSourceSubscriberCount() === 3 && ipc.onPipelineStateCalls === 1,
  `raw consumers = ${mod.pipelineSourceSubscriberCount()}, ipc = ${ipc.onPipelineStateCalls}`,
);
extraStoreUnsubs.forEach((u) => u());

pushFrame({ command: "NO_COMMAND", transport_status: "MOCK", last_tx: "NONE" });
check("vision received the frame", frames.length >= 1);

/* ------------------------------------------------------------------ */
/* 3. CALL / OK reach the adapter                                      */
/* ------------------------------------------------------------------ */

section("3. CALL / OK reach the serial adapter unchanged");
pushFrame({ command: "CALL", confidence: 0.97, fps: 30, hand_count: "1", handedness: "Right", pipeline_state: "STABLE_CMD" });
const callFrame = frames.at(-1);
check("frame gesture is CALL", callFrame?.gesture === "CALL", `got ${callFrame?.gesture}`);
check("frame handDetected", callFrame?.handDetected === true);

pushFrame({ command: "OK", confidence: 0.95, fps: 30, hand_count: "1", handedness: "Right", pipeline_state: "STABLE_CMD" });
const okFrame = frames.at(-1);
check("frame gesture is OK", okFrame?.gesture === "OK", `got ${okFrame?.gesture}`);

check(
  "store snapshot exposes the raw command for CALL",
  (pushFrame({ command: "CALL", confidence: 0.9, hand_count: "1", pipeline_state: "STABLE_CMD" }),
  mod.getPipelineSnapshot().command === "CALL"),
);
check("store snapshot gesture is CALL", mod.getPipelineSnapshot().gesture === "CALL");

/* ------------------------------------------------------------------ */
/* 4. The double-send regression                                       */
/* ------------------------------------------------------------------ */

section("4. desktopVision.overrideGesture must not double-send");

const sent = [];
const emitter = mod.createCommandEmitter({
  send: (gesture, source) => sent.push({ gesture, source }),
});
const overrideFrames = [];
const unsubOverrideFrames = vision.subscribe((f) => overrideFrames.push(f));

// Simulate SystemContext's subscriber: feed every frame through the detector path.
const unsubFromFrame = vision.subscribe((f) => {
  // Mirrors SystemContext's subscriber, which now routes through emitFromFrame().
  if (f.model === mod.OVERRIDE_MODEL_TAG) return;
  if (!f.handDetected || f.confidence <= 0.7) return;
  emitter.emit(f.gesture, "detector");
});

// The exact Phase 1 fix: one override = one synthetic tagged frame + one emission.
vision.overrideGesture("CALL");
const overrideFrame = overrideFrames.at(-1);
check("override emits a tagged frame", overrideFrame?.model === mod.OVERRIDE_MODEL_TAG, `model = ${overrideFrame?.model}`);
check("override frame carries CALL", overrideFrame?.gesture === "CALL");
check("override frame is not treated as a detection", overrideFrame?.model === mod.OVERRIDE_MODEL_TAG);

// Old behaviour: setOverrideGesture sent directly AND the synthetic frame re-entered the
// subscriber. The emitter must now accept the explicit emission and refuse the echo.
const first = emitter.emit("CALL", "override");
const echo = emitter.emit("CALL", "detector");
check("explicit override is accepted", first.accepted === true, `reason=${first.reason}`);
check("synthetic-frame echo is refused exactly once", sent.filter((s) => s.gesture === "CALL").length === 1, `sent ${sent.filter((s) => s.gesture === "CALL").length}`);
check("sent exactly one command in total", sent.length === 1, `sent ${sent.length}`);

section("4a-2. Real pipeline frames are unaffected by an active override");
vision.overrideGesture("CALL");
const beforeReal = overrideFrames.length;
pushFrame({ command: "OK", confidence: 0.95, fps: 30, hand_count: "1", handedness: "Right", pipeline_state: "STABLE_CMD" });
const realDuringOverride = overrideFrames.at(-1);
check("live frame still reaches the UI during an override", overrideFrames.length > beforeReal);
check("live frame stays untagged (not an override)", realDuringOverride?.model !== mod.OVERRIDE_MODEL_TAG, `model = ${realDuringOverride?.model}`);
check("live frame reports the real gesture OK", realDuringOverride?.gesture === "OK", `gesture = ${realDuringOverride?.gesture}`);
vision.overrideGesture(null);

section("4b. Same override with an untagged (legacy) echo — re-entrancy guard");
const sent2 = [];
let reentered = null;
const emitter2 = mod.createCommandEmitter({
  send: (gesture) => {
    sent2.push(gesture);
    // Adapter fans a frame back out synchronously while we are still inside emit().
    reentered = emitter2.emit(gesture, "detector");
  },
});
const outer = emitter2.emit("OK", "override");
check("outer emission accepted", outer.accepted === true);
check("re-entrant echo refused as 'reentrant'", reentered?.reason === "reentrant", `reason=${reentered?.reason}`);
check("only one OK reached the wire", sent2.length === 1, `sent ${sent2.length}`);

section("4c. Duplicate + detector rules");
let clock = 1000;
const sent3 = [];
const emitter3 = mod.createCommandEmitter({ send: (g) => sent3.push(g), now: () => clock, dedupeMs: 300 });
check("first detector emission accepted", emitter3.emit("CALL", "detector").accepted === true);
check("same detector gesture refused (unchanged)", emitter3.emit("CALL", "detector").reason === "unchanged");
check("changing gesture accepted", emitter3.emit("OK", "detector").accepted === true);
check("repeat inside dedupe window refused", emitter3.emit("OK", "override").reason === "duplicate");
clock += 500;
check("repeat after dedupe window accepted", emitter3.emit("OK", "override").accepted === true);
check("only 3 frames reached the wire", sent3.length === 3, `sent ${sent3.length}`);

section("4c-2. reset() re-arms transmission (camera off → on)");
emitter3.reset();
check("same gesture transmits again after reset", emitter3.emit("OK", "detector").accepted === true);
check("reset did not corrupt the counters", emitter3.stats().dispatched === 4, JSON.stringify(emitter3.stats()));

section("4d. Invalid commands are rejected before touching the adapter");
const sent4 = [];
const emitter4 = mod.createCommandEmitter({ send: (g) => sent4.push(g) });
for (const bad of ["NO_COMMAND", "REST", "UNKNOWN", "", "GRABBING"]) {
  check(`'${bad}' rejected`, emitter4.emit(bad).accepted === false);
}
check("nothing was transmitted", sent4.length === 0, `sent ${sent4.length}`);

section("4e. Stats expose the suppression reasons");
const stats = emitter4.stats();
check("rejected counter incremented 5 times", stats.rejected === 5, JSON.stringify(stats));
check("dispatched is 0", stats.dispatched === 0, JSON.stringify(stats));

/* ------------------------------------------------------------------ */
/* 5. Cleanup                                                          */
/* ------------------------------------------------------------------ */

section("5. Teardown releases the IPC subscription");
unsubFrames();
unsubStoreA();
unsubStoreB();
unsubSerial();
unsubOverrideFrames();
unsubFromFrame();
check("all raw subscribers released", mod.pipelineSourceSubscriberCount() === 0, `${mod.pipelineSourceSubscriberCount()} left`);
check("IPC channel closed", ipc.unsubscribes >= 1, `unsubscribes = ${ipc.unsubscribes}`);
check("store default state restored", mod.getPipelineSnapshot() === mod.DEFAULT_PIPELINE_STATE);

/* ------------------------------------------------------------------ */

rmSync(outDir, { recursive: true, force: true });
console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log("All frontend Phase 1 checks passed.");
