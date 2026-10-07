/**
 * Verification harness for `electron/bridge.cjs` (the Electron side of the control channel).
 *
 * The Python side of the wire contract is covered by `pc/tests/test_control_channel.py`.
 * This harness covers the half that runs inside Electron, with a scripted fake pipeline:
 *
 *   1. `onStatus(true)` fires when the pipeline is *accepted* (the old code listened for a
 *      "connect" event that never fires on an accepted socket, so the UI's backend
 *      indicator could never turn on).
 *   2. Control requests are correctly correlated by id, even when replies arrive out of
 *      order, and unknown ids are surfaced instead of dropped.
 *   3. Every failure mode resolves the caller — no pipeline, no answer (timeout), socket
 *      error, mid-request disconnect — so the renderer can never hang.
 *   4. Malformed input from the pipeline cannot break the channel.
 *
 * Run:  node scripts/verify_bridge_control.mjs
 */
import { createRequire } from "node:module";
import net from "node:net";
import { resolve } from "node:path";

const require = createRequire(import.meta.url);
const { createPipelineBridge } = require(resolve(import.meta.dirname, "..", "electron", "bridge.cjs"));

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

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** Minimal fake pipeline: connects, records control requests, answers on demand. */
function createFakePipeline(port) {
  const socket = net.connect({ host: "127.0.0.1", port });
  const received = [];
  let buffer = "";
  const waiting = [];

  const ready = new Promise((res, rej) => {
    socket.once("connect", res);
    socket.once("error", rej);
  });

  socket.on("data", (data) => {
    buffer += data.toString();
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) received.push(JSON.parse(line));
    }
    while (waiting.length) waiting.shift()();
  });

  const push = (obj) => socket.write(JSON.stringify(obj) + "\n");

  return {
    socket,
    received,
    ready,
    push,
    /** Wait for the next request with this action (robust against ordering changes). */
    async waitForAction(action, timeout = 2000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        const found = received.find((r) => r.action === action);
        if (found) return found;
        await wait(5);
      }
      throw new Error(`pipeline never received control request '${action}'`);
    },
    async waitForRequest(index = 0, timeout = 2000) {
      const deadline = Date.now() + timeout;
      while (Date.now() < deadline) {
        if (received.length > index) return received[index];
        await wait(5);
      }
      throw new Error(`pipeline never received control request #${index}`);
    },
    reply(id, ok, data, error = null) {
      push({ type: "control_result", id, ok, data: data || {}, error });
    },
    close() {
      socket.destroy();
    },
  };
}

const onStatusCalls = [];
const onStateCalls = [];
const onFrameCalls = [];
const unsolicited = [];

const bridge = createPipelineBridge({
  port: 0,
  controlTimeoutMs: 400,
  log: () => {},
  logError: () => {},
  onStatus: (connected) => onStatusCalls.push(connected),
  onState: (s) => onStateCalls.push(s),
  onFrame: (f) => onFrameCalls.push(f),
  onControlResult: (r) => unsolicited.push(r),
});
bridge.start();
await wait(30);
const port = bridge.boundPort();

/* ------------------------------------------------------------------ */

section("1. No pipeline connected");

check("isConnected() is false before any pipeline connects", bridge.isConnected() === false);
const offline = await bridge.request("camera.list", {});
check("a request without a pipeline resolves (never hangs)", offline.ok === false && /not connected/i.test(offline.error), offline.error);

/* ------------------------------------------------------------------ */

section("2. Pipeline connects");

const pipeline = createFakePipeline(port);
await pipeline.ready;
await wait(30);

check("onStatus(true) fires on accept, not on a 'connect' event", onStatusCalls.at(-1) === true, JSON.stringify(onStatusCalls));
check("isConnected() is true after accept", bridge.isConnected() === true);

pipeline.push({ type: "state", pipeline_state: "TRACKING", fps: 30, camera_info: { index: 0 } });
pipeline.push({ type: "frame", data: "AAAA", ts: 1 });
await wait(30);
check("state lines reach onState", onStateCalls.length === 1 && onStateCalls[0].pipeline_state === "TRACKING");
check("frame lines reach onFrame", onFrameCalls.length === 1 && onFrameCalls[0].data === "AAAA");

/* ------------------------------------------------------------------ */

section("3. Request/response correlation");

const pendingA = bridge.request("camera.list", { probe: true });
const requestA = await pipeline.waitForRequest(0);
check("requests are JSON lines with type/id/action/payload", requestA.type === "control" && typeof requestA.id === "string" && requestA.action === "camera.list" && requestA.payload.probe === true, JSON.stringify(requestA));

const pendingB = bridge.request("serial.list_ports", {});
const requestB = await pipeline.waitForRequest(1);
check("each request gets a unique id", requestA.id !== requestB.id, `${requestA.id} vs ${requestB.id}`);

// Answer the second request first: correlation must be by id, not by order.
pipeline.reply(requestB.id, true, { ports: [{ device: "COM9" }] });
pipeline.reply(requestA.id, true, { cameras: [{ index: 0 }] });
const [resA, resB] = await Promise.all([pendingA, pendingB]);
check("out-of-order replies resolve the right callers", resA.data.cameras?.[0]?.index === 0 && resB.data.ports?.[0]?.device === "COM9", JSON.stringify([resA.data, resB.data]));

pipeline.reply("ui-does-not-exist", true, { late: true });
await wait(20);
check("an uncorrelated result is surfaced, not dropped", unsolicited.length === 1 && unsolicited[0].data.late === true);

/* ------------------------------------------------------------------ */

section("4. Failure modes always resolve");

const timedOut = await bridge.request("stop.set_armed", { armed: true });
check("an unanswered request times out with a real error", timedOut.ok === false && /did not answer/i.test(timedOut.error), timedOut.error);

pipeline.socket.write("this is not json\n");
await wait(20);
const stillWorking = bridge.request("pipeline.snapshot", {});
const snapshotReq = await pipeline.waitForAction("pipeline.snapshot");
pipeline.reply(snapshotReq.id, true, { running: true });
check("malformed input does not break the channel", (await stillWorking).data.running === true);

const inFlight = bridge.request("serial.connect", { port: "COM9" });
await pipeline.waitForAction("serial.connect");
pipeline.close();
const afterClose = await inFlight;
check("a pending request fails when the pipeline disconnects", afterClose.ok === false && /disconnected/i.test(afterClose.error), afterClose.error);
check("onStatus(false) fires on disconnect", onStatusCalls.at(-1) === false, JSON.stringify(onStatusCalls));

const afterDisconnect = await bridge.request("camera.list", {});
check("requests after disconnect resolve immediately", afterDisconnect.ok === false);

bridge.stop();

console.log(`\n${checks - failures}/${checks} checks passed.`);
if (failures > 0) {
  console.log(`${failures} check(s) FAILED.`);
  process.exit(1);
}
