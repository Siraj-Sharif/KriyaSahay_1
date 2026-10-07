/**
 * On-device speech-to-text (Whisper) running in Electron's main process.
 *
 * - Model is loaded lazily, once, and kept in memory.
 * - If a bundled copy exists (electron/models or <resources>/models) it is used with no network at all.
 * - Otherwise it is downloaded ONCE into the user's app-data folder and reused offline afterwards.
 * - The renderer sends 16 kHz mono PCM; this returns plain text.
 */
const path = require("path");
const fs = require("fs");
const { app } = require("electron");

const MODEL_ID = process.env.NEUROGRIP_STT_MODEL || "Xenova/whisper-base.en";
const IS_ENGLISH_ONLY = /\.en$/.test(MODEL_ID);

let status = { phase: "idle", progress: 0, message: "Voice engine not started" };
const listeners = new Set();
let asrPromise = null;
let queue = Promise.resolve();

function setStatus(patch) {
  status = { ...status, ...patch };
  for (const l of listeners) {
    try {
      l(status);
    } catch {
      /* window closed */
    }
  }
}

function bundledModelsDir() {
  const candidates = [app.isPackaged ? path.join(process.resourcesPath, "models") : path.join(__dirname, "models")];
  return candidates.find((d) => fs.existsSync(path.join(d, MODEL_ID)));
}

async function load() {
  setStatus({ phase: "loading", progress: 0, message: "Starting voice engine…" });
  const { pipeline, env } = await import("@huggingface/transformers");

  const bundled = bundledModelsDir();
  env.cacheDir = path.join(app.getPath("userData"), "models");
  if (bundled) {
    env.localModelPath = bundled;
    env.allowLocalModels = true;
    env.allowRemoteModels = false; // fully offline
  } else {
    env.allowRemoteModels = true; // first run only; cached afterwards
  }

  const files = new Map();
  const t0 = Date.now();
  let lastPct = -1;
  const asr = await pipeline("automatic-speech-recognition", MODEL_ID, {
    dtype: "q8",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files.set(p.file, { loaded: p.loaded, total: p.total });
        let loaded = 0;
        let total = 0;
        for (const f of files.values()) {
          loaded += f.loaded;
          total += f.total;
        }
        const pct = total ? Math.round((loaded / total) * 100) : 0;
        if (pct === lastPct) return; // don't flood the UI with identical updates
        lastPct = pct;
        // reading from disk is near-instant; only call it a download if it is actually taking time on the network
        const downloading = !bundled && pct < 100 && Date.now() - t0 > 1200;
        setStatus({
          phase: downloading ? "downloading" : "loading",
          progress: pct,
          message: downloading ? `Downloading voice model (first run only)… ${pct}%` : `Loading voice model… ${pct}%`,
        });
      }
    },
  });

  // warm-up pass so the first real command is fast
  try {
    await asr(new Float32Array(16000));
  } catch {
    /* non-fatal */
  }
  setStatus({ phase: "ready", progress: 100, message: bundled ? "Ready · offline model" : "Ready · model cached on this PC" });
  return asr;
}

function ensure() {
  if (!asrPromise) {
    asrPromise = load().catch((err) => {
      asrPromise = null; // allow retry
      const offline = /fetch|network|ENOTFOUND|EAI_AGAIN|ECONNREFUSED|ETIMEDOUT/i.test(String(err && (err.message || err)));
      setStatus({
        phase: "error",
        progress: 0,
        message: offline
          ? "Couldn't download the voice model. Connect to the internet once (or bundle it with `npm run prefetch-model`)."
          : `Voice engine failed to start: ${err && err.message ? err.message : err}`,
      });
      throw err;
    });
  }
  return asrPromise;
}

function toFloat32(pcm) {
  if (pcm instanceof Float32Array) return pcm;
  if (pcm instanceof ArrayBuffer) return new Float32Array(pcm);
  if (ArrayBuffer.isView(pcm)) return new Float32Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.byteLength / 4));
  throw new Error("Unsupported audio payload");
}

function transcribe(pcm) {
  // serialise requests: one inference at a time
  const run = async () => {
    const asr = await ensure();
    const audio = toFloat32(pcm);
    const t0 = Date.now();
    const opts = IS_ENGLISH_ONLY ? {} : { language: "en", task: "transcribe" };
    const out = await asr(audio, opts);
    return { text: String(out && out.text ? out.text : "").trim(), ms: Date.now() - t0 };
  };
  const p = queue.then(run, run);
  queue = p.catch(() => undefined);
  return p;
}

module.exports = {
  modelId: MODEL_ID,
  getStatus: () => status,
  warmup: () => ensure().then(() => status, () => status),
  transcribe,
  onStatus(cb) {
    listeners.add(cb);
    return () => listeners.delete(cb);
  },
};
