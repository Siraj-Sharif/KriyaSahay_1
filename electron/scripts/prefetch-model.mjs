// Downloads the Whisper model into electron/models so the installer ships with it
// and the voice assistant works with NO internet, ever.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline, env } from "@huggingface/transformers";

const here = path.dirname(fileURLToPath(import.meta.url));
const MODEL_ID = process.env.NEUROGRIP_STT_MODEL || "Xenova/whisper-base.en";

env.cacheDir = path.join(here, "..", "models");
env.allowRemoteModels = true;

let last = 0;
console.log(`› Downloading ${MODEL_ID} …`);
await pipeline("automatic-speech-recognition", MODEL_ID, {
  dtype: "q8",
  progress_callback: (p) => {
    if (p.status === "progress" && Date.now() - last > 1500) {
      last = Date.now();
      console.log(`  ${p.file} ${Math.round(p.progress)}%`);
    }
  },
});
console.log(`✓ Model saved to ${env.cacheDir}`);
