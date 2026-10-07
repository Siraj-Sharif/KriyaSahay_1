/**
 * Records ONE spoken utterance from a chosen microphone and returns 16 kHz mono PCM
 * (the format Whisper expects). Uses simple adaptive voice-activity detection:
 *   - waits for speech (calibrating to the room's noise floor),
 *   - keeps a short pre-roll so the first syllable isn't clipped,
 *   - stops after a pause or when the time limit is hit,
 *   - boosts quiet microphones.
 */
export interface CaptureOptions {
  /** "" = system default microphone */
  deviceId: string;
  /** called ~8×/s with the live input level 0..1 */
  onLevel: (level: number) => void;
  /** called once when speech is first detected */
  onSpeechStart?: () => void;
  signal: AbortSignal;
  silenceMs?: number;
  maxMs?: number;
  noSpeechMs?: number;
}

export type CaptureResult =
  | { kind: "speech"; pcm: Float32Array }
  | { kind: "no-speech" }
  | { kind: "cancelled" };

const TARGET_RATE = 16000;

async function openMic(deviceId: string): Promise<MediaStream> {
  const base: MediaTrackConstraints = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  if (!deviceId) return navigator.mediaDevices.getUserMedia({ audio: base });
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: { ...base, deviceId: { exact: deviceId } } });
  } catch {
    // the chosen microphone was unplugged → fall back to the default one
    return navigator.mediaDevices.getUserMedia({ audio: base });
  }
}

function resampleLinear(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input;
  const ratio = from / to;
  const n = Math.floor(input.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const f = x - i0;
    out[i] = input[i0] * (1 - f) + (input[Math.min(input.length - 1, i0 + 1)] ?? 0) * f;
  }
  return out;
}

export async function captureUtterance(opts: CaptureOptions): Promise<CaptureResult> {
  const { deviceId, onLevel, onSpeechStart, signal, silenceMs = 1000, maxMs = 12000, noSpeechMs = 8000 } = opts;
  if (signal.aborted) return { kind: "cancelled" };

  const stream = await openMic(deviceId);
  if (signal.aborted) {
    stream.getTracks().forEach((t) => t.stop());
    return { kind: "cancelled" };
  }

  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ sampleRate: TARGET_RATE });
  } catch {
    ctx = new AudioContext();
  }
  const rate = ctx.sampleRate;
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(2048, 1, 1);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(processor);
  processor.connect(mute);
  mute.connect(ctx.destination);

  return new Promise<CaptureResult>((resolve) => {
    const chunkMs = (2048 / rate) * 1000;
    const preRollChunks = Math.ceil(450 / chunkMs);
    const preRoll: Float32Array[] = [];
    const recorded: Float32Array[] = [];

    let noise = 0;
    let calibrated = 0;
    const calibrateChunks = Math.ceil(300 / chunkMs);
    let threshold = 0.015;
    let speaking = false;
    let loudChunks = 0;
    let startedAt = performance.now();
    let speechAt = 0;
    let lastVoiceAt = 0;
    let done = false;

    const finish = (r: CaptureResult) => {
      if (done) return;
      done = true;
      signal.removeEventListener("abort", onAbort);
      processor.onaudioprocess = null;
      try {
        source.disconnect();
        processor.disconnect();
        mute.disconnect();
      } catch {
        /* already disconnected */
      }
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close().catch(() => undefined);
      onLevel(0);
      resolve(r);
    };
    const onAbort = () => finish({ kind: "cancelled" });
    signal.addEventListener("abort", onAbort);

    processor.onaudioprocess = (e) => {
      if (done) return;
      const data = new Float32Array(e.inputBuffer.getChannelData(0)); // copy
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      onLevel(Math.min(1, rms * 7));
      const now = performance.now();

      // 1) calibrate to ambient noise
      if (calibrated < calibrateChunks) {
        noise += rms;
        calibrated++;
        if (calibrated === calibrateChunks) threshold = Math.max(0.015, (noise / calibrateChunks) * 3.2);
        preRoll.push(data);
        return;
      }

      if (!speaking) {
        preRoll.push(data);
        if (preRoll.length > preRollChunks) preRoll.shift();
        loudChunks = rms > threshold ? loudChunks + 1 : 0;
        if (loudChunks >= 2) {
          speaking = true;
          speechAt = now;
          lastVoiceAt = now;
          recorded.push(...preRoll);
          onSpeechStart?.();
        } else if (now - startedAt > noSpeechMs) {
          finish({ kind: "no-speech" });
        }
        return;
      }

      recorded.push(data);
      if (rms > threshold * 0.8) lastVoiceAt = now;
      const spokeFor = now - speechAt;
      if ((now - lastVoiceAt > silenceMs && spokeFor > 350) || spokeFor > maxMs) {
        let total = 0;
        for (const c of recorded) total += c.length;
        let pcm: Float32Array = new Float32Array(total);
        let off = 0;
        for (const c of recorded) {
          pcm.set(c, off);
          off += c.length;
        }
        pcm = resampleLinear(pcm, rate, TARGET_RATE);
        // boost quiet microphones (max ×8) so Whisper gets a healthy signal
        let peak = 0;
        for (let i = 0; i < pcm.length; i++) peak = Math.max(peak, Math.abs(pcm[i]));
        if (peak > 0.002 && peak < 0.5) {
          const gain = Math.min(8, 0.8 / peak);
          for (let i = 0; i < pcm.length; i++) pcm[i] *= gain;
        }
        finish({ kind: "speech", pcm });
      }
    };
    startedAt = performance.now();
  });
}
