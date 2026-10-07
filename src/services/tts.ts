/**
 * Voice output (text-to-speech) using the operating system's voices — offline on Windows / macOS.
 * One shared module so every voice adapter speaks the same way and the UI can control it
 * (mute, volume, choice of voice).
 */

export interface TtsSettings {
  enabled: boolean;
  /** 0..1 */
  volume: number;
  /** "" = automatic best English voice */
  voiceURI: string;
}

const KEY = "neurogrip.tts";
const DEFAULTS: TtsSettings = { enabled: true, volume: 1, voiceURI: "" };

function load(): TtsSettings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

let settings: TtsSettings = load();
const listeners = new Set<() => void>();

export const ttsSupported = () => typeof window !== "undefined" && "speechSynthesis" in window;
export const getTtsSettings = () => settings;
export function setTtsSettings(patch: Partial<TtsSettings>) {
  settings = { ...settings, ...patch };
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable */
  }
  listeners.forEach((l) => l());
}
export function subscribeTts(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

export function getVoicesNow(): SpeechSynthesisVoice[] {
  return ttsSupported() ? window.speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang)) : [];
}

function waitForVoices(timeoutMs = 900): Promise<SpeechSynthesisVoice[]> {
  return new Promise((resolve) => {
    if (!ttsSupported()) return resolve([]);
    if (window.speechSynthesis.getVoices().length) return resolve(window.speechSynthesis.getVoices());
    const done = () => resolve(window.speechSynthesis.getVoices());
    window.speechSynthesis.addEventListener("voiceschanged", done, { once: true });
    window.setTimeout(done, timeoutMs);
  });
}

function chooseVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | undefined {
  if (settings.voiceURI) {
    const chosen = voices.find((v) => v.voiceURI === settings.voiceURI);
    if (chosen) return chosen;
  }
  return (
    voices.find((v) => /en[-_](US|GB)/i.test(v.lang) && /natural|aria|jenny|zira|david|mark|samantha|google/i.test(v.name)) ??
    voices.find((v) => /^en/i.test(v.lang)) ??
    voices[0]
  );
}

export function cancelSpeech() {
  if (ttsSupported()) window.speechSynthesis.cancel();
}

/**
 * Speaks `text` and resolves when finished. `isCurrent` lets callers abort stale speech.
 * When voice replies are muted it just pauses briefly so the reply still "lands" in the UI.
 * Pass `force` to speak even if muted (used by the "Test voice" button).
 */
export async function speakText(text: string, isCurrent: () => boolean = () => true, force = false): Promise<void> {
  try {
    await speakInner(text, isCurrent, force);
  } catch {
    // a speech-engine failure must never leave the assistant stuck in "speaking"
    cancelSpeech();
  }
}

async function speakInner(text: string, isCurrent: () => boolean, force: boolean): Promise<void> {
  if (!isCurrent()) return;
  if (!settings.enabled && !force) return sleep(700);
  if (!ttsSupported()) return sleep(700);

  const voices = await waitForVoices();
  if (!isCurrent()) return;

  window.speechSynthesis.cancel();
  await sleep(60); // Chromium drops an utterance queued in the same tick as cancel()
  if (!isCurrent()) return;

  await new Promise<void>((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.02;
    u.pitch = 0.95;
    u.volume = Math.max(0, Math.min(1, settings.volume));
    const v = chooseVoice(voices);
    if (v) {
      u.voice = v;
      u.lang = v.lang;
    } else {
      u.lang = "en-US";
    }
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(watch);
      window.clearTimeout(cap);
      resolve();
    };
    u.onend = finish;
    u.onerror = finish;
    // stop early if the caller moved on (user pressed stop / started a new command)
    const watch = window.setInterval(() => {
      if (!isCurrent()) {
        window.speechSynthesis.cancel();
        finish();
      }
    }, 150);
    // some engines never fire onend
    const cap = window.setTimeout(finish, Math.max(4000, text.length * 110));
    window.speechSynthesis.speak(u);
  });
}
