import type { Unsubscribe, VoiceAdapter } from "./adapters";
import { parseIntent } from "./intents";
import { cancelSpeech, speakText } from "./tts";
import type { Gesture, VoiceMessage, VoiceState } from "./types";

/* eslint-disable @typescript-eslint/no-explicit-any */

export function isSpeechRecognitionSupported() {
  return typeof window !== "undefined" && !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
}

interface Options {
  /** gesture = drive the hand, null = hand control back to the camera */
  onCommand: (g: Gesture | null) => void;
  /** returns a spoken-friendly system status sentence */
  getStatus: () => string;
}

/**
 * Real voice adapter built on the browser's Web Speech API:
 *   mic → SpeechRecognition → parseIntent → command → speechSynthesis reply.
 * Each stage can be swapped for an open-source model (Whisper / Vosk / Piper / LLM)
 * without touching the UI — only this file changes.
 */
const IS_ELECTRON = typeof navigator !== "undefined" && /electron/i.test(navigator.userAgent);

export function createWebSpeechVoice({ onCommand, getStatus }: Options): VoiceAdapter {
  const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  const listeners = new Set<(s: VoiceState, m: VoiceMessage[], l: number) => void>();

  let state: VoiceState = "idle";
  let level = 0;
  let messages: VoiceMessage[] = [
    { id: "m0", role: "assistant", text: "Voice interface online. Tap the microphone and speak a command, or type one below.", timestamp: Date.now() },
  ];
  let interim = "";
  let recognition: any = null;
  let session = 0;
  let inputId = ""; // "" = system default mic

  let micStream: MediaStream | null = null;
  let audioCtx: AudioContext | null = null;
  let analyser: AnalyserNode | null = null;
  let levelTimer: number | undefined;

  const emit = () => {
    const shown = interim ? [...messages, { id: "interim", role: "user" as const, text: interim + " …", timestamp: Date.now() }] : messages;
    listeners.forEach((l) => l(state, shown, level));
  };
  const setState = (s: VoiceState) => {
    state = s;
    emit();
  };
  const push = (role: VoiceMessage["role"], text: string, command?: Gesture) => {
    messages = [...messages, { id: `${role[0]}${Date.now()}${Math.random().toString(36).slice(2, 5)}`, role, text, command, timestamp: Date.now() }];
  };

  /* ---------------- mic level meter ---------------- */
  const startMeter = async (): Promise<boolean> => {
    try {
      const base = { echoCancellation: true, noiseSuppression: true };
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: inputId ? { ...base, deviceId: { exact: inputId } } : base });
      } catch (err) {
        // selected mic was unplugged → fall back to the default one
        if (!inputId) throw err;
        micStream = await navigator.mediaDevices.getUserMedia({ audio: base });
      }
      audioCtx = new AudioContext();
      const src = audioCtx.createMediaStreamSource(micStream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      return true;
    } catch {
      return false;
    }
  };
  const stopMeter = () => {
    micStream?.getTracks().forEach((t) => t.stop());
    micStream = null;
    void audioCtx?.close().catch(() => undefined);
    audioCtx = null;
    analyser = null;
  };
  const startLevelLoop = () => {
    window.clearInterval(levelTimer);
    const buf = new Uint8Array(256);
    levelTimer = window.setInterval(() => {
      if (state === "listening" && analyser) {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        level = Math.min(1, Math.sqrt(sum / buf.length) * 5);
      } else if (state === "speaking") level = 0.35 + Math.random() * 0.65;
      else if (state === "processing") level = 0.2;
      else level = 0;
      emit();
    }, 70);
  };
  const stopLevelLoop = () => {
    window.clearInterval(levelTimer);
    level = 0;
  };

  /* ---------------- core pipeline ---------------- */
  const respond = async (heard: string, mySession: number) => {
    push("user", heard);
    interim = "";
    setState("processing");
    if (mySession !== session) return;

    const intent = parseIntent(heard);
    let reply: string;
    let command: Gesture | undefined;
    switch (intent.kind) {
      case "gesture":
        reply = intent.reply;
        command = intent.gesture;
        onCommand(intent.gesture);
        break;
      case "release":
        reply = intent.reply;
        onCommand(null);
        break;
      case "status":
        reply = getStatus();
        break;
      default:
        reply = intent.reply;
    }
    push("assistant", reply, command);
    setState("speaking");
    await speakText(reply, () => mySession === session);
    if (mySession !== session) return;
    stopLevelLoop();
    setState("idle");
  };

  const finishWithError = (text: string) => {
    interim = "";
    push("assistant", text);
    stopMeter();
    stopLevelLoop();
    setState("idle");
  };

  const start = async () => {
    if (state !== "idle") return;
    const mySession = ++session;
    cancelSpeech();

    const micOk = await startMeter();
    if (mySession !== session) return stopMeter();
    if (!micOk) return finishWithError("I can't access the microphone. Allow microphone permission for this page and try again.");

    recognition = new SR();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    let finalText = "";
    let errored = false;

    recognition.onresult = (e: any) => {
      let live = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) finalText += r[0].transcript;
        else live += r[0].transcript;
      }
      interim = (finalText + live).trim();
      emit();
    };
    recognition.onerror = (e: any) => {
      errored = true;
      if (mySession !== session) return;
      const map: Record<string, string> = {
        "not-allowed": "Microphone or speech permission was denied. Allow it in the browser's site settings.",
        "service-not-allowed": "Speech recognition is blocked in this browser context.",
        network: IS_ELECTRON
          ? "The desktop app has no built-in speech service. Plug in an offline engine (Whisper or Vosk) — typed commands still work."
          : "The speech service couldn't be reached. Chrome's recognition needs an internet connection.",
        "no-speech": "I didn't hear anything. Tap the microphone and try again.",
        "audio-capture": "No microphone was found.",
        aborted: "",
      };
      const msg = map[e.error] ?? `Speech recognition error: ${e.error}`;
      if (msg) finishWithError(msg);
    };
    recognition.onend = () => {
      stopMeter();
      if (mySession !== session || errored) return;
      const text = (finalText || interim).trim();
      if (text) void respond(text, mySession);
      else finishWithError("I didn't catch that. Tap the microphone and try again.");
    };

    try {
      const track = micStream?.getAudioTracks()[0];
      if (inputId && track) {
        // Newer Chromium can recognise from a specific track; older ones ignore the argument
        // and use the system default mic.
        try {
          recognition.start(track);
        } catch {
          recognition.start();
        }
      } else {
        recognition.start();
      }
      setState("listening");
      startLevelLoop();
    } catch {
      finishWithError("Couldn't start speech recognition. Please try again.");
    }
  };

  return {
    mode: "live",
    modeLabel: IS_ELECTRON ? "Desktop · offline engine required" : "Live · Browser Speech API",
    setInputDevice(id: string) {
      inputId = id;
    },
    subscribeEngine(cb): Unsubscribe {
      cb({ phase: "ready", progress: 100, message: "Browser speech service" });
      return () => undefined;
    },
    engine: {
      name: IS_ELECTRON ? "Browser Speech API (unavailable in desktop)" : "Browser Speech API (Chrome / Edge)",
      offline: false,
      note: IS_ELECTRON
        ? "Desktop builds can't use Google's speech service. Swap this engine for Whisper or Vosk, which read audio from the selected microphone directly."
        : "Chrome's recognizer may ignore the selected mic and use the system default. If your external mic isn't heard, set it as the default input in Windows Sound settings.",
    },
    subscribe(cb): Unsubscribe {
      listeners.add(cb);
      cb(state, messages, level);
      return () => {
        listeners.delete(cb);
      };
    },
    startListening() {
      void start();
    },
    stopListening() {
      session++;
      try {
        recognition?.abort();
      } catch {
        /* noop */
      }
      cancelSpeech();
      stopMeter();
      stopLevelLoop();
      interim = "";
      setState("idle");
    },
    sendText(text: string) {
      const t = text.trim();
      if (!t || state !== "idle") return;
      const mySession = ++session;
      startLevelLoop();
      void respond(t, mySession);
    },
  };
}
