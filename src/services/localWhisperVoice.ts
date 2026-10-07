import type { Unsubscribe, VoiceAdapter, VoiceInterpretation, VoiceInterpreter } from "./adapters";
import { captureUtterance } from "./audioCapture";
import { parseIntent } from "./intents";
import { cancelSpeech, speakText } from "./tts";
import type { EngineStatus, Gesture, VoiceMessage, VoiceState } from "./types";

interface Options {
  /** gesture = drive the hand, null = hand control back to the camera */
  onCommand: (g: Gesture | null) => void;
  /** returns a spoken-friendly system status sentence */
  getStatus: () => string;
  /**
   * Phase 2: authoritative interpretation by the running Python pipeline.
   *
   * When present the pipeline decides the intent, validates the command and transmits it
   * over its own serial link; this adapter only previews the returned gesture. When absent
   * (browser demo) the local `intents.ts` parser is used and the command goes through the
   * serial adapter.
   */
  interpret?: VoiceInterpreter;
}

export function isDesktopSttAvailable() {
  return typeof window !== "undefined" && !!window.neurogrip?.stt;
}

/** Whisper sometimes "hears" these in near-silence — never treat them as commands. */
function cleanTranscript(raw: string): string {
  const t = raw
    .replace(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*/g, " ") // [BLANK_AUDIO], (music), *noise*
    .replace(/\s+/g, " ")
    .trim();
  if (/^(you|thank you\.?|thanks( for watching)?\.?|bye\.?|the end\.?|uh|um|hmm|oh|\.+|!+)$/i.test(t)) return "";
  return t;
}

/**
 * Desktop voice adapter — fully on-device:
 *   selected mic → VAD recorder (16 kHz) → Whisper (Electron main process) → intent → command → OS voice reply.
 * Nothing leaves the computer and no internet is needed once the model is on disk.
 */
export function createLocalWhisperVoice({ onCommand, getStatus, interpret }: Options): VoiceAdapter {
  const stt = window.neurogrip!.stt;
  const listeners = new Set<(s: VoiceState, m: VoiceMessage[], l: number) => void>();
  const engineListeners = new Set<(s: EngineStatus) => void>();

  let state: VoiceState = "idle";
  let level = 0;
  let inputId = "";
  let session = 0;
  let abort: AbortController | null = null;
  let heardSpeech = false;
  let engine: EngineStatus = { phase: "loading", progress: 0, message: "Starting voice engine…" };
  let messages: VoiceMessage[] = [
    {
      id: "m0",
      role: "assistant",
      text: "Voice interface online. Everything runs on this computer. Tap the microphone and speak, or type a command.",
      timestamp: Date.now(),
    },
  ];

  const emit = () => listeners.forEach((l) => l(state, messages, level));
  const setState = (s: VoiceState) => {
    state = s;
    emit();
  };
  const push = (role: VoiceMessage["role"], text: string, command?: Gesture) => {
    messages = [...messages, { id: `${role[0]}${Date.now()}${Math.random().toString(36).slice(2, 5)}`, role, text, command, timestamp: Date.now() }];
  };
  const setEngine = (s: EngineStatus) => {
    engine = s;
    engineListeners.forEach((l) => l(s));
  };

  // engine status from the desktop side
  stt.onStatus(setEngine);
  // The Electron main-process supervisor may already be warming this same cached model;
  // warmup() shares that single promise and reports its real state without reloading it.
  void stt.warmup().then(setEngine);

  // processing "pulse" level while Whisper works
  let pulse: number | undefined;
  const startPulse = () => {
    window.clearInterval(pulse);
    pulse = window.setInterval(() => {
      level = state === "speaking" ? 0.35 + Math.random() * 0.65 : state === "processing" ? 0.25 : level;
      emit();
    }, 80);
  };
  const stopPulse = () => {
    window.clearInterval(pulse);
    level = 0;
  };

  const fail = (text: string) => {
    push("assistant", text);
    stopPulse();
    setState("idle");
  };

  const respond = async (heard: string, mySession: number) => {
    push("user", heard);
    setState("processing");
    startPulse();
    if (mySession !== session) return;

    // 1. The running pipeline interprets the transcript, validates the command and
    //    transmits it (same serial owner as camera detections).
    let remote: VoiceInterpretation | null = null;
    if (interpret) {
      try {
        remote = await interpret(heard);
      } catch (err) {
        remote = { kind: "unknown", gesture: null, reply: "Voice command failed.", error: String(err) };
      }
    }
    if (mySession !== session) return;

    let reply: string;
    let command: Gesture | undefined;
    if (remote) {
      reply = remote.reply || "Done.";
      command = remote.gesture ?? undefined;
      if (remote.gesture) onCommand(remote.gesture);
      if (remote.kind === "release") onCommand(null);
      if (remote.error) reply = `${reply} (${remote.error})`;
    } else {
      // Browser / mock path: local intent parser + the serial adapter.
      const intent = parseIntent(heard);
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
    }
    push("assistant", reply, command);
    setState("speaking");
    await speakText(reply, () => mySession === session);
    if (mySession !== session) return;
    stopPulse();
    setState("idle");
  };

  const start = async () => {
    if (state !== "idle") return;

    if (engine.phase !== "ready") {
      if (engine.phase === "error" || engine.phase === "idle") void stt.warmup().then(setEngine);
      push(
        "assistant",
        engine.phase === "error"
          ? `${engine.message} I'm retrying now — tap the microphone again in a moment. Typed commands work meanwhile.`
          : `The voice model is still getting ready (${engine.progress}%). Typed commands work right now — try the microphone again in a moment.`,
      );
      emit();
      return;
    }

    const mySession = ++session;
    cancelSpeech();
    abort = new AbortController();
    heardSpeech = false;
    setState("listening");

    let result;
    try {
      result = await captureUtterance({
        deviceId: inputId,
        signal: abort.signal,
        onLevel: (l) => {
          if (mySession !== session) return;
          level = l;
          emit();
        },
        onSpeechStart: () => {
          heardSpeech = true;
        },
      });
    } catch {
      if (mySession === session) fail("I can't open the microphone. Check that Windows allows microphone access for desktop apps, and that no other app is using it.");
      return;
    }
    if (mySession !== session || result.kind === "cancelled") return;
    if (result.kind === "no-speech") return fail(heardSpeech ? "I didn't catch that. Try again." : "I didn't hear anything. Check the selected microphone with the Mic test, then try again.");

    // Whisper
    setState("processing");
    startPulse();
    const res = await stt.transcribe(result.pcm);
    if (mySession !== session) return;
    if (res.error) return fail(`Speech recognition failed: ${res.error}`);

    const text = cleanTranscript(res.text);
    if (!text) return fail("I didn't catch that. Try speaking a little closer to the microphone.");
    await respond(text, mySession);
  };

  return {
    mode: "live",
    modeLabel: "Live · On-device Whisper",
    engine: {
      name: "Whisper base.en · on-device",
      offline: true,
      note: "Runs entirely on this computer — no internet, and your audio never leaves it. It listens to the microphone selected above.",
    },
    setInputDevice(id: string) {
      inputId = id;
    },
    subscribe(cb): Unsubscribe {
      listeners.add(cb);
      cb(state, messages, level);
      return () => {
        listeners.delete(cb);
      };
    },
    subscribeEngine(cb): Unsubscribe {
      engineListeners.add(cb);
      cb(engine);
      return () => {
        engineListeners.delete(cb);
      };
    },
    startListening() {
      void start();
    },
    stopListening() {
      session++;
      abort?.abort();
      cancelSpeech();
      stopPulse();
      setState("idle");
    },
    sendText(text: string) {
      const t = text.trim();
      if (!t || state !== "idle") return;
      const mySession = ++session;
      void respond(t, mySession);
    },
  };
}
