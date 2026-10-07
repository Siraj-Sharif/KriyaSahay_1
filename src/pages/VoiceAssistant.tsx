import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { AIFace } from "../components/AIFace";
import { MicSelector } from "../components/MicSelector";
import { Corner, GlassCard, Icon } from "../components/ui";
import { useMicDevices } from "../hooks/useMicDevices";
import { useTts } from "../hooks/useTts";
import { useSystem } from "../services/SystemContext";
import { cancelSpeech, setTtsSettings, speakText } from "../services/tts";
import type { EngineStatus, VoiceMessage, VoiceState } from "../services/types";
import { cn } from "../utils/cn";

const STATE_COPY: Record<VoiceState, { title: string; hint: string }> = {
  idle: { title: "Ready", hint: "Tap the microphone and speak a command." },
  listening: { title: "Listening…", hint: "Speak now — I'll stop when you pause" },
  processing: { title: "Thinking…", hint: "Understanding intent → generating command" },
  speaking: { title: "Responding", hint: "Speaking the reply" },
};

const STEPS: { id: VoiceState; label: string }[] = [
  { id: "listening", label: "Listen" },
  { id: "processing", label: "Understand" },
  { id: "speaking", label: "Reply" },
];

const CHIPS = ["Close the hand", "Grip", "Open the hand", "Thumbs up", "OK", "Resume camera control", "System status"];

function SpeakerIcon({ on, className }: { on: boolean; className?: string }) {
  return (
    <svg className={className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4z" />
      {on ? <path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" /> : <path d="M16 9.5l5 5M21 9.5l-5 5" />}
    </svg>
  );
}

function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={() => onChange(!on)}
      className={cn("relative h-5 w-9 shrink-0 rounded-full transition-colors", on ? "bg-cyan-400/80 shadow-[0_0_12px_rgba(34,211,238,0.5)]" : "bg-slate-700")}
    >
      <motion.span className="absolute top-0.5 h-4 w-4 rounded-full bg-white shadow" animate={{ left: on ? 18 : 2 }} transition={{ type: "spring", stiffness: 500, damping: 32 }} />
    </button>
  );
}

export function VoiceAssistant() {
  const { voice } = useSystem();
  const [state, setState] = useState<VoiceState>("idle");
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [level, setLevel] = useState(0);
  const [draft, setDraft] = useState("");
  const [engineStatus, setEngineStatus] = useState<EngineStatus>({ phase: "ready", progress: 100, message: "" });
  const [replayId, setReplayId] = useState<string | null>(null);
  const levelRef = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);
  const replayToken = useRef(0);
  const mics = useMicDevices();
  const tts = useTts();

  // keep the voice engine pointed at the chosen microphone
  useEffect(() => {
    voice.setInputDevice(mics.selected.id);
  }, [voice, mics.selected.id]);

  useEffect(() => {
    return voice.subscribe((s, m, l) => {
      setState(s);
      setMessages(m);
      levelRef.current = l;
      setLevel(l);
    });
  }, [voice]);

  useEffect(() => voice.subscribeEngine(setEngineStatus), [voice]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  // Leaving the voice tab releases microphone/audio capture and cancels any reply in flight.
  useEffect(() => () => {
    voice.stopListening();
    cancelSpeech();
  }, [voice]);

  const active = state !== "idle";
  const onMic = () => (active ? voice.stopListening() : voice.startListening());
  const stepIdx = STEPS.findIndex((x) => x.id === state);

  const replay = async (m: VoiceMessage) => {
    if (replayId === m.id) {
      replayToken.current++;
      cancelSpeech();
      setReplayId(null);
      return;
    }
    const mine = ++replayToken.current;
    setReplayId(m.id);
    await speakText(m.text, () => mine === replayToken.current, true);
    if (mine === replayToken.current) setReplayId(null);
  };

  const testVoice = async () => {
    const mine = ++replayToken.current;
    setReplayId("test");
    await speakText("Kriya Sahay voice output is working. You should hear this reply.", () => mine === replayToken.current, true);
    if (mine === replayToken.current) setReplayId(null);
  };

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-4 lg:grid-cols-12 lg:grid-rows-1">
      {/* ───────────── Left: audio in / out ───────────── */}
      <div className="flex min-h-0 flex-col gap-3 lg:col-span-3">
        <GlassCard title="Audio Input" icon={<Icon.Mic />} right={<span className="mono text-[10px] text-slate-500">{mics.devices.length > 1 ? `${mics.devices.length - 1} detected` : "default"}</span>}>
          <MicSelector
            devices={mics.devices}
            selected={mics.selected}
            onSelect={mics.select}
            onRescan={() => void mics.refresh()}
            needsPermission={mics.needsPermission}
            permission={mics.permission}
            inputsFound={mics.inputsFound}
            missingSelected={mics.missingSelected}
            onRequestAccess={mics.requestAccess}
            locked={active}
          />
          <div className="border-t border-white/5 px-3 py-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="mono text-[9px] uppercase tracking-[0.25em] text-slate-500">Engine</span>
              <span className={cn("mono rounded-full px-2 py-0.5 text-[9px] uppercase tracking-wider ring-1", voice.engine.offline ? "bg-emerald-500/10 text-emerald-300 ring-emerald-400/30" : "bg-white/5 text-slate-300 ring-white/10")}>
                {voice.engine.offline ? "Offline" : "Online"}
              </span>
            </div>
            <div className="mt-0.5 truncate text-xs text-slate-200" title={voice.engine.name}>
              {voice.engine.name}
            </div>
            {(engineStatus.phase === "loading" || engineStatus.phase === "downloading") && (
              <div className="mt-1.5">
                <div className="mono mb-1 flex justify-between text-[10px] text-cyan-200">
                  <span className="truncate pr-2">{engineStatus.message}</span>
                  <span>{engineStatus.progress}%</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-white/5">
                  <motion.div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-500" animate={{ width: `${Math.max(4, engineStatus.progress)}%` }} />
                </div>
              </div>
            )}
            {engineStatus.phase === "ready" && engineStatus.message && (
              <div className="mono mt-1 inline-flex items-center gap-1.5 text-[10px] text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                {engineStatus.message}
              </div>
            )}
            {engineStatus.phase === "error" && <p className="mt-1.5 rounded-lg bg-rose-500/10 px-2 py-1.5 text-[10px] leading-snug text-rose-300 ring-1 ring-rose-400/30">{engineStatus.message}</p>}
            <p className="mt-1.5 hidden text-[10px] leading-snug text-slate-500 [@media(min-height:880px)]:block">{voice.engine.note}</p>
          </div>
        </GlassCard>

        <GlassCard
          title="Voice Output"
          icon={<SpeakerIcon on className="h-4 w-4" />}
          right={
            <div className="flex items-center gap-2">
              <span className="mono text-[9px] uppercase tracking-wider text-slate-500">{tts.settings.enabled ? "Speaking" : "Muted"}</span>
              <Switch on={tts.settings.enabled} onChange={(v) => setTtsSettings({ enabled: v })} label="Speak replies aloud" />
            </div>
          }
        >
          {!tts.supported ? (
            <p className="p-3 text-[11px] leading-snug text-amber-300">Voice output isn't available in this environment. Replies will show as text only.</p>
          ) : (
            <div className={cn("flex flex-col gap-2.5 p-3 transition-opacity", !tts.settings.enabled && "opacity-50")}>
              <div className="flex items-center gap-2.5">
                <SpeakerIcon on={tts.settings.volume > 0} className="h-4 w-4 shrink-0 text-slate-400" />
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={tts.settings.volume}
                  onChange={(e) => setTtsSettings({ volume: Number(e.target.value) })}
                  aria-label="Reply volume"
                  className="h-1 min-w-0 flex-1 cursor-pointer accent-cyan-400"
                />
                <span className="mono w-8 text-right text-[10px] text-slate-300">{Math.round(tts.settings.volume * 100)}%</span>
              </div>
              <select
                value={tts.settings.voiceURI}
                onChange={(e) => setTtsSettings({ voiceURI: e.target.value })}
                aria-label="Reply voice"
                className="w-full rounded-lg border border-white/10 bg-black/40 px-2.5 py-1.5 text-xs text-slate-100 outline-none transition focus:border-cyan-400/50"
              >
                <option value="" className="bg-[#070a12]">
                  Automatic voice
                </option>
                {tts.voices.map((v) => (
                  <option key={v.voiceURI} value={v.voiceURI} className="bg-[#070a12]">
                    {v.name}
                  </option>
                ))}
              </select>
              <button
                onClick={() => void (replayId === "test" ? (replayToken.current++, cancelSpeech(), setReplayId(null)) : testVoice())}
                disabled={active}
                className="flex items-center justify-center gap-2 rounded-lg bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-200 ring-1 ring-cyan-400/30 transition hover:bg-cyan-500/20 disabled:opacity-40"
              >
                <SpeakerIcon on className="h-3.5 w-3.5" />
                {replayId === "test" ? "Stop" : "Test voice"}
              </button>
            </div>
          )}
        </GlassCard>

        <GlassCard title="Pipeline" icon={<Icon.Chart />} className="hidden [@media(min-height:740px)]:block">
          <div className="flex items-center gap-1.5 px-3 py-3">
            {STEPS.map((p, i) => {
              const on = state === p.id;
              const done = stepIdx > i;
              return (
                <div key={p.id} className="flex flex-1 items-center gap-1.5">
                  <div className={cn("flex flex-1 flex-col items-center gap-1 rounded-lg border py-1.5 transition", on ? "border-cyan-400/40 bg-cyan-500/10" : "border-white/5 bg-white/[0.02]")}>
                    <span className={cn("mono flex h-5 w-5 items-center justify-center rounded-full text-[9px] ring-1", on ? "bg-cyan-400 text-black ring-cyan-300" : done ? "bg-emerald-500/20 text-emerald-300 ring-emerald-400/40" : "text-slate-500 ring-white/10")}>{done ? "✓" : i + 1}</span>
                    <span className={cn("text-[10px]", on ? "text-white" : "text-slate-400")}>{p.label}</span>
                  </div>
                  {i < STEPS.length - 1 && <span className="h-px w-2 bg-white/10" />}
                </div>
              );
            })}
          </div>
        </GlassCard>
      </div>

      {/* ───────────── Centre: avatar ───────────── */}
      <GlassCard strong className="relative min-h-[420px] lg:col-span-5 lg:min-h-0">
        <Corner />
        <div className="absolute inset-0 grid-bg opacity-40 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
        <div className="absolute left-5 top-4 z-10">
          <div className="mono text-[9px] uppercase tracking-[0.3em] text-slate-500">Kriya Sahay Assistant</div>
          <AnimatePresence mode="wait">
            <motion.div key={state} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
              <div className="text-lg font-semibold text-white">{STATE_COPY[state].title}</div>
              <div className="text-xs text-slate-400">{STATE_COPY[state].hint}</div>
            </motion.div>
          </AnimatePresence>
          <span className={cn("mono mt-2 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[9px] uppercase tracking-wider ring-1", voice.mode === "live" ? "bg-emerald-500/10 text-emerald-300 ring-emerald-400/30" : "bg-amber-500/10 text-amber-300 ring-amber-400/30")}>
            <span className={cn("h-1.5 w-1.5 rounded-full", voice.mode === "live" ? "bg-emerald-400" : "bg-amber-400")} />
            {voice.modeLabel}
          </span>
        </div>
        <div className="absolute right-5 top-4 z-10 flex flex-col items-end gap-1">
          <div className="mono text-[9px] uppercase tracking-[0.3em] text-slate-500">Level</div>
          <div className="flex h-5 items-end gap-0.5">
            {Array.from({ length: 14 }).map((_, i) => (
              <motion.span key={i} className="w-1 rounded-sm bg-cyan-400" animate={{ height: active ? `${20 + Math.max(0, level - i / 18) * 80 * (0.6 + 0.4 * Math.sin(i * 1.7 + level * 9) ** 2)}%` : "15%" }} transition={{ duration: 0.08 }} style={{ opacity: i / 14 < level || !active ? 1 : 0.25 }} />
            ))}
          </div>
        </div>

        <div className="absolute inset-0 pb-24 pt-16">
          <AIFace state={state} levelRef={levelRef} />
        </div>

        {/* Mic + quick mute */}
        <div className="absolute inset-x-0 bottom-3 z-10 flex flex-col items-center gap-1.5">
          <div className="relative flex items-center justify-center">
            <button onClick={onMic} aria-label={active ? "Cancel" : "Start listening"} className="group relative flex h-14 w-14 items-center justify-center rounded-full">
              {active && (
                <>
                  <span className="absolute inset-0 rounded-full bg-cyan-400/30 pulse-ring" />
                  <span className="absolute inset-0 rounded-full bg-violet-400/20 pulse-ring [animation-delay:0.5s]" />
                </>
              )}
              <motion.span
                animate={{ scale: active ? 1 + level * 0.15 : 1 }}
                className={cn(
                  "relative flex h-14 w-14 items-center justify-center rounded-full ring-2 transition",
                  active ? "bg-gradient-to-br from-cyan-400 to-violet-500 ring-cyan-300/60 shadow-[0_0_40px_rgba(34,211,238,0.5)]" : "bg-white/[0.06] ring-white/15 group-hover:ring-cyan-400/50",
                )}
              >
                <Icon.Mic className={cn("h-6 w-6", active ? "text-black" : "text-cyan-200")} />
              </motion.span>
            </button>
            <button
              onClick={() => {
                if (tts.settings.enabled) cancelSpeech();
                setTtsSettings({ enabled: !tts.settings.enabled });
              }}
              title={tts.settings.enabled ? "Mute spoken replies" : "Unmute spoken replies"}
              aria-label="Toggle spoken replies"
              className={cn(
                "absolute left-[calc(100%+14px)] flex h-9 w-9 items-center justify-center rounded-full border transition",
                tts.settings.enabled ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-200 hover:bg-cyan-500/20" : "border-white/10 bg-white/[0.04] text-slate-500 hover:text-slate-300",
              )}
            >
              <SpeakerIcon on={tts.settings.enabled} className="h-4 w-4" />
            </button>
          </div>
          <span className="mono text-[10px] uppercase tracking-[0.25em] text-slate-500">{active ? "Tap to cancel" : "Tap to speak"}</span>
        </div>
      </GlassCard>

      {/* ───────────── Right: conversation ───────────── */}
      <GlassCard
        title="Conversation"
        icon={<Icon.Chart />}
        className="flex min-h-[360px] flex-col lg:col-span-4 lg:min-h-0"
        right={<span className="mono text-[10px] text-slate-500">{messages.length} msgs</span>}
      >
        {/* only the chat log scrolls internally; the page itself never does */}
        <div ref={logRef} className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
          <AnimatePresence initial={false}>
            {messages.map((m) => (
              <motion.div key={m.id} initial={{ opacity: 0, y: 10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                <div className={cn("max-w-[88%] rounded-2xl px-3 py-2 text-[13px] leading-relaxed", m.role === "user" ? "rounded-br-sm bg-white/[0.06] text-slate-100 ring-1 ring-white/10" : "rounded-bl-sm bg-gradient-to-br from-cyan-500/15 to-violet-500/15 text-slate-100 ring-1 ring-cyan-400/20")}>
                  {m.text}
                  <div className="mt-1 flex items-center gap-2">
                    {m.command && (
                      <span className="inline-flex items-center gap-1.5 rounded-md bg-black/40 px-1.5 py-0.5 ring-1 ring-cyan-400/30">
                        <Icon.Serial className="h-3 w-3 text-cyan-300" />
                        <span className="mono text-[10px] text-cyan-200">NG1|{m.command}</span>
                      </span>
                    )}
                    <span className="mono text-[9px] text-slate-500">{new Date(m.timestamp).toLocaleTimeString()}</span>
                    {m.role === "assistant" && tts.supported && (
                      <button
                        onClick={() => void replay(m)}
                        disabled={active}
                        title={replayId === m.id ? "Stop" : "Play this reply aloud"}
                        aria-label="Play reply aloud"
                        className={cn("ml-auto flex h-5 w-5 items-center justify-center rounded-full transition disabled:opacity-30", replayId === m.id ? "bg-cyan-400 text-black" : "text-slate-400 hover:bg-white/10 hover:text-cyan-200")}
                      >
                        <SpeakerIcon on className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {state === "processing" && (
            <div className="flex gap-1 px-3">
              {[0, 1, 2].map((i) => (
                <motion.span key={i} className="h-1.5 w-1.5 rounded-full bg-cyan-300" animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1, delay: i * 0.2 }} />
              ))}
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (draft.trim() && !active) {
              voice.sendText(draft);
              setDraft("");
            }
          }}
          className="flex shrink-0 gap-2 border-t border-white/5 p-3"
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={active}
            placeholder="Type a command, e.g. “grip”"
            className="min-w-0 flex-1 rounded-lg border border-white/10 bg-black/30 px-3 py-1.5 text-xs text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-400/50 disabled:opacity-50"
          />
          <button type="submit" disabled={active || !draft.trim()} className="rounded-lg bg-cyan-500/15 px-3 py-1.5 text-xs font-semibold text-cyan-200 ring-1 ring-cyan-400/40 transition hover:bg-cyan-500/25 disabled:opacity-40">
            Send
          </button>
        </form>
        <div className="hidden shrink-0 flex-wrap gap-1.5 border-t border-white/5 p-3 [@media(min-height:700px)]:flex">
          {CHIPS.map((c) => (
            <button key={c} disabled={active} onClick={() => voice.sendText(c)} className="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400/40 hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-40">
              “{c}”
            </button>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
