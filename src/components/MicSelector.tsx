import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { isDesktopApp, type MicDevice, type MicKind, type MicPermission } from "../hooks/useMicDevices";
import { cn } from "../utils/cn";
import { Popover } from "./Popover";

interface Props {
  devices: MicDevice[];
  selected: MicDevice;
  onSelect: (id: string) => void;
  onRescan: () => void;
  needsPermission: boolean;
  permission: MicPermission;
  /** audio inputs the OS reported */
  inputsFound: number;
  /** a mic was chosen earlier but is no longer connected */
  missingSelected: boolean;
  onRequestAccess: () => Promise<void>;
  /** Disable switching while the assistant is actively listening. */
  locked?: boolean;
}

const KIND_TITLE: Record<MicKind, string> = {
  default: "System default",
  "built-in": "Laptop microphone",
  headset: "Headset / earbuds",
  external: "External / USB microphone",
  virtual: "Other inputs (Stereo Mix, virtual)",
};
const KIND_SHORT: Record<MicKind, string> = {
  default: "System default",
  "built-in": "Laptop",
  headset: "Headset",
  external: "External",
  virtual: "Other",
};
const GROUP_ORDER: MicKind[] = ["default", "built-in", "headset", "external", "virtual"];

function MicGlyph({ kind, className }: { kind: MicKind; className?: string }) {
  const c = className ?? "h-4 w-4";
  const p = { className: c, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (kind) {
    case "headset":
      return (
        <svg {...p}>
          <path d="M4 14v-2a8 8 0 0 1 16 0v2" />
          <rect x="3" y="13" width="4" height="6" rx="1.5" />
          <rect x="17" y="13" width="4" height="6" rx="1.5" />
          <path d="M19 19c0 1.7-1.8 2.5-4 2.5" />
        </svg>
      );
    case "external":
      return (
        <svg {...p}>
          <rect x="9" y="2.5" width="6" height="11" rx="3" />
          <path d="M5 11a7 7 0 0 0 14 0M12 18v3M8.5 21h7" />
        </svg>
      );
    case "built-in":
      return (
        <svg {...p}>
          <rect x="5" y="5" width="14" height="10" rx="1.5" />
          <path d="M2.5 19h19M12 8v4" />
        </svg>
      );
    case "virtual":
      return (
        <svg {...p}>
          <path d="M3 12h2.5l2-6 3 12 3-9 2 5 1.5-2H21" />
        </svg>
      );
    default:
      return (
        <svg {...p}>
          <circle cx="12" cy="12" r="8.5" />
          <path d="M12 7.5v5l3 2" />
        </svg>
      );
  }
}

/** Opens the chosen mic briefly and reports its live input level (0..1). */
function useMicTest(deviceId: string) {
  const [testing, setTesting] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!testing) return;
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let raf = 0;
    let cancelled = false;
    const timeout = window.setTimeout(() => setTesting(false), 12000);
    setError(null);

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { deviceId: { exact: deviceId } } : true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        ctx = new AudioContext();
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 512;
        ctx.createMediaStreamSource(stream).connect(analyser);
        const buf = new Uint8Array(analyser.fftSize);
        const loop = () => {
          raf = requestAnimationFrame(loop);
          analyser.getByteTimeDomainData(buf);
          let sum = 0;
          for (let i = 0; i < buf.length; i++) {
            const v = (buf[i] - 128) / 128;
            sum += v * v;
          }
          setLevel((p) => p + (Math.min(1, Math.sqrt(sum / buf.length) * 6) - p) * 0.4);
        };
        loop();
      } catch {
        setError("Couldn't open this microphone. It may be disabled in Windows Sound settings, or another app is using it exclusively.");
        setTesting(false);
      }
    })();

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      void ctx?.close().catch(() => undefined);
      setLevel(0);
    };
  }, [testing, deviceId]);

  return { testing, level, error, toggle: () => setTesting((t) => !t) };
}

export function MicSelector({ devices, selected, onSelect, onRescan, needsPermission, permission, inputsFound, missingSelected, onRequestAccess, locked }: Props) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement>(null);
  const test = useMicTest(selected.id);
  const desktop = isDesktopApp();
  const close = useCallback(() => setOpen(false), []);

  // fresh scan every time the list is opened
  useEffect(() => {
    if (open) onRescan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const groups = GROUP_ORDER.map((k) => ({ kind: k, items: devices.filter((d) => d.kind === k) })).filter((g) => g.items.length);
  const realCount = devices.length - 1; // minus the "System default" row
  const BARS = 28;

  return (
    <div className="flex flex-col gap-2.5 p-3">
      {permission === "denied" ? (
        <div className="rounded-lg bg-rose-500/10 px-3 py-2 text-[11px] leading-snug text-rose-200 ring-1 ring-rose-400/30">
          Microphone access is <b>blocked</b>.{" "}
          {desktop ? (
            <>Open Windows <i>Settings → Privacy &amp; security → Microphone</i> and turn on <b>“Let desktop apps access your microphone”</b>, then restart NeuroGrip.</>
          ) : (
            <>Click the lock icon in the address bar → allow Microphone, then press Rescan.</>
          )}
        </div>
      ) : (
        needsPermission && (
          <button
            onClick={() => void onRequestAccess()}
            className="flex items-center justify-between gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-left ring-1 ring-amber-400/30 transition hover:bg-amber-500/20"
          >
            <span className="text-[11px] leading-snug text-amber-200">Allow microphone access to see your device names</span>
            <span className="mono shrink-0 rounded-md bg-amber-400/20 px-2 py-1 text-[9px] uppercase tracking-wider text-amber-200">Allow</span>
          </button>
        )
      )}

      {missingSelected && (
        <div className="rounded-lg bg-amber-500/10 px-3 py-1.5 text-[11px] leading-snug text-amber-200 ring-1 ring-amber-400/30">
          Your previously selected microphone isn't connected — using the system default. Reconnect it and it will be picked up automatically.
        </div>
      )}

      <div ref={anchorRef}>
        <button
          disabled={locked}
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="listbox"
          aria-expanded={open}
          className={cn(
            "flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition disabled:cursor-not-allowed disabled:opacity-50",
            open ? "border-cyan-400/50 bg-cyan-500/10" : "border-white/10 bg-white/[0.03] hover:border-cyan-400/40",
          )}
        >
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-500/10 ring-1 ring-cyan-400/20">
            <MicGlyph kind={selected.kind} className="h-4 w-4 text-cyan-300" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="mono block text-[9px] uppercase tracking-[0.25em] text-slate-500">Input · {KIND_SHORT[selected.kind]}</span>
            <span className="block truncate text-sm text-slate-100" title={selected.label}>
              {selected.label}
            </span>
          </span>
          <motion.svg animate={{ rotate: open ? 180 : 0 }} className="h-4 w-4 shrink-0 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
            <path d="M6 9l6 6 6-6" />
          </motion.svg>
        </button>
      </div>

      <Popover open={open} anchorRef={anchorRef} onClose={close} minWidth={340}>
        <div className="flex items-center justify-between px-2.5 pb-1 pt-1">
          <span className="mono text-[9px] uppercase tracking-[0.25em] text-slate-500">
            Audio inputs · {realCount} detected
          </span>
          <button onClick={onRescan} className="mono text-[9px] uppercase tracking-wider text-cyan-300/80 hover:text-cyan-200">
            Rescan
          </button>
        </div>

        {groups.map((g) => (
          <div key={g.kind} className="mb-1">
            <div className="mono flex items-center justify-between px-2.5 py-1 text-[9px] uppercase tracking-[0.25em] text-slate-600">
              <span>{KIND_TITLE[g.kind]}</span>
              {g.kind !== "default" && <span>{g.items.length}</span>}
            </div>
            {g.items.map((d) => {
              const active = d.id === selected.id;
              return (
                <button
                  key={d.id || "default"}
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    onSelect(d.id);
                    close();
                  }}
                  className={cn("flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition", active ? "bg-cyan-500/15 ring-1 ring-cyan-400/30" : "hover:bg-white/[0.06]")}
                >
                  <MicGlyph kind={d.kind} className={cn("h-4 w-4 shrink-0", active ? "text-cyan-300" : "text-slate-400")} />
                  <span className={cn("min-w-0 flex-1 break-words text-xs leading-snug", active ? "text-white" : "text-slate-200")}>{d.label}</span>
                  {active && (
                    <svg className="h-4 w-4 shrink-0 text-cyan-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
                      <path d="M5 12.5l4.5 4.5L19 7.5" />
                    </svg>
                  )}
                </button>
              );
            })}
          </div>
        ))}

        {realCount === 0 && (
          <div className="mx-1 my-1 rounded-lg bg-amber-500/10 px-2.5 py-2 text-[11px] leading-snug text-amber-200 ring-1 ring-amber-400/30">
            {needsPermission ? "No microphones are visible yet — allow microphone access above." : "Windows reports no microphones. Plug one in or enable it under Settings → System → Sound → Input, then press Rescan."}
          </div>
        )}

        <div className="mx-1 mb-1 mt-1 rounded-lg bg-white/[0.03] px-2.5 py-2 text-[10px] leading-snug text-slate-400">
          <div className="mono mb-1 text-[9px] uppercase tracking-[0.2em] text-slate-500">
            {desktop ? "Desktop app" : "Browser"} · system reports {inputsFound} input{inputsFound === 1 ? "" : "s"}
          </div>
          This list is live — plugging in or pairing a device adds it automatically. A device missing here is missing in Windows: check <i>Settings → System → Sound → Input</i>. Bluetooth earbuds only expose a mic when connected as <b>“Headset / Hands-Free”</b>.
        </div>
      </Popover>

      {/* Test meter */}
      <div className="rounded-xl border border-white/5 bg-black/30 px-3 py-2">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="mono text-[9px] uppercase tracking-[0.25em] text-slate-500">{test.testing ? "Speak to test…" : "Mic test"}</span>
          <button
            onClick={test.toggle}
            disabled={locked}
            className={cn(
              "mono rounded-md px-2.5 py-1 text-[10px] uppercase tracking-wider ring-1 transition disabled:opacity-40",
              test.testing ? "bg-rose-500/10 text-rose-300 ring-rose-400/30 hover:bg-rose-500/20" : "bg-cyan-500/10 text-cyan-200 ring-cyan-400/30 hover:bg-cyan-500/20",
            )}
          >
            {test.testing ? "Stop" : "Test mic"}
          </button>
        </div>
        <div className="flex h-4 items-end gap-[3px]">
          {Array.from({ length: BARS }).map((_, i) => {
            const on = test.testing && i / BARS < test.level;
            return <span key={i} className={cn("flex-1 rounded-sm transition-all duration-75", on ? (i / BARS > 0.8 ? "bg-rose-400" : i / BARS > 0.55 ? "bg-violet-400" : "bg-cyan-400") : "bg-white/10")} style={{ height: `${25 + (i / BARS) * 75}%` }} />;
          })}
        </div>
        {test.error && <p className="mt-2 text-[11px] text-rose-300">{test.error}</p>}
      </div>
    </div>
  );
}
