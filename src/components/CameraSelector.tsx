import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { Popover } from "./Popover";
import { useSystem } from "../services/SystemContext";
import type { CameraDevice } from "../hooks/useCameraDevices";
import { cn } from "../utils/cn";


const KIND_LABEL = { "built-in": "Built-in", external: "External" } as const;

function KindIcon({ kind, className }: { kind: CameraDevice["kind"]; className?: string }) {
  return kind === "built-in" ? (
    // laptop
    <svg className={className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      <rect x="5" y="5" width="14" height="10" rx="1.5" />
      <path d="M2.5 19h19" />
    </svg>
  ) : (
    // usb webcam
    <svg className={className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="9" r="5.5" />
      <circle cx="12" cy="9" r="2" />
      <path d="M12 14.5V19M8 21h8M10 19h4" />
    </svg>
  );
}

/** Camera source dropdown: lists built-in and external webcams and switches the active feed. */
export function CameraSelector() {
  const { cameraDevices, selectedCamera, selectCamera, refreshCameras, backendControlled } = useSystem();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (open) void refreshCameras();
  }, [open, refreshCameras]);

  const groups = (["built-in", "external"] as const).map((k) => ({ kind: k, items: cameraDevices.filter((d) => d.kind === k) })).filter((g) => g.items.length);

  return (
    <div ref={rootRef} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Select camera source"
        className={cn(
          "flex h-8 max-w-[210px] items-center gap-2 rounded-lg border px-2.5 transition",
          open ? "border-cyan-400/50 bg-cyan-500/10" : "border-white/10 bg-white/[0.03] hover:border-cyan-400/40",
        )}
      >
        <KindIcon kind={selectedCamera.kind} className="h-4 w-4 shrink-0 text-cyan-300" />
        <span className="flex min-w-0 flex-col items-start leading-none">
          <span className="mono text-[8px] uppercase tracking-[0.25em] text-slate-500">Source · {KIND_LABEL[selectedCamera.kind]}</span>
          <span className="mt-0.5 w-full truncate text-left text-[11px] text-slate-200">{selectedCamera.label}</span>
        </span>
        <motion.svg animate={{ rotate: open ? 180 : 0 }} className="h-3 w-3 shrink-0 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
          <path d="M6 9l6 6 6-6" />
        </motion.svg>
      </button>

      <Popover open={open} anchorRef={rootRef} onClose={close} align="right" minWidth={288}>
        <div className="flex items-center justify-between px-2.5 pb-1.5 pt-1">
          <span className="mono text-[9px] uppercase tracking-[0.25em] text-slate-500">Camera source</span>
          <button onClick={() => void refreshCameras()} className="mono text-[9px] uppercase tracking-wider text-cyan-300/80 hover:text-cyan-200">
            Rescan
          </button>
        </div>
        {groups.map((g) => (
          <div key={g.kind} className="mb-1">
            <div className="mono px-2.5 py-1 text-[9px] uppercase tracking-[0.25em] text-slate-600">{g.kind === "built-in" ? "Built-in camera" : "External camera"}</div>
            {g.items.map((d) => {
              const active = d.id === selectedCamera.id;
              return (
                <button
                  key={d.id}
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    selectCamera(d.id);
                    setOpen(false);
                  }}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition",
                    active ? "bg-cyan-500/15 ring-1 ring-cyan-400/30" : "hover:bg-white/[0.06]",
                  )}
                >
                  <KindIcon kind={d.kind} className={cn("h-4 w-4 shrink-0", active ? "text-cyan-300" : "text-slate-400")} />
                  <span className="min-w-0 flex-1">
                    <span className={cn("block truncate text-xs", active ? "text-white" : "text-slate-200")}>{d.label}</span>
                    <span className="mono block truncate text-[9px] text-slate-500">{d.simulated ? "no device access · simulated feed" : `ID ${d.id.slice(0, 10)}…`}</span>
                  </span>
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
        <p className="px-2.5 pb-1.5 pt-1 text-[10px] leading-snug text-slate-600">
          {backendControlled
            ? "Indices come from Python OpenCV discovery. Selection changes the camera owned by the live pipeline."
            : "Browser camera names are not connected to the simulated CV preview and will not be used for capture."}
        </p>
      </Popover>
    </div>
  );
}

