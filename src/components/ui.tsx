import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { cn } from "../utils/cn";

export function GlassCard({
  children,
  className,
  title,
  icon,
  right,
  strong,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  icon?: ReactNode;
  right?: ReactNode;
  strong?: boolean;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className={cn("relative overflow-hidden rounded-2xl", strong ? "glass-strong" : "glass", className)}
    >
      {title && (
        <header className="flex items-center justify-between border-b border-white/5 px-5 py-3">
          <div className="flex items-center gap-2">
            {icon && <span className="text-cyan-300/80">{icon}</span>}
            <h3 className="mono text-[11px] font-medium uppercase tracking-[0.2em] text-slate-400">{title}</h3>
          </div>
          {right}
        </header>
      )}
      {children}
    </motion.section>
  );
}

export function StatusDot({ status, label }: { status: "ok" | "warn" | "off" | "busy"; label?: string }) {
  const color =
    status === "ok" ? "bg-emerald-400" : status === "warn" ? "bg-amber-400" : status === "busy" ? "bg-cyan-400" : "bg-rose-500";
  return (
    <span className="inline-flex items-center gap-2">
      <span className="relative flex h-2 w-2">
        {(status === "ok" || status === "busy") && (
          <span className={cn("absolute inline-flex h-full w-full rounded-full opacity-70 pulse-ring", color)} />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", color)} />
      </span>
      {label && <span className="mono text-[11px] uppercase tracking-wider text-slate-300">{label}</span>}
    </span>
  );
}

export function Metric({
  label,
  value,
  unit,
  accent,
  className,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  accent?: boolean;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <span className="mono text-[10px] uppercase tracking-[0.2em] text-slate-500">{label}</span>
      <span className={cn("mono text-lg font-semibold leading-none", accent ? "text-cyan-300 text-glow" : "text-slate-100")}>
        {value}
        {unit && <span className="ml-1 text-xs font-normal text-slate-500">{unit}</span>}
      </span>
    </div>
  );
}

export function Corner({ className }: { className?: string }) {
  return (
    <>
      <span className={cn("pointer-events-none absolute left-2 top-2 h-3 w-3 border-l border-t border-cyan-400/50", className)} />
      <span className={cn("pointer-events-none absolute right-2 top-2 h-3 w-3 border-r border-t border-cyan-400/50", className)} />
      <span className={cn("pointer-events-none absolute bottom-2 left-2 h-3 w-3 border-b border-l border-cyan-400/50", className)} />
      <span className={cn("pointer-events-none absolute bottom-2 right-2 h-3 w-3 border-b border-r border-cyan-400/50", className)} />
    </>
  );
}

/* Minimal inline icons (no icon lib needed) */
export const Icon = {
  Camera: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M4 7h3l2-2h6l2 2h3v12H4z" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  ),
  Hand: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M8 12V5a1.5 1.5 0 0 1 3 0v6M11 11V4a1.5 1.5 0 0 1 3 0v7M14 11V5.5a1.5 1.5 0 0 1 3 0V13M8 12V9.5a1.5 1.5 0 0 0-3 0V15c0 4 3 7 7 7s7-3 7-7v-2" />
    </svg>
  ),
  Chip: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
      <path d="M9 2v4M12 2v4M15 2v4M9 18v4M12 18v4M15 18v4M2 9h4M2 12h4M2 15h4M18 9h4M18 12h4M18 15h4" />
    </svg>
  ),
  Mic: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  ),
  Grid: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  ),
  Chart: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M3 20h18M5 16l4-6 4 3 6-8" />
    </svg>
  ),
  Expand: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
    </svg>
  ),
  Compress: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />
    </svg>
  ),
  Serial: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M4 12h4l2-5 3 10 2-5h5" />
    </svg>
  ),
  Eye: (p: { className?: string }) => (
    <svg className={p.className ?? "h-4 w-4"} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}>
      <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
};
