import { Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useMemo } from "react";

interface HistoryPoint {
  t: number;
  confidence: number;
  fps: number;
}

export function ConfidenceChart({ data }: { data: HistoryPoint[] }) {
  const series = useMemo(() => data.map((d, i) => ({ ...d, i, time: new Date(d.t).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" }) })), [data]);
  const avg = series.length ? Math.round(series.reduce((a, b) => a + b.confidence, 0) / series.length) : 0;
  const last = series.at(-1)?.confidence ?? 0;
  const min = series.length ? Math.min(...series.map((s) => s.confidence)) : 0;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-3 px-5 pt-4 sm:w-1/2">
        {[
          ["Current", `${last}%`, "text-cyan-300"],
          ["Average", `${avg}%`, "text-slate-100"],
          ["Minimum", `${min}%`, "text-violet-300"],
        ].map(([k, v, c]) => (
          <div key={k} className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
            <div className="mono text-[9px] uppercase tracking-[0.25em] text-slate-500">{k}</div>
            <div className={`mono text-xl font-semibold ${c}`}>{v}</div>
          </div>
        ))}
      </div>
      <div className="h-72 w-full px-2 pb-3">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={series} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="confFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.45} />
                <stop offset="60%" stopColor="#8b5cf6" stopOpacity={0.12} />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="confStroke" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#22d3ee" />
                <stop offset="100%" stopColor="#a78bfa" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 6" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} minTickGap={40} />
            <YAxis domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              cursor={{ stroke: "rgba(34,211,238,0.3)", strokeWidth: 1 }}
              contentStyle={{ background: "rgba(5,7,12,0.92)", border: "1px solid rgba(34,211,238,0.25)", borderRadius: 10, fontFamily: "JetBrains Mono", fontSize: 11 }}
              labelStyle={{ color: "#94a3b8" }}
              itemStyle={{ color: "#22d3ee" }}
              formatter={(v, name) => [name === "fps" ? `${v}` : `${v}%`, name === "fps" ? "FPS" : "Confidence"]}
            />
            <Area type="monotone" dataKey="confidence" stroke="url(#confStroke)" strokeWidth={2.2} fill="url(#confFill)" isAnimationActive={false} filter="url(#glow)" dot={false} activeDot={{ r: 4, fill: "#fff", stroke: "#22d3ee" }} />
            <Line type="monotone" dataKey="fps" stroke="rgba(148,163,184,0.35)" strokeDasharray="4 4" strokeWidth={1} dot={false} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
