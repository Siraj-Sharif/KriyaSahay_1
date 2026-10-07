import { HAND_CONNECTIONS, landmarksForPose } from "../services/gestures";
import type { FingerPose } from "../services/types";

/** Compact SVG skeleton rendering of a finger pose (same landmark format as live CV). */
export function HandPoseSVG({ pose, size = 120, active }: { pose: FingerPose; size?: number; active?: boolean }) {
  const pts = landmarksForPose(pose);
  const s = size;
  const c = active ? "#22d3ee" : "#94a3b8";
  return (
    <svg width={s} height={s} viewBox="0 0 1 1" className="overflow-visible">
      {HAND_CONNECTIONS.map(([a, b]) => (
        <line key={`${a}-${b}`} x1={pts[a].x} y1={pts[a].y} x2={pts[b].x} y2={pts[b].y} stroke={c} strokeWidth={0.018} strokeLinecap="round" opacity={0.85} />
      ))}
      {pts.map((p, i) => {
        const tip = [4, 8, 12, 16, 20].includes(i);
        return <circle key={i} cx={p.x} cy={p.y} r={tip ? 0.028 : 0.018} fill={tip ? "#fff" : c} style={{ filter: active ? "drop-shadow(0 0 3px #22d3ee)" : undefined }} />;
      })}
    </svg>
  );
}
