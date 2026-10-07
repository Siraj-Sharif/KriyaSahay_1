import type { FingerPose, Gesture, Landmark } from "./types";

export interface GestureDefinition {
  id: Gesture;
  label: string;
  description: string;
  pose: FingerPose; // thumb, index, middle, ring, pinky (0 open → 1 curled)
  action: string;
  emoji: string;
}

export const GESTURE_DEFS: Record<Gesture, GestureDefinition> = {
  CALL: {
    id: "CALL",
    label: "Call",
    description: "Thumb and pinky extended, remaining fingers curled — the classic 'call me' sign.",
    pose: [0, 1, 1, 1, 0, 0],
    action: "Trigger communication mode",
    emoji: "🤙",
  },
  CLOSED_FIST: {
    id: "CLOSED_FIST",
    label: "Closed Fist",
    description: "All five fingers fully curled into the palm.",
    pose: [1, 1, 1, 1, 1, 0],
    action: "Full actuator close",
    emoji: "✊",
  },
  FOUR_FINGERS: {
    id: "FOUR_FINGERS",
    label: "Four Fingers",
    description: "Index through pinky extended with thumb tucked.",
    pose: [1, 0, 0, 0, 0, 0],
    action: "Numeric input: 4",
    emoji: "🖖",
  },
  GRIP: {
    id: "GRIP",
    label: "Grip",
    description: "All fingers half closed, with the index fingertip and thumb tip touching — adaptive power grasp.",
    pose: [0, 0.65, 0.6, 0.6, 0.6, 1],
    action: "Adaptive object grasp",
    emoji: "🫳",
  },
  THUMBS_UP: {
    id: "THUMBS_UP",
    label: "Thumbs Up",
    description: "Thumb extended upward with all other fingers curled.",
    pose: [0, 1, 1, 1, 1, 0],
    action: "Confirm / acknowledge",
    emoji: "👍",
  },
  PINKY: {
    id: "PINKY",
    label: "Pinky",
    description: "Only the little finger extended.",
    pose: [1, 1, 1, 1, 0, 0],
    action: "Auxiliary channel select",
    emoji: "🤙",
  },
  MIDDLE_FINGER: {
    id: "MIDDLE_FINGER",
    label: "Middle Finger",
    description: "Only the middle finger extended; others curled.",
    pose: [1, 1, 0, 1, 1, 0],
    action: "Reserved diagnostic pose",
    emoji: "🖕",
  },
  OK: {
    id: "OK",
    label: "OK",
    description: "Thumb and index tips touching to form a ring; middle, ring and pinky fully open.",
    pose: [0, 0.65, 0, 0, 0, 1],
    action: "Precision confirm",
    emoji: "👌",
  },
  INDEX_FINGER: {
    id: "INDEX_FINGER",
    label: "Index Finger",
    description: "Index finger pointing with the remaining fingers curled.",
    pose: [1, 0, 1, 1, 1, 0],
    action: "Pointer / select target",
    emoji: "☝️",
  },
  INDEX_PINKY: {
    id: "INDEX_PINKY",
    label: "Index + Pinky",
    description: "Index and pinky extended, thumb and middle fingers curled.",
    pose: [1, 0, 1, 1, 0, 0],
    action: "Dual channel mode",
    emoji: "🤘",
  },
  STOP: {
    id: "STOP",
    label: "Stop",
    description: "Open palm facing the camera with all fingers extended.",
    pose: [0, 0, 0, 0, 0, 0],
    action: "Emergency halt / release",
    emoji: "✋",
  },
  TWO_FINGERS: {
    id: "TWO_FINGERS",
    label: "Two Fingers",
    description: "Index and middle fingers extended in a V shape.",
    pose: [1, 0, 0, 1, 1, 0],
    action: "Numeric input: 2",
    emoji: "✌️",
  },
  THREE_FINGERS: {
    id: "THREE_FINGERS",
    label: "Three Fingers",
    description: "Index, middle, and ring fingers extended.",
    pose: [1, 0, 0, 0, 1, 0],
    action: "Numeric input: 3",
    emoji: "🤟",
  },
};

export const GESTURE_LIST = Object.values(GESTURE_DEFS);

/* ------------------------------------------------------------------ */
/* Synthetic MediaPipe-style landmark generation (21 points)           */
/* ------------------------------------------------------------------ */

const FINGER_BASES: [number, number][] = [
  [0.42, 0.62], // thumb cmc
  [0.44, 0.48], // index mcp
  [0.5, 0.46], // middle mcp
  [0.56, 0.47], // ring mcp
  [0.62, 0.5], // pinky mcp
];
const FINGER_DIRS: [number, number][] = [
  [-0.16, -0.12],
  [-0.03, -0.22],
  [0.0, -0.24],
  [0.03, -0.22],
  [0.07, -0.18],
];

export function landmarksForPose(pose: FingerPose, jitter = 0): Landmark[] {
  const r = () => (Math.random() - 0.5) * jitter;
  const pts: Landmark[] = [{ x: 0.5 + r(), y: 0.78 + r(), z: 0 }]; // wrist
  for (let f = 0; f < 5; f++) {
    const [bx, by] = FINGER_BASES[f];
    const [dx, dy] = FINGER_DIRS[f];
    const curl = pose[f];
    let x = bx;
    let y = by;
    let angle = Math.atan2(dy, dx);
    const segLen = Math.hypot(dx, dy) / 3;
    pts.push({ x: x + r(), y: y + r(), z: 0 });
    for (let s = 0; s < 3; s++) {
      angle += curl * 1.05 * (f === 0 ? 0.8 : 1);
      x += Math.cos(angle) * segLen * (1 - curl * 0.15);
      y += Math.sin(angle) * segLen * (1 - curl * 0.15);
      pts.push({ x: x + r(), y: y + r(), z: -s * 0.01 });
    }
  }
  // Thumb opposition: pull thumb joints toward the index fingertip so the tips touch
  const opp = pose[5];
  if (opp > 0) {
    const target = pts[8];
    for (let i = 2; i <= 4; i++) {
      const k = opp * ((i - 1) / 3) * (i === 4 ? 1 : 0.85);
      const tx = target.x - 0.012;
      const ty = target.y + 0.012;
      pts[i] = { x: pts[i].x + (tx - pts[i].x) * k, y: pts[i].y + (ty - pts[i].y) * k, z: pts[i].z };
    }
  }
  return pts;
}

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export const DEVICE_ID = "NG1";
export const encodeCommand = (g: Gesture) => `${DEVICE_ID}|${g}`;
