import type { Gesture } from "./types";
import { GESTURE_DEFS } from "./gestures";

export type Intent =
  | { kind: "gesture"; gesture: Gesture; reply: string }
  | { kind: "release"; reply: string }
  | { kind: "status" }
  | { kind: "help"; reply: string }
  | { kind: "unknown"; reply: string };

/**
 * Rule-based intent parser. Order matters: more specific phrases first.
 * This is the swap point for an LLM / NLU model later — keep the signature
 * `parseIntent(text) => Intent` and the rest of the app stays unchanged.
 */
const RULES: { test: RegExp; gesture: Gesture }[] = [
  { test: /\bindex\b.*\bpinky\b|\bpinky\b.*\bindex\b|rock/, gesture: "INDEX_PINKY" },
  { test: /middle finger/, gesture: "MIDDLE_FINGER" },
  { test: /thumbs? up|thumbs-up|like|confirm|good job/, gesture: "THUMBS_UP" },
  { test: /\bok(ay)?\b|all right|perfect/, gesture: "OK" },
  { test: /\bcall\b|phone/, gesture: "CALL" },
  { test: /four|\b4\b/, gesture: "FOUR_FINGERS" },
  { test: /three|\b3\b/, gesture: "THREE_FINGERS" },
  { test: /two|\b2\b|peace|victory/, gesture: "TWO_FINGERS" },
  { test: /pinky|little finger|small finger/, gesture: "PINKY" },
  { test: /index|point|one finger|\b1\b/, gesture: "INDEX_FINGER" },
  { test: /fist|close|clench|shut/, gesture: "CLOSED_FIST" },
  { test: /grip|grasp|grab|hold|pick|\brip(ley|ly)?\b|\bcrip\b|\bgrap\b|\bgreep\b/, gesture: "GRIP" },
  { test: /stop|halt|open|release|palm|freeze|relax/, gesture: "STOP" },
];

export function parseIntent(raw: string): Intent {
  const text = raw.toLowerCase().replace(/[^a-z0-9\s-]/g, " ").replace(/\s+/g, " ").trim();
  if (!text) return { kind: "unknown", reply: "I didn't catch that." };

  if (/resume|auto(matic)?|follow (my|the) hand|take over|release control|camera control|mirror/.test(text)) {
    return { kind: "release", reply: "Returning control to the camera. The hand will follow your gestures again." };
  }
  if (/status|stat(us|is)|report|how are you|diagnos|health|system s? ?(address|adders|check|state)/.test(text)) return { kind: "status" };
  if (/help|what can you|commands|gestures/.test(text)) {
    return { kind: "help", reply: "You can say things like close the hand, grip, open, thumbs up, OK, call, two fingers, or resume camera control." };
  }

  // combinations must win over their single-finger parts ("index and pinky" ≠ "pinky")
  if (/\bindex\b.*\bpinky\b|\bpinky\b.*\bindex\b|\brock\b/.test(text)) return gestureIntent("INDEX_PINKY");

  // exact gesture id spoken, e.g. "closed fist", "four fingers"
  for (const id of Object.keys(GESTURE_DEFS) as Gesture[]) {
    if (text.includes(id.toLowerCase().replace(/_/g, " "))) return gestureIntent(id);
  }
  for (const r of RULES) if (r.test.test(text)) return gestureIntent(r.gesture);

  // Last resort: tolerate small speech-recognition slips ("grap", "thumbsup", "fiss"…)
  const fuzzy = fuzzyIntent(text);
  if (fuzzy) return fuzzy;

  return { kind: "unknown", reply: "Sorry, I don't have a command for that. Say help to hear what I can do." };
}

function gestureIntent(gesture: Gesture): Intent {
  return { kind: "gesture", gesture, reply: `${GESTURE_DEFS[gesture].label} — sending ${gesture} to the hand.` };
}

/* ------------------------------------------------------------------ */
/* Fuzzy matching                                                      */
/* ------------------------------------------------------------------ */

type FuzzyTarget = Gesture | "status" | "release" | "help";

const FUZZY_VOCAB: Record<string, FuzzyTarget> = {
  grip: "GRIP",
  grasp: "GRIP",
  fist: "CLOSED_FIST",
  close: "CLOSED_FIST",
  clench: "CLOSED_FIST",
  stop: "STOP",
  open: "STOP",
  release: "STOP",
  relax: "STOP",
  thumbs: "THUMBS_UP",
  thumbsup: "THUMBS_UP",
  call: "CALL",
  pinky: "PINKY",
  index: "INDEX_FINGER",
  point: "INDEX_FINGER",
  middle: "MIDDLE_FINGER",
  peace: "TWO_FINGERS",
  okay: "OK",
  status: "status",
  resume: "release",
  help: "help",
};

function levenshtein(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)] as number[]);
  for (let j = 1; j <= b.length; j++) dp[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return dp[a.length][b.length];
}

function fuzzyIntent(text: string): Intent | null {
  let best: { target: FuzzyTarget; dist: number } | null = null;
  for (const token of text.split(" ")) {
    if (token.length < 4) continue;
    for (const [word, target] of Object.entries(FUZZY_VOCAB)) {
      const max = word.length >= 7 ? 2 : 1;
      const d = levenshtein(token, word);
      if (d <= max && (!best || d < best.dist)) best = { target, dist: d };
    }
  }
  if (!best) return null;
  if (best.target === "status") return { kind: "status" };
  if (best.target === "release") return { kind: "release", reply: "Returning control to the camera. The hand will follow your gestures again." };
  if (best.target === "help") return { kind: "help", reply: "You can say things like close the hand, grip, open, thumbs up, OK, call, two fingers, or resume camera control." };
  return gestureIntent(best.target);
}
