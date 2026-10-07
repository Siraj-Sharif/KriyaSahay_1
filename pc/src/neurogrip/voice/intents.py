"""
neurogrip/voice/intents.py
--------------------------
Rule-based voice-intent parser.

This is a faithful port of the renderer's original ``src/services/intents.ts`` so that
``"close the hand"`` resolves to the same gesture it always did — but the decision now
happens **inside the running pipeline**, which means the voice command travels the exact
same validate → encode → transmit path as a camera-detected command.

Swap point for a real NLU model: keep the ``parse_intent(text) -> Intent`` signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Imported lazily-safe: hagrid.taxonomy has no heavy dependencies.
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY

HELP_REPLY = (
    "You can say things like close the hand, grip, open, thumbs up, OK, call, "
    "two fingers, or resume camera control."
)
RELEASE_REPLY = "Returning control to the camera. The hand will follow your gestures again."


@dataclass(frozen=True)
class Intent:
    """Parsed voice intent."""

    kind: str                      # "gesture" | "release" | "status" | "help" | "unknown"
    gesture: Optional[str] = None  # taxonomy command name when kind == "gesture"
    reply: str = ""

    @property
    def is_command(self) -> bool:
        return self.kind == "gesture" and self.gesture is not None


def _gesture_intent(gesture: str) -> Intent:
    label = gesture.replace("_", " ").title()
    return Intent(kind="gesture", gesture=gesture, reply=f"{label} — sending {gesture} to the hand.")


# Order matters: more specific phrases first.
_RULES: tuple[tuple[str, str], ...] = (
    (r"thumbs? up|thumbs-up|\blike\b|confirm|good job", "THUMBS_UP"),
    (r"\bok(ay)?\b|all right|perfect", "OK"),
    (r"\bcall\b|phone", "CALL"),
    (r"middle finger", "MIDDLE_FINGER"),
    (r"four|\b4\b", "FOUR_FINGERS"),
    (r"three|\b3\b", "THREE_FINGERS"),
    (r"\btwo\b|\b2\b|peace|victory", "TWO_FINGERS"),
    (r"pinky|little finger|small finger", "PINKY"),
    (r"index|point|one finger|\b1\b", "INDEX_FINGER"),
    (r"fist|close|clench|shut", "CLOSED_FIST"),
    (r"grip|grasp|grab|hold|pick|\brip(ley|ly)?\b|\bcrip\b|\bgrap\b|\bgreep\b", "GRIP"),
    (r"stop|halt|open|release|palm|freeze|relax", "STOP"),
)

_FUZZY_VOCAB: dict[str, str] = {
    "grip": "GRIP", "grasp": "GRIP", "fist": "CLOSED_FIST", "close": "CLOSED_FIST",
    "clench": "CLOSED_FIST", "stop": "STOP", "open": "STOP", "release": "STOP",
    "relax": "STOP", "thumbs": "THUMBS_UP", "thumbsup": "THUMBS_UP", "call": "CALL",
    "pinky": "PINKY", "index": "INDEX_FINGER", "point": "INDEX_FINGER",
    "middle": "MIDDLE_FINGER", "peace": "TWO_FINGERS", "okay": "OK",
    "status": "status", "resume": "release", "help": "help",
}


def _normalise(raw: str) -> str:
    text = (raw or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # deletion
                    current[j - 1] + 1,     # insertion
                    previous[j - 1] + (0 if ca == cb else 1),  # substitution
                )
            )
        previous = current
    return previous[-1]


def _fuzzy_intent(text: str) -> Optional[Intent]:
    """Tolerate small speech-recognition slips ("grap", "thumbsup", "fiss")."""
    best: Optional[tuple[str, int]] = None
    for token in text.split(" "):
        if len(token) < 4:
            continue
        for word, target in _FUZZY_VOCAB.items():
            max_distance = 2 if len(word) >= 7 else 1
            distance = _levenshtein(token, word)
            if distance <= max_distance and (best is None or distance < best[1]):
                best = (target, distance)
    if best is None:
        return None
    target = best[0]
    if target == "status":
        return Intent(kind="status")
    if target == "release":
        return Intent(kind="release", reply=RELEASE_REPLY)
    if target == "help":
        return Intent(kind="help", reply=HELP_REPLY)
    return _gesture_intent(target)


def parse_intent(raw: str) -> Intent:
    """Parse a transcript into an intent. Never raises; unknown input yields kind='unknown'."""
    text = _normalise(raw)
    if not text:
        return Intent(kind="unknown", reply="I didn't catch that.")

    if re.search(r"resume|auto(matic)?|follow (my|the) hand|take over|release control|camera control|mirror", text):
        return Intent(kind="release", reply=RELEASE_REPLY)
    if re.search(r"status|stat(us|is)|report|how are you|diagnos|health|system s? ?(address|adders|check|state)", text):
        return Intent(kind="status")
    if re.search(r"help|what can you|commands|gestures", text):
        return Intent(kind="help", reply=HELP_REPLY)

    # Combinations must win over their single-finger parts ("index and pinky" != "pinky").
    if re.search(r"\bindex\b.*\bpinky\b|\bpinky\b.*\bindex\b|\brock\b", text):
        return _gesture_intent("INDEX_PINKY")

    # Exact gesture id spoken, e.g. "closed fist", "four fingers".
    for command in NEUROGRIP_TAXONOMY:
        if command.replace("_", " ").lower() in text:
            return _gesture_intent(command)
    for pattern, gesture in _RULES:
        if re.search(pattern, text):
            return _gesture_intent(gesture)

    fuzzy = _fuzzy_intent(text)
    if fuzzy is not None:
        return fuzzy

    return Intent(kind="unknown", reply="Sorry, I don't have a command for that. Say help to hear what I can do.")
