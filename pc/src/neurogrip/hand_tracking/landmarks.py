"""
neurogrip/hand_tracking/landmarks.py
────────────────────────────────────
Internal landmark representations.
Decouples MediaPipe landmark types from downstream CV and recognition modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Handedness(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, label: str) -> "Handedness":
        if not label:
            return cls.UNKNOWN
        upper = label.strip().upper()
        if "LEFT" in upper:
            return cls.LEFT
        elif "RIGHT" in upper:
            return cls.RIGHT
        return cls.UNKNOWN


@dataclass(frozen=True)
class NormalizedLandmark:
    """
    Single hand landmark in 3D space.
    x, y: normalized coordinates in [0.0, 1.0] relative to image dimensions.
    z: relative depth estimate from wrist.
    """
    x: float
    y: float
    z: float


@dataclass
class HandLandmarks:
    """
    21-landmark set for a single hand detection.
    """
    landmarks: list[NormalizedLandmark]
    handedness: Handedness = Handedness.UNKNOWN
    score: float = 1.0
    world_landmarks: Optional[list[NormalizedLandmark]] = None

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(
                f"HandLandmarks requires exactly 21 landmarks, got {len(self.landmarks)}."
            )


@dataclass
class DetectionResult:
    """
    Output of a frame hand tracking operation.
    """
    hands: list[HandLandmarks] = field(default_factory=list)
    num_hands: int = 0
    timestamp_ms: int = 0

    @property
    def has_hand(self) -> bool:
        """True if exactly 1 hand is detected."""
        return self.num_hands == 1

    @property
    def is_ambiguous(self) -> bool:
        """True if 2 or more hands are detected (requires safe state / NO_COMMAND)."""
        return self.num_hands > 1

    @property
    def is_empty(self) -> bool:
        """True if 0 hands are detected."""
        return self.num_hands == 0

    @property
    def primary_hand(self) -> Optional[HandLandmarks]:
        """Return the single hand if exactly 1 hand is present, else None."""
        return self.hands[0] if self.num_hands == 1 else None
