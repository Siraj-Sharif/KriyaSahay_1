"""
neurogrip/features/finger_state.py
──────────────────────────────────
Per-finger state representation and rule-based state determination.
Independent from MediaPipe — operates purely on internal HandLandmarks representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

from neurogrip.config.settings import FeaturesConfig
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark


class Finger(str, Enum):
    THUMB = "THUMB"
    INDEX = "INDEX"
    MIDDLE = "MIDDLE"
    RING = "RING"
    PINKY = "PINKY"


class FingerState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FingerStateVector:
    """Dataclass storing the state of all 5 individual fingers and palm facing status."""
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    is_palm_facing: bool = True

    def as_dict(self) -> dict[Finger, FingerState]:
        return {
            Finger.THUMB: self.thumb,
            Finger.INDEX: self.index,
            Finger.MIDDLE: self.middle,
            Finger.RING: self.ring,
            Finger.PINKY: self.pinky,
        }

    def __getitem__(self, finger: Finger | str) -> FingerState:
        f_name = finger.value if isinstance(finger, Finger) else str(finger).upper()
        if f_name == "THUMB":
            return self.thumb
        elif f_name == "INDEX":
            return self.index
        elif f_name == "MIDDLE":
            return self.middle
        elif f_name == "RING":
            return self.ring
        elif f_name == "PINKY":
            return self.pinky
        raise KeyError(f"Invalid finger name '{finger}'.")


class FingerStateDetector:
    """
    Determines per-finger states (OPEN / CLOSED / UNKNOWN) from normalized HandLandmarks.
    """

    # Finger landmark index mappings (MCP, PIP/IP, TIP)
    FINGER_LANDMARKS = {
        Finger.THUMB: {"mcp": 2, "pip": 3, "tip": 4},
        Finger.INDEX: {"mcp": 5, "pip": 6, "tip": 8},
        Finger.MIDDLE: {"mcp": 9, "pip": 10, "tip": 12},
        Finger.RING: {"mcp": 13, "pip": 14, "tip": 16},
        Finger.PINKY: {"mcp": 17, "pip": 18, "tip": 20},
    }

    def __init__(self, config: Optional[FeaturesConfig] = None) -> None:
        self.config = config or FeaturesConfig()

    def detect(self, hand_landmarks: HandLandmarks) -> FingerStateVector:
        """
        Detect individual finger states from a 21-point HandLandmarks object.
        """
        lms = hand_landmarks.landmarks
        wrist = lms[self.config.wrist_landmark]
        middle_mcp = lms[self.config.span_landmark]

        # Calculate reference hand span (wrist to middle MCP distance)
        span = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        if span < 1e-6:
            span = 1.0  # Fallback to prevent zero division

        # Normalize landmarks relative to wrist and span, applying handedness mirroring
        is_left = hand_landmarks.handedness == Handedness.LEFT
        norm_lms = []
        for lm in lms:
            dx = (lm.x - wrist.x) / span
            dy = (lm.y - wrist.y) / span
            dz = (lm.z - wrist.z) / span if hasattr(wrist, "z") else lm.z
            if is_left:
                dx = -dx  # Mirror left hand on x-axis
            norm_lms.append((dx, dy, dz))

        # Check palm orientation: reject strongly sideways or back-facing hands
        if not self.is_palm_facing_camera(norm_lms):
            return FingerStateVector(
                thumb=FingerState.UNKNOWN,
                index=FingerState.UNKNOWN,
                middle=FingerState.UNKNOWN,
                ring=FingerState.UNKNOWN,
                pinky=FingerState.UNKNOWN,
                is_palm_facing=False,
            )

        states: dict[Finger, FingerState] = {}
        for finger in Finger:
            states[finger] = self._detect_single_finger(finger, norm_lms)

        return FingerStateVector(
            thumb=states[Finger.THUMB],
            index=states[Finger.INDEX],
            middle=states[Finger.MIDDLE],
            ring=states[Finger.RING],
            pinky=states[Finger.PINKY],
            is_palm_facing=True,
        )

    @staticmethod
    def is_palm_facing_camera(norm_lms: list[tuple[float, float, float]]) -> bool:
        """Check if palm faces camera (rejects strongly sideways 90-degree yaw poses)."""
        if len(norm_lms) < 18:
            return True
        v_idx = norm_lms[5]
        v_pky = norm_lms[17]
        n_z = v_idx[0] * v_pky[1] - v_idx[1] * v_pky[0]
        n_x = v_idx[1] * v_pky[2] - v_idx[2] * v_pky[1]
        n_y = v_idx[2] * v_pky[0] - v_idx[0] * v_pky[2]
        n_mag = math.sqrt(n_x**2 + n_y**2 + n_z**2)
        if n_mag < 1e-4:
            return False  # Degenerate edge-on sideways pose
        return abs(n_z / n_mag) >= 0.15

    def _detect_single_finger(
        self, finger: Finger, norm_lms: list[tuple[float, float, float]]
    ) -> FingerState:
        mapping = self.FINGER_LANDMARKS[finger]
        mcp_idx, pip_idx, tip_idx = mapping["mcp"], mapping["pip"], mapping["tip"]

        tip = norm_lms[tip_idx]
        mcp = norm_lms[mcp_idx]
        pip = norm_lms[pip_idx]

        # 3D Tip-to-MCP extension ratio in normalized space
        ext_ratio = math.sqrt((tip[0] - mcp[0])**2 + (tip[1] - mcp[1])**2 + (tip[2] - mcp[2])**2)
        # 3D Tip-to-wrist distance
        tip_wrist_dist = math.sqrt(tip[0]**2 + tip[1]**2 + tip[2]**2)

        # Joint cosine calculation at proximal joint in 3D
        if finger == Finger.THUMB:
            a = norm_lms[1]
            b = norm_lms[2]
            c = norm_lms[3]
        else:
            a = norm_lms[mcp_idx]
            b = norm_lms[pip_idx]
            c = norm_lms[tip_idx]

        cos_val = self._compute_cosine(a, b, c)

        # THUMB special lateral extension check
        if finger == Finger.THUMB:
            thumb_lateral_x = tip[0]
            if (
                thumb_lateral_x > self.config.thumb_lateral_threshold
                and ext_ratio > self.config.open_extension_threshold
            ):
                return FingerState.OPEN
            elif ext_ratio >= self.config.open_extension_threshold and cos_val >= self.config.open_cosine_threshold:
                return FingerState.OPEN
            elif ext_ratio <= self.config.closed_extension_threshold or cos_val <= self.config.closed_cosine_threshold:
                return FingerState.CLOSED
            return FingerState.UNKNOWN

        # Per-finger open distance threshold (pinky is naturally shorter)
        open_dist_thresh = self.config.open_distance_threshold
        if finger == Finger.PINKY:
            open_dist_thresh = max(0.28, open_dist_thresh - 0.08)

        # General non-thumb finger logic
        is_open = (
            ext_ratio >= self.config.open_extension_threshold
            and tip_wrist_dist >= open_dist_thresh
            and cos_val >= self.config.open_cosine_threshold
        )

        is_closed = (
            ext_ratio <= self.config.closed_extension_threshold
            or cos_val <= self.config.closed_cosine_threshold
        )

        if is_open and not is_closed:
            return FingerState.OPEN
        elif is_closed and not is_open:
            return FingerState.CLOSED

        return FingerState.UNKNOWN

    @staticmethod
    def _compute_cosine(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
    ) -> float:
        v1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        return max(-1.0, min(1.0, dot / (mag1 * mag2)))
