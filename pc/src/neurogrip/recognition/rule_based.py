"""
neurogrip/recognition/rule_based.py
───────────────────────────────────
Rule-Based Diagnostic and Fallback Recognizer.
Uses FingerStateDetector and geometric features to map hand configurations to the 13 official ML commands.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, GrabCalibrationConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import (
    Finger,
    FingerState,
    FingerStateDetector,
    FingerStateVector,
)
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult


class RuleBasedRecognizer(GestureRecognizer):
    """
    Diagnostic and fallback gesture recognizer.
    Evaluates finger states and landmark geometry against rules for the 13 official ML commands.
    """

    THUMB_SPREAD_LOWER_THRESH: float = 0.35
    THUMB_SPREAD_UPPER_THRESH: float = 0.45

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.default()
        self.finger_detector = FingerStateDetector(self.config.features)
        self.feature_extractor = FeatureExtractor(self.config.features)
        self.grab_cfg: GrabCalibrationConfig = self.config.grab_calibration

    @property
    def is_ready(self) -> bool:
        return True

    def get_thumb_spread_metric(self, landmarks: HandLandmarks) -> float:
        """
        Calculate normalized 3D Thumb-Tip to Index-MCP distance metric.
        Used for robust, mutually-exclusive STOP vs FOUR_FINGERS decision hierarchy.
        """
        if landmarks is None or len(landmarks.landmarks) < 21:
            return 0.0

        lms = landmarks.landmarks
        w_idx = self.config.features.wrist_landmark
        s_idx = self.config.features.span_landmark

        wrist = lms[w_idx]
        middle_mcp = lms[s_idx]

        span = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        if span < 1e-6:
            span = 1.0

        is_left = landmarks.handedness == Handedness.LEFT

        th_tip = lms[4]
        idx_mcp = lms[5]

        dx_th = (th_tip.x - wrist.x) / span
        dy_th = (th_tip.y - wrist.y) / span
        dz_th = (th_tip.z - wrist.z) / span if hasattr(wrist, "z") else th_tip.z
        if is_left:
            dx_th = -dx_th

        dx_idx = (idx_mcp.x - wrist.x) / span
        dy_idx = (idx_mcp.y - wrist.y) / span
        dz_idx = (idx_mcp.z - wrist.z) / span if hasattr(wrist, "z") else idx_mcp.z
        if is_left:
            dx_idx = -dx_idx

        return math.sqrt((dx_th - dx_idx)**2 + (dy_th - dy_idx)**2 + (dz_th - dz_idx)**2)

    def predict(
        self,
        features: np.ndarray,
        hand_landmarks: Optional[HandLandmarks] = None,
        is_stop_armed: bool = False,
        raw_frame: Optional[np.ndarray] = None,
    ) -> RecognitionResult:
        """
        Predict a gesture command using rule-based classification logic.
        """
        if hand_landmarks is None:
            # Cannot run rule-based finger state detector without landmark coordinates
            return RecognitionResult(
                label="UNKNOWN",
                confidence=0.0,
                recognizer_type="rule_based",
            )

        finger_states = self.finger_detector.detect(hand_landmarks)
        label, conf, reason = self._classify_finger_states(
            finger_states, hand_landmarks, features, is_stop_armed
        )

        all_scores = {cmd.value: (1.0 if cmd.value == label else 0.0) for cmd in NeuroGripCommand}
        all_scores["UNKNOWN"] = 1.0 if label == "UNKNOWN" else 0.0

        return RecognitionResult(
            label=label,
            confidence=conf,
            all_scores=all_scores,
            recognizer_type="rule_based",
            reason=reason,
        )

    def _classify_finger_states(
        self,
        fs: FingerStateVector,
        landmarks: HandLandmarks,
        features: np.ndarray,
        is_stop_armed: bool,
    ) -> tuple[str, float, Optional[str]]:
        """Classify finger states into one of the 13 locked commands or UNKNOWN."""
        t, i, m, r, p = fs.thumb, fs.index, fs.middle, fs.ring, fs.pinky

        # Reject non-palm-facing / edge-on poses at live recognition boundary
        if not fs.is_palm_facing:
            return "UNKNOWN", 0.0, "NOT_PALM_FACING"

        # Reject all-UNKNOWN finger states
        if t == FingerState.UNKNOWN and i == FingerState.UNKNOWN and m == FingerState.UNKNOWN and r == FingerState.UNKNOWN and p == FingerState.UNKNOWN:
            return "UNKNOWN", 0.0, "ALL_FINGERS_UNKNOWN"

        # Count open, closed, unknown fingers
        num_open = sum(1 for s in (t, i, m, r, p) if s == FingerState.OPEN)
        num_closed = sum(1 for s in (t, i, m, r, p) if s == FingerState.CLOSED)

        # 1. STOP & FOUR_FINGERS handling with strict dead-band partitioning
        if i == FingerState.OPEN and m == FingerState.OPEN and r == FingerState.OPEN and p == FingerState.OPEN:
            spread_metric = self.get_thumb_spread_metric(landmarks)

            # Below lower threshold -> FOUR_FINGERS (Thumb is narrow/relaxed)
            if spread_metric < self.THUMB_SPREAD_LOWER_THRESH:
                return NeuroGripCommand.FOUR_FINGERS.value, 1.0, "FOUR_NON_THUMB_OPEN_THUMB_NARROW"

            # Above upper threshold AND thumb is NOT CLOSED -> STOP
            if spread_metric >= self.THUMB_SPREAD_UPPER_THRESH and t != FingerState.CLOSED:
                return NeuroGripCommand.STOP.value, 1.0, "FOUR_NON_THUMB_OPEN_THUMB_WIDE_OPEN"

            # Above upper threshold but thumb is CLOSED -> FOUR_FINGERS
            if spread_metric >= self.THUMB_SPREAD_UPPER_THRESH and t == FingerState.CLOSED:
                return NeuroGripCommand.FOUR_FINGERS.value, 1.0, "FOUR_NON_THUMB_OPEN_THUMB_NOT_OPEN"

            # Ambiguity region (0.35 <= spread_metric < 0.45) -> UNKNOWN
            return "UNKNOWN", 0.0, "FOUR_NON_THUMB_OPEN_THUMB_AMBIGUOUS"

        # 2. THUMB_ONLY: Thumb OPEN, all 4 main fingers CLOSED
        if t == FingerState.OPEN and i == FingerState.CLOSED and m == FingerState.CLOSED and r == FingerState.CLOSED and p == FingerState.CLOSED:
            return NeuroGripCommand.THUMB_ONLY.value, 1.0, None

        # 3. CLOSE: All 4 main fingers closed (and thumb not OPEN)
        if i == FingerState.CLOSED and m == FingerState.CLOSED and r == FingerState.CLOSED and p == FingerState.CLOSED:
            if self._is_grab_claw_pose(landmarks, features):
                return NeuroGripCommand.GRAB.value, 0.90, "FUNCTIONAL_CLAW_POSE"
            return NeuroGripCommand.CLOSE.value, 1.0, None

        # 4. GRAB: Functional partial-curl claw posture (calibration configurable)
        if self._is_grab_claw_pose(landmarks, features):
            return NeuroGripCommand.GRAB.value, 0.90, "FUNCTIONAL_CLAW_POSE"

        # 5. THREE_FINGER: Index, Middle, Ring OPEN, Pinky NOT OPEN (CLOSED or UNKNOWN)
        if i == FingerState.OPEN and m == FingerState.OPEN and r == FingerState.OPEN and p != FingerState.OPEN:
            return NeuroGripCommand.THREE_FINGER.value, 1.0, None

        # 6. INDEX_PINKY: Index, Pinky OPEN, Middle & Ring NOT OPEN (CLOSED or UNKNOWN)
        if i == FingerState.OPEN and p == FingerState.OPEN and m != FingerState.OPEN and r != FingerState.OPEN:
            return NeuroGripCommand.INDEX_PINKY.value, 1.0, None

        # 7. TWO_FINGER: Index, Middle OPEN, Ring & Pinky NOT OPEN (CLOSED or UNKNOWN)
        if i == FingerState.OPEN and m == FingerState.OPEN and r != FingerState.OPEN and p != FingerState.OPEN:
            return NeuroGripCommand.TWO_FINGER.value, 1.0, None

        # 8. Single Finger Commands
        if i == FingerState.OPEN and m == FingerState.CLOSED and r == FingerState.CLOSED and p == FingerState.CLOSED:
            return NeuroGripCommand.INDEX.value, 1.0, None
        if m == FingerState.OPEN and i == FingerState.CLOSED and r == FingerState.CLOSED and p == FingerState.CLOSED:
            return NeuroGripCommand.MIDDLE.value, 1.0, None
        if r == FingerState.OPEN and i == FingerState.CLOSED and m == FingerState.CLOSED and p == FingerState.CLOSED:
            return NeuroGripCommand.RING.value, 1.0, None
        if p == FingerState.OPEN and i == FingerState.CLOSED and m == FingerState.CLOSED and r == FingerState.CLOSED:
            return NeuroGripCommand.PINKY.value, 1.0, None

        # 9. REST: Relaxed idle hand (e.g. 0-2 fingers open/unknown without matching action)
        if num_open <= 2 and num_closed <= 3:
            return NeuroGripCommand.REST.value, 0.70, None

        return "UNKNOWN", 0.0, None

    def _is_grab_claw_pose(self, landmarks: HandLandmarks, features: np.ndarray) -> bool:
        """
        Check if hand landmarks match the GRAB partial-curl claw configuration
        using configurable grab_calibration parameters.
        """
        if len(features) < 67:
            return False

        # Group D (features 57..61): 5 tip-to-MCP extension ratios
        ext_ratios = features[57:62]
        # Group C (features 47..56): 10 joint cosines
        joint_cosines = features[47:57]

        # Check if all extension ratios fall within calibrated GRAB range [min_ext, max_ext]
        min_ext = self.grab_cfg.min_extension_ratio
        max_ext = self.grab_cfg.max_extension_ratio
        in_ext_range = all(min_ext <= r <= max_ext for r in ext_ratios)

        # Check if joint cosines fall within calibrated range [min_cos, max_cos]
        min_cos = self.grab_cfg.min_cosine
        max_cos = self.grab_cfg.max_cosine
        in_cos_range = all(min_cos <= c <= max_cos for c in joint_cosines)

        return in_ext_range and in_cos_range

    def _is_compact_flat_hand(self, landmarks: HandLandmarks) -> bool:
        """
        Check if extended flat hand has fingers compact/adducted together (used for STOP distinction).
        """
        lms = landmarks.landmarks
        # Tips: Index(8), Middle(12), Ring(16), Pinky(20)
        idx_tip = lms[8]
        mid_tip = lms[12]
        rng_tip = lms[16]
        pky_tip = lms[20]

        span_im = math.hypot(mid_tip.x - idx_tip.x, mid_tip.y - idx_tip.y)
        span_mr = math.hypot(rng_tip.x - mid_tip.x, rng_tip.y - mid_tip.y)
        span_rp = math.hypot(pky_tip.x - rng_tip.x, pky_tip.y - rng_tip.y)

        # Total tip spread across index->middle->ring->pinky
        total_tip_spread = span_im + span_mr + span_rp
        # Compact threshold: sum of tip distances < 0.15 in image space
        return total_tip_spread < 0.15

    def _is_wide_thumb_spread(self, landmarks: HandLandmarks) -> bool:
        """
        Check if extended thumb is widely spread/abducted away from palm & index finger.
        Distinguishes a true wide STOP hand from a FOUR_FINGERS pose with a relaxed/noisy thumb.
        """
        if landmarks is None or len(landmarks.landmarks) < 21:
            return False

        lms = landmarks.landmarks
        w_idx = self.config.features.wrist_landmark
        s_idx = self.config.features.span_landmark

        wrist = lms[w_idx]
        middle_mcp = lms[s_idx]

        span = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        if span < 1e-6:
            span = 1.0

        is_left = landmarks.handedness == Handedness.LEFT

        # Thumb tip: LM 4, Index MCP: LM 5
        th_tip = lms[4]
        idx_mcp = lms[5]

        # Normalized coordinates relative to wrist and span
        dx_th = (th_tip.x - wrist.x) / span
        dy_th = (th_tip.y - wrist.y) / span
        dz_th = (th_tip.z - wrist.z) / span if hasattr(wrist, "z") else th_tip.z
        if is_left:
            dx_th = -dx_th

        dx_idx = (idx_mcp.x - wrist.x) / span
        dy_idx = (idx_mcp.y - wrist.y) / span
        dz_idx = (idx_mcp.z - wrist.z) / span if hasattr(wrist, "z") else idx_mcp.z
        if is_left:
            dx_idx = -dx_idx

        # 3D distance between Thumb tip and Index MCP in normalized space
        dist_th_idx = math.sqrt((dx_th - dx_idx)**2 + (dy_th - dy_idx)**2 + (dz_th - dz_idx)**2)

        # In STOP, thumb tip is widely spread away from index MCP
        return dist_th_idx >= 0.38 or dx_th >= 0.25
