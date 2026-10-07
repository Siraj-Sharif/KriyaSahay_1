"""
neurogrip/features/extractor.py
───────────────────────────────
Feature Extraction Layer.
Transforms raw 21 HandLandmarks into normalized feature vectors.
Supports Feature Version v2 (canonical 68-D vector) and Feature Version v1 (legacy 67-D vector).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from neurogrip.config.settings import FeaturesConfig
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts normalized feature vectors from HandLandmarks objects.

    Default Feature Version: v2 (68 dimensions)
    Legacy Feature Version:  v1 (67 dimensions)

    v2 Feature Vector Composition (Total: 68 dimensions):
    - [0..39]  (40): Normalized (x, y) coordinates for landmarks 1..20 (wrist origin excluded, 2D span rescaled, x-mirrored for left hand)
    - [40..44] (5) : Normalized relative 3D fingertip z-depths (landmarks 4, 8, 12, 16, 20 relative to wrist, divided by 3D span)
    - [45..54] (10): 10 3D joint-angle cosines (2 per finger x 5 fingers, clipped to [-1.0, +1.0])
    - [55..59] (5) : 5 Finger Straightness Indices (ratio of 3D chord length to 3D phalanx polyline length)
    - [60..64] (5) : 5 Inter-Finger Contrast Indices (S_k minus average straightness of other 4 fingers, clipped to [-1.0, +1.0])
    - [65]     (1) : THUMB_INDEX_DIRECTION_COSINE (3D directional divergence between thumb and index, clipped to [-1.0, +1.0])
    - [66]     (1) : Thumb-to-Palm 3D distance (distance between landmark 4 and middle MCP landmark 9, divided by 3D span)
    - [67]     (1) : Minimum Finger Straightness (min straightness across all 5 fingers)
    """

    FEATURE_DIM: int = 68
    FEATURE_VERSION: str = "v2"

    # Joint angle triplets (A -> joint B -> C)
    JOINT_TRIPLETS: list[tuple[int, int, int]] = [
        # Thumb
        (1, 2, 3), (2, 3, 4),
        # Index
        (5, 6, 7), (6, 7, 8),
        # Middle
        (9, 10, 11), (10, 11, 12),
        # Ring
        (13, 14, 15), (14, 15, 16),
        # Pinky
        (17, 18, 19), (18, 19, 20),
    ]

    # Fingertip landmark indices (Thumb, Index, Middle, Ring, Pinky)
    TIP_INDICES: list[int] = [4, 8, 12, 16, 20]

    # MCP landmark indices corresponding to fingertips
    MCP_INDICES: list[int] = [2, 5, 9, 13, 17]

    # PIP / IP landmark indices corresponding to fingers (Thumb IP=3)
    PIP_INDICES: list[int] = [3, 6, 10, 14, 18]

    # DIP / IP landmark indices corresponding to fingers (Thumb IP=3)
    DIP_INDICES: list[int] = [3, 7, 11, 15, 19]

    def __init__(self, config: Optional[FeaturesConfig] = None, version: str = "v2") -> None:
        self.config = config or FeaturesConfig()
        self.version = version.lower()

    def extract(self, hand_landmarks: HandLandmarks) -> np.ndarray:
        """
        Extract normalized feature vector based on current configured version.
        """
        if self.version == "v1":
            return self.extract_v1(hand_landmarks)
        return self.extract_v2(hand_landmarks)

    def extract_v2(self, hand_landmarks: HandLandmarks) -> np.ndarray:
        """
        Extract the locked 68-dimensional Feature Version v2 vector.
        """
        if hand_landmarks is None or len(hand_landmarks.landmarks) != 21:
            raise ValueError(
                f"FeatureExtractor requires exactly 21 landmarks, got "
                f"{len(hand_landmarks.landmarks) if hand_landmarks else 0}."
            )

        lms = hand_landmarks.landmarks
        w_idx = self.config.wrist_landmark  # 0
        s_idx = self.config.span_landmark   # 9

        wrist = lms[w_idx]
        span_lm = lms[s_idx]

        # 1. Span Calculations
        dx_span = span_lm.x - wrist.x
        dy_span = span_lm.y - wrist.y
        dz_span = span_lm.z - wrist.z

        span2d_val = float(np.hypot(dx_span, dy_span))
        l_span2d = max(span2d_val, 1e-6)

        span3d_val = float(np.sqrt(dx_span**2 + dy_span**2 + dz_span**2))
        l_span3d = max(span3d_val, 1e-6)

        is_left = hand_landmarks.handedness == Handedness.LEFT

        # Group 1: Normalized 2D Coordinates [0..39] (40 features for landmarks 1..20)
        norm_2d = np.zeros((20, 2), dtype=np.float32)
        for i in range(1, 21):
            lm = lms[i]
            dx = (lm.x - wrist.x) / l_span2d
            dy = (lm.y - wrist.y) / l_span2d
            if is_left:
                dx = -dx  # Mirror x coordinate for left hand
            norm_2d[i - 1, 0] = dx
            norm_2d[i - 1, 1] = dy

        coords_flat = norm_2d.flatten()  # 40 floats

        # Group 2: Normalized Relative 3D Fingertip Z [40..44] (5 features)
        z_depths_norm = np.zeros(5, dtype=np.float32)
        for idx, tip_i in enumerate(self.TIP_INDICES):
            dz = (lms[tip_i].z - wrist.z) / l_span3d
            z_depths_norm[idx] = float(dz)

        # Group 3: 3D Joint-Angle Cosines [45..54] (10 features)
        joint_cosines_3d = np.zeros(10, dtype=np.float32)
        for idx, (a_i, b_i, c_i) in enumerate(self.JOINT_TRIPLETS):
            p_a = np.array([lms[a_i].x, lms[a_i].y, lms[a_i].z], dtype=np.float64)
            p_b = np.array([lms[b_i].x, lms[b_i].y, lms[b_i].z], dtype=np.float64)
            p_c = np.array([lms[c_i].x, lms[c_i].y, lms[c_i].z], dtype=np.float64)

            v1 = p_a - p_b
            v2 = p_c - p_b

            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))

            dot_val = float(np.dot(v1, v2))
            cos_val = dot_val / (norm1 * norm2 + 1e-7)
            joint_cosines_3d[idx] = float(np.clip(cos_val, -1.0, 1.0))

        # Group 4: Finger Straightness Index [55..59] (5 features)
        # Order: THUMB (0), INDEX (1), MIDDLE (2), RING (3), PINKY (4)
        straightness = np.zeros(5, dtype=np.float32)
        for idx in range(5):
            tip_i = self.TIP_INDICES[idx]
            mcp_i = self.MCP_INDICES[idx]
            pip_i = self.PIP_INDICES[idx]
            dip_i = self.DIP_INDICES[idx]

            p_tip = np.array([lms[tip_i].x, lms[tip_i].y, lms[tip_i].z], dtype=np.float64)
            p_mcp = np.array([lms[mcp_i].x, lms[mcp_i].y, lms[mcp_i].z], dtype=np.float64)
            p_pip = np.array([lms[pip_i].x, lms[pip_i].y, lms[pip_i].z], dtype=np.float64)
            p_dip = np.array([lms[dip_i].x, lms[dip_i].y, lms[dip_i].z], dtype=np.float64)

            d_chord = float(np.linalg.norm(p_tip - p_mcp))

            if idx == 0:
                # Thumb: MCP (2), IP (3), TIP (4)
                d_poly = float(np.linalg.norm(p_pip - p_mcp) + np.linalg.norm(p_tip - p_pip))
            else:
                d_poly = float(
                    np.linalg.norm(p_pip - p_mcp) +
                    np.linalg.norm(p_dip - p_pip) +
                    np.linalg.norm(p_tip - p_dip)
                )

            s_val = d_chord / max(d_poly, 1e-6)
            straightness[idx] = float(s_val)

        # Group 5: Inter-Finger Contrast [60..64] (5 features)
        # C_k = S_k - (1/4) * sum(S_j for j != k)
        contrast = np.zeros(5, dtype=np.float32)
        s_sum = float(np.sum(straightness))
        for k in range(5):
            s_k = straightness[k]
            s_other_avg = (s_sum - s_k) / 4.0
            c_k = s_k - s_other_avg
            contrast[k] = float(np.clip(c_k, -1.0, 1.0))

        # Group 6: Thumb-Index Direction Cosine [65] (1 feature)
        # v_thumb = LM4 - LM2, v_index = LM8 - LM5
        p_lm4 = np.array([lms[4].x, lms[4].y, lms[4].z], dtype=np.float64)
        p_lm2 = np.array([lms[2].x, lms[2].y, lms[2].z], dtype=np.float64)
        p_lm8 = np.array([lms[8].x, lms[8].y, lms[8].z], dtype=np.float64)
        p_lm5 = np.array([lms[5].x, lms[5].y, lms[5].z], dtype=np.float64)

        v_thumb = p_lm4 - p_lm2
        v_index = p_lm8 - p_lm5

        norm_th = float(np.linalg.norm(v_thumb))
        norm_in = float(np.linalg.norm(v_index))

        dot_ti = float(np.dot(v_thumb, v_index))
        cos_alpha = dot_ti / (norm_th * norm_in + 1e-7)
        cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))

        # Group 7: Thumb-to-Palm Distance [66] (1 feature)
        # D_tp = ||LM4 - LM9||_3D / L_span3D
        p_lm9 = np.array([lms[9].x, lms[9].y, lms[9].z], dtype=np.float64)
        d_tp = float(np.linalg.norm(p_lm4 - p_lm9) / l_span3d)

        # Group 8: Minimum Finger Straightness [67] (1 feature)
        s_min = float(np.min(straightness))

        vector = np.concatenate([
            coords_flat,                              # 40
            z_depths_norm,                            # 5
            joint_cosines_3d,                         # 10
            straightness,                             # 5
            contrast,                                 # 5
            np.array([cos_alpha], dtype=np.float32),  # 1
            np.array([d_tp], dtype=np.float32),       # 1
            np.array([s_min], dtype=np.float32),      # 1
        ], axis=0).astype(np.float32)

        assert vector.shape == (68,), f"Expected shape (68,), got {vector.shape}"
        assert np.all(np.isfinite(vector)), "Feature vector v2 contains non-finite values."
        return vector

    def extract_v1(self, hand_landmarks: HandLandmarks) -> np.ndarray:
        """
        Extract legacy 67-dimensional Feature Version v1 vector.
        Preserved for backward compatibility and regression testing.
        """
        if hand_landmarks is None or len(hand_landmarks.landmarks) != 21:
            raise ValueError(
                f"FeatureExtractor requires exactly 21 landmarks, got "
                f"{len(hand_landmarks.landmarks) if hand_landmarks else 0}."
            )

        lms = hand_landmarks.landmarks
        w_idx = self.config.wrist_landmark
        s_idx = self.config.span_landmark

        wrist = lms[w_idx]
        span_lm = lms[s_idx]

        span = float(np.hypot(span_lm.x - wrist.x, span_lm.y - wrist.y))
        if span < 1e-6:
            span = 1.0

        is_left = hand_landmarks.handedness == Handedness.LEFT

        # Group A: 42 normalized (x, y) coordinates
        norm_coords = np.zeros((21, 2), dtype=np.float32)
        for i, lm in enumerate(lms):
            dx = (lm.x - wrist.x) / span
            dy = (lm.y - wrist.y) / span
            if is_left:
                dx = -dx
            norm_coords[i, 0] = dx
            norm_coords[i, 1] = dy

        coords_flat = norm_coords.flatten()

        # Group B: 5 fingertip z-depths
        z_depths = np.array([lms[idx].z for idx in self.TIP_INDICES], dtype=np.float32)

        # Group C: 10 joint-angle cosines
        joint_cosines = np.zeros(10, dtype=np.float32)
        for idx, (a_i, b_i, c_i) in enumerate(self.JOINT_TRIPLETS):
            pos_a = norm_coords[a_i]
            pos_b = norm_coords[b_i]
            pos_c = norm_coords[c_i]

            v1 = pos_b - pos_a
            v2 = pos_c - pos_b

            mag1 = float(np.hypot(v1[0], v1[1]))
            mag2 = float(np.hypot(v2[0], v2[1]))

            if mag1 < 1e-6 or mag2 < 1e-6:
                joint_cosines[idx] = 0.0
            else:
                dot = float(np.dot(v1, v2))
                cos_val = dot / (mag1 * mag2 + 1e-7)
                joint_cosines[idx] = float(np.clip(cos_val, -1.0, 1.0))

        # Group D: 5 tip-to-MCP extension ratios
        extension_ratios = np.zeros(5, dtype=np.float32)
        for idx, (tip_i, mcp_i) in enumerate(zip(self.TIP_INDICES, self.MCP_INDICES)):
            pos_tip = norm_coords[tip_i]
            pos_mcp = norm_coords[mcp_i]
            extension_ratios[idx] = float(np.hypot(pos_tip[0] - pos_mcp[0], pos_tip[1] - pos_mcp[1]))

        # Group E: 5 tip-to-wrist distances
        tip_wrist_dists = np.zeros(5, dtype=np.float32)
        for idx, tip_i in enumerate(self.TIP_INDICES):
            pos_tip = norm_coords[tip_i]
            tip_wrist_dists[idx] = float(np.hypot(pos_tip[0], pos_tip[1]))

        vector = np.concatenate([
            coords_flat,
            z_depths,
            joint_cosines,
            extension_ratios,
            tip_wrist_dists,
        ], axis=0).astype(np.float32)

        assert vector.shape == (67,), f"Expected shape (67,), got {vector.shape}"
        return vector
