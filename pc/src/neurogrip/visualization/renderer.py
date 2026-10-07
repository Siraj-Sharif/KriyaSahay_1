"""
neurogrip/visualization/renderer.py
───────────────────────────────────
Overlay Renderer Implementation.
Draws 21 hand landmarks, skeleton connections, status HUD panel, and safety STOP alerts
onto OpenCV BGR image frames.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from neurogrip.config.settings import AppConfig, VisualizationConfig
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks
from neurogrip.visualization.base import VisualizationInterface, VisualizationState

logger = logging.getLogger(__name__)


class OverlayRenderer(VisualizationInterface):
    """
    OpenCV-based overlay renderer.
    Annotates image frames with hand skeleton graphics and real-time status HUD metadata.
    """

    # 21 MediaPipe hand landmark connections
    HAND_CONNECTIONS: list[tuple[int, int]] = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (5, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (9, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (13, 17), (17, 18), (18, 19), (19, 20),
        # Wrist to Pinky base
        (0, 17),
    ]

    FINGERTIP_INDICES: set[int] = {4, 8, 12, 16, 20}

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or AppConfig.default().visualization

    def render(self, frame: np.ndarray, state: VisualizationState) -> np.ndarray:
        """
        Render visual overlays onto a copy of the input frame.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()
        if not self.config.enabled:
            return out

        h, w = out.shape[:2]

        # 1. Draw Hand Landmarks & Skeleton
        if self.config.show_landmarks and state.landmarks:
            for hand in state.landmarks:
                self._draw_hand_skeleton(out, hand, w, h)

        # 2. Draw Status HUD Panel
        self._draw_status_hud(out, state, w, h)

        return out

    def _draw_hand_skeleton(self, img: np.ndarray, hand: HandLandmarks, width: int, height: int) -> None:
        """Draw skeleton connection lines and landmark circles for a single hand."""
        pts: list[tuple[int, int]] = []
        for lm in hand.landmarks:
            px = int(np.clip(lm.x * width, 0, width - 1))
            py = int(np.clip(lm.y * height, 0, height - 1))
            pts.append((px, py))

        # Draw skeleton connections
        if self.config.show_skeleton:
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                pt1 = pts[start_idx]
                pt2 = pts[end_idx]
                cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)  # Cyan lines

        # Draw joint nodes and fingertips
        for idx, pt in enumerate(pts):
            if idx in self.FINGERTIP_INDICES:
                cv2.circle(img, pt, 6, (255, 0, 255), -1, cv2.LINE_AA)  # Magenta fingertips
                cv2.circle(img, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)  # Yellow joints

        # Annotate handedness near wrist (landmark 0)
        if hand.handedness != Handedness.UNKNOWN and len(pts) > 0:
            wrist_pt = pts[0]
            label_str = hand.handedness.value
            cv2.putText(
                img,
                label_str,
                (wrist_pt[0] - 20, wrist_pt[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_status_hud(self, img: np.ndarray, state: VisualizationState, width: int, height: int) -> None:
        """Draw semi-transparent status HUD panel at top-left displaying production hybrid metrics."""
        # Panel bounds
        px1, py1 = 15, 15
        px2, py2 = 380, 295

        # Semi-transparent dark background box
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (80, 80, 80), 1)

        y_cursor = py1 + 22

        # Line 1: MODEL & ROUTING
        model_str = f"MODEL: {state.model_used} | ROUTING: {state.routing_str}"
        cv2.putText(img, str(model_str), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Line 2: RAW PREDICTION
        y_cursor += 24
        raw_str = f"RAW PREDICTION: {state.raw_prediction}"
        cv2.putText(img, str(raw_str), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Line 3: FINAL COMMAND
        y_cursor += 26
        cv2.putText(img, "FINAL COMMAND:", (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        cmd_val = str(state.command).upper()
        if state.is_stop_active or cmd_val == "STOP":
            cmd_text = "STOP"
            cmd_color = (0, 0, 255)  # Bright Red
        elif state.is_ambiguous or cmd_val == "AMBIGUOUS":
            cmd_text = "AMBIGUOUS (2+ HANDS)"
            cmd_color = (0, 165, 255)  # Orange
        elif cmd_val in ("NO_HAND", "NO_COMMAND") or str(state.hand_count) == "0":
            cmd_text = "NO_COMMAND"
            cmd_color = (160, 160, 160)  # Gray
        elif cmd_val == "UNKNOWN":
            cmd_text = "UNKNOWN"
            cmd_color = (0, 200, 255)  # Amber
        else:
            cmd_text = cmd_val
            cmd_color = (0, 255, 0)  # Green

        cv2.putText(img, str(cmd_text), (px1 + 140, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.60, cmd_color, 2, cv2.LINE_AA)

        # Line 4: CONFIDENCE & HAND COUNT
        y_cursor += 24
        conf_val = float(state.confidence) if isinstance(state.confidence, (int, float)) else 0.0
        conf_display = f"{conf_val * 100:.1f}%" if conf_val > 0.0 else "N/A"
        metrics_str = f"CONFIDENCE: {conf_display} | HAND COUNT: {state.hand_count}"
        cv2.putText(img, str(metrics_str), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Line 5: STABILITY & SAFETY
        y_cursor += 24
        stab_val = str(state.stabilizer_state)
        stab_color = (0, 255, 0) if stab_val == "STABLE" else (0, 220, 255)
        if stab_val == "STOP":
            stab_color = (0, 0, 255)
        stab_str = f"STABILITY: {stab_val} | SAFETY: {state.safety_status}"
        cv2.putText(img, str(stab_str), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stab_color, 1, cv2.LINE_AA)

        # Line 6: TRANSPORT & LAST TX
        y_cursor += 24
        tx_str = f"Last TX: {state.last_tx}" if state.last_tx != "NONE" else ""
        transport_line = f"Transport: {state.transport_status} {tx_str}".strip()
        t_color = (0, 255, 0) if "CONNECTED" in state.transport_status or "MOCK" in state.transport_status else (150, 150, 150)
        cv2.putText(img, str(transport_line), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, t_color, 1, cv2.LINE_AA)

        # Line 7: FPS & LATENCY
        y_cursor += 22
        fps_val = float(state.fps) if isinstance(state.fps, (int, float)) else 0.0
        lat_val = float(state.latency_ms) if isinstance(state.latency_ms, (int, float)) else 0.0
        fps_lat_str = f"FPS: {fps_val:.1f} | LATENCY: {lat_val:.1f} ms"
        cv2.putText(img, str(fps_lat_str), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

        # Line 8: Custom Message (if present)
        if state.message:
            y_cursor += 20
            cv2.putText(img, str(state.message), (px1 + 10, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)

