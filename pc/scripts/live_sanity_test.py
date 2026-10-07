"""
scripts/live_sanity_test.py
────────────────────────────
Read-Only Live Camera Sanity Test for Phase 10H Recognition & Validation.

Runs the live camera pipeline (Camera -> HandDetector -> FingerStateDetector -> RuleBasedRecognizer)
and displays real-time diagnostic information on screen and terminal:
- Detected handedness (RIGHT / LEFT)
- Individual finger states (THUMB, INDEX, MIDDLE, RING, PINKY)
- Palm-facing status (True / False)
- Rule-based gesture recognition & confidence

Usage:
    python pc/scripts/live_sanity_test.py
    python pc/scripts/live_sanity_test.py --mock-camera --max-frames 30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

# Ensure src directory is in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.camera.base import CameraInterface
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import Finger, FingerStateDetector
from neurogrip.hand_tracking.detector import HandDetector
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult
from neurogrip.recognition.rule_based import RuleBasedRecognizer

logger = logging.getLogger("neurogrip.live_sanity_test")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 10H Read-Only Live Recognition Sanity Test Utility."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without OpenCV visualization window.",
    )
    parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use MockCamera instead of physical webcam (for automated testing).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting automatically.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file.",
    )
    return parser


def run_sanity_test(
    no_gui: bool = False,
    mock_camera: bool = False,
    max_frames: Optional[int] = None,
    config_path: Optional[Path] = None,
    camera_override: Optional[CameraInterface] = None,
    detector_override: Optional[HandDetector] = None,
    extractor_override: Optional[FeatureExtractor] = None,
    recognizer_override: Optional[GestureRecognizer] = None,
) -> int:
    """
    Execute read-only live sanity test loop.
    """
    # 1. Load config
    if config_path is None:
        config = AppConfig.default()
    else:
        config = AppConfig.from_yaml(config_path)

    configure_logging(config.logging)

    # 2. Initialize components
    if camera_override is not None:
        camera = camera_override
    elif mock_camera or config.camera.backend == "mock":
        camera = MockCamera(config=config.camera)
    else:
        camera = OpenCVCamera(config=config.camera)

    detector = detector_override or HandDetector(config=config.hand_tracking)
    extractor = extractor_override or FeatureExtractor(config=config.features, version="v2")
    recognizer = recognizer_override or RuleBasedRecognizer(config=config)
    finger_detector = FingerStateDetector(config.features)

    logger.info("Starting Phase 10H Live Recognition Sanity Test...")
    logger.info("  Camera Backend : %s", camera.__class__.__name__)
    logger.info("  Headless Mode  : %s", no_gui)
    logger.info("  Max Frames     : %s", max_frames)
    logger.info("  (Press ESC or 'q' in video window to exit)")

    if not camera.open():
        logger.error("Failed to open camera.")
        return 1

    try:
        detector.initialize()
    except Exception as e:
        logger.error("Failed to initialize HandDetector: %s", e)
        camera.close()
        return 1

    frames_processed = 0
    last_terminal_log_time = 0.0

    try:
        window_name = "NeuroGrip - Phase 10H Live Recognition Sanity Test"
        if not no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            if max_frames is not None and frames_processed >= max_frames:
                logger.info("Max frames reached (%d). Exiting sanity test loop.", max_frames)
                break

            frame_container = camera.read_frame()
            if not frame_container.is_valid or frame_container.frame is None:
                frames_processed += 1
                time.sleep(0.001)
                continue

            frames_processed += 1
            frame = frame_container.frame
            timestamp_ms = frame_container.timestamp_ms
            detection_res = detector.detect(frame, timestamp_ms)

            h, w, _ = frame.shape
            display_lines: list[str] = []
            display_lines.append(f"Hands Detected: {detection_res.num_hands}")

            if detection_res.is_empty:
                display_lines.append("Status: NO_HAND")
            else:
                for idx, hand in enumerate(detection_res.hands):
                    handedness_str = hand.handedness.value
                    finger_states = finger_detector.detect(hand)
                    features = extractor.extract(hand)
                    rec_res = recognizer.predict(features, hand_landmarks=hand, is_stop_armed=False)

                    # Compute palm-facing orientation check manually for display
                    lms = hand.landmarks
                    wrist = lms[config.features.wrist_landmark]
                    middle_mcp = lms[config.features.span_landmark]
                    span = max(1e-6, math_hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y))
                    is_left = hand.handedness == Handedness.LEFT
                    norm_lms = []
                    for lm in lms:
                        dx = (lm.x - wrist.x) / span
                        dy = (lm.y - wrist.y) / span
                        dz = (lm.z - wrist.z) / span if hasattr(wrist, "z") else lm.z
                        if is_left:
                            dx = -dx
                        norm_lms.append((dx, dy, dz))

                    is_palm_facing = FingerStateDetector.is_palm_facing_camera(norm_lms)

                    states_summary = (
                        f"T:{finger_states.thumb.value[0]} "
                        f"I:{finger_states.index.value[0]} "
                        f"M:{finger_states.middle.value[0]} "
                        f"R:{finger_states.ring.value[0]} "
                        f"P:{finger_states.pinky.value[0]}"
                    )

                    spread_metric = (
                        recognizer.get_thumb_spread_metric(hand)
                        if isinstance(recognizer, RuleBasedRecognizer)
                        else 0.0
                    )
                    if spread_metric < RuleBasedRecognizer.THUMB_SPREAD_LOWER_THRESH:
                        spread_status = "NARROW"
                    elif spread_metric >= RuleBasedRecognizer.THUMB_SPREAD_UPPER_THRESH:
                        spread_status = "WIDE"
                    else:
                        spread_status = "AMBIGUOUS"

                    hand_prefix = f"Hand #{idx+1} ({handedness_str})"
                    display_lines.append(f"{hand_prefix} PalmFacing: {is_palm_facing}")
                    display_lines.append(f"  Fingers: {states_summary}")
                    display_lines.append(
                        f"  ThumbSpread: {spread_metric:.2f} [{spread_status}] (T_low: 0.35, T_high: 0.45)"
                    )
                    display_lines.append(f"  Gesture: {rec_res.label} (conf: {rec_res.confidence:.2f})")

            # Periodically print to terminal log (every 1 second)
            now = time.time()
            if now - last_terminal_log_time >= 1.0:
                last_terminal_log_time = now
                logger.info("Frame %d | %s", frames_processed, " | ".join(display_lines))

            # Overlay rendering if GUI enabled
            if not no_gui:
                y_offset = 30
                cv2.putText(
                    frame,
                    "NeuroGrip Phase 10H Sanity Test (ESC/Q to exit)",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_offset += 25
                for line in display_lines:
                    color = (0, 255, 0) if "Gesture:" in line and "UNKNOWN" not in line else (255, 255, 255)
                    if "NO_HAND" in line:
                        color = (0, 0, 255)
                    cv2.putText(
                        frame,
                        line,
                        (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_offset += 20

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q") or key == ord("Q"):
                    logger.info("User requested exit.")
                    break

    finally:
        camera.close()
        detector.close()
        if not no_gui:
            cv2.destroyAllWindows()

    logger.info("Live Sanity Test finished. Processed %d frames.", frames_processed)
    return 0


def math_hypot(dx: float, dy: float) -> float:
    import math
    return math.hypot(dx, dy)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return run_sanity_test(
        no_gui=args.no_gui,
        mock_camera=args.mock_camera,
        max_frames=args.max_frames,
        config_path=args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
