"""
pc/scripts/benchmark_pretrained_gesture.py
─────────────────────────────────────────────
Phase 10K — Pretrained Gesture Recognizer Benchmark CLI.
ISOLATED EXPERIMENTAL CLI BENCHMARK.
Does NOT modify or replace MLRecognizer, HybridRecognizer, or hardware pipeline.
Does NOT transmit active hardware/serial/TCP commands.

Runs real-time webcam inference using open-source Pretrained Gesture Recognizer (HaGRID / MediaPipe).
Measures:
- Pretrained gesture class & confidence
- Mapped NeuroGrip command & correspondence level (DIRECT / APPROXIMATE / NONE)
- MediaPipe hand detection status & handedness
- Real-time inference latency (ms) & FPS

Usage:
    python pc/scripts/benchmark_pretrained_gesture.py --camera 0
    python pc/scripts/benchmark_pretrained_gesture.py --camera 0 --no-gui
    python pc/scripts/benchmark_pretrained_gesture.py --camera 0 --max-frames 150
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

# Ensure src directory is in sys.path
PC_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PC_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.recognition.pretrained_gesture import PretrainedGestureRecognizer, PretrainedRecognitionResult

logger = logging.getLogger("neurogrip.benchmark_pretrained")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 10K Pretrained Gesture Model Benchmark CLI."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to pretrained gesture model task bundle (default: pc/models/mediapipe/gesture_recognizer.task).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without OpenCV visualization window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting.",
    )
    return parser


def draw_benchmark_overlay(frame: np.ndarray, res: PretrainedRecognitionResult, fps: float) -> np.ndarray:
    """Draw experimental HUD overlay on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw dark header banner
    banner_h = 130
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.rectangle(canvas, (0, banner_h - 2), (w, banner_h), (0, 200, 255), 2)

    title_str = "NEUROGRIP PRETRAINED GESTURE BENCHMARK (EXPERIMENTAL)"
    cv2.putText(canvas, title_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    if not res.is_hand_detected:
        status_str = "STATUS: NO HAND DETECTED"
        cv2.putText(canvas, status_str, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        raw_str = f"PRETRAINED CLASS : {res.raw_pretrained_class} ({res.pretrained_confidence*100:.1f}%)"
        mapped_str = f"MAPPED COMMAND   : {res.mapped_neurogrip_command} [{res.correspondence_level}]"
        handed_str = f"HANDEDNESS       : {res.handedness} ({res.handedness_confidence*100:.0f}%)"

        color_corr = (0, 255, 0) if res.correspondence_level == "DIRECT" else ((0, 255, 255) if res.correspondence_level == "APPROXIMATE" else (0, 0, 255))

        cv2.putText(canvas, raw_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, mapped_str, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_corr, 2)
        cv2.putText(canvas, handed_str, (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    perf_str = f"FPS: {fps:.1f} | Latency: {res.inference_latency_ms:.1f} ms"
    cv2.putText(canvas, perf_str, (w - 280, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return canvas


def run_benchmark(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = AppConfig.default()
    cfg.camera.index = args.camera
    cfg.camera.backend = "opencv"
    configure_logging(cfg.logging)

    print("==========================================================================")
    print("NEUROGRIP EXPERIMENTAL PRETRAINED GESTURE BENCHMARK")
    print("==========================================================================")
    print(f"  Camera Device Index : {args.camera}")
    print(f"  Model Path          : {args.model_path or 'default'}")
    print(f"  GUI Window Enabled  : {not args.no_gui}")
    print("==========================================================================")

    camera = OpenCVCamera(config=cfg.camera)
    recognizer = PretrainedGestureRecognizer(model_path=args.model_path, config=cfg)

    if not camera.open():
        logger.error("Failed to open camera index %d.", args.camera)
        return 1

    if not recognizer.initialize():
        logger.error("Failed to initialize PretrainedGestureRecognizer.")
        camera.close()
        return 1

    window_name = "NeuroGrip Pretrained Gesture Benchmark (Experimental)"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    frames_processed = 0
    fps_history: list[float] = []
    last_t = time.time()

    try:
        while True:
            frame_container = camera.read_frame()
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            avg_fps = float(np.mean(fps_history)) if fps_history else 0.0

            if not frame_container.is_valid or frame_container.frame is None:
                time.sleep(0.005)
                continue

            raw_frame = frame_container.frame
            timestamp_ms = frame_container.timestamp_ms

            res = recognizer.predict_frame(raw_frame, timestamp_ms)
            frames_processed += 1

            if frames_processed % 10 == 0:
                logger.info(
                    "[F%05d] Raw: %s (%.2f) | Mapped: %s [%s] | Hand: %s | Latency: %.1fms | FPS: %.1f",
                    frames_processed,
                    res.raw_pretrained_class,
                    res.pretrained_confidence,
                    res.mapped_neurogrip_command,
                    res.correspondence_level,
                    res.handedness,
                    res.inference_latency_ms,
                    avg_fps,
                )

            if not args.no_gui:
                rendered = draw_benchmark_overlay(raw_frame, res, avg_fps)
                cv2.imshow(window_name, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("Quit key pressed. Exiting benchmark.")
                    break

            if args.max_frames is not None and frames_processed >= args.max_frames:
                logger.info("Max frames reached (%d). Exiting benchmark loop.", args.max_frames)
                break

    finally:
        recognizer.close()
        camera.close()
        if not args.no_gui:
            cv2.destroyAllWindows()

    print(f"\nPretrained gesture benchmark complete. Processed {frames_processed} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
