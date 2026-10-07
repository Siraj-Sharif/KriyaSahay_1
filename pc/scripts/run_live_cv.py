"""
scripts/run_live_cv.py
───────────────────────
Phase 10K.1 — Production Live Computer Vision Pipeline (Stage 1 Boundary).

Executes real-time webcam CV processing loop:
  Webcam (Index 0)
  → MediaPipe Hand Detection
  → 68-D Feature Extraction
  → Deterministic STOP Safety Detector
  → Extra Trees ML Model (NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl)
  → Command Validation (REST = IDLE)
  → Temporal Stabilization
  → Command Transport Boundary (DisplayConsoleTransport: NG1|<COMMAND>)
  → On-Screen Visualization HUD

Usage:
    python pc/scripts/run_live_cv.py --camera 0
    python pc/scripts/run_live_cv.py --camera 0 --no-gui
    python pc/scripts/run_live_cv.py --camera 0 --max-frames 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.communication.transport import DisplayConsoleTransport
from neurogrip.config.settings import AppConfig, configure_logging

logger = logging.getLogger("neurogrip.run_live_cv")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 10K.1 Live Computer Vision Pipeline Launcher."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target camera frame width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target camera frame height (default: 720).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera capture FPS (default: 30).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Minimum recognition confidence threshold (default: 0.60).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without OpenCV visualization GUI window.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Enable compact read-only diagnostic output every 10 frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for testing/benchmarking).",
    )
    return parser


def run_live_pipeline(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    cfg = AppConfig.default()
    cfg.camera.index = args.camera
    cfg.camera.width = args.width
    cfg.camera.height = args.height
    cfg.camera.fps = args.fps
    cfg.camera.backend = "opencv"
    cfg.recognition.normal_min_confidence = args.confidence
    cfg.visualization.enabled = not args.no_gui

    # Disable serial hardware interface in Stage 1
    cfg.serial.enabled = False
    cfg.serial.mock = True

    configure_logging(cfg.logging)

    print("==========================================================================")
    print(" NEUROGRIP — FINAL PRODUCTION HYBRID LIVE CV PIPELINE")
    print("==========================================================================")
    print(f"  Camera Device Index : {args.camera}")
    print(f"  Target Resolution   : {args.width}x{args.height} @ {args.fps} FPS")
    print(f"  Confidence Threshold: {args.confidence}")
    print(f"  GUI Window Enabled  : {not args.no_gui}")
    print(f"  Diagnostic Mode     : {args.diagnostic}")
    print("--------------------------------------------------------------------------")
    print("  Models & Routing Verification:")
    print("    - Primary Model   : HaGRID ResNet18 (Full-Frame, 34 raw classes)")
    print("    - Specialized ML  : Extra Trees (68-D features, 6100 samples)")
    print("    - Routing Rule    : HaGRID [INDEX_FINGER, TWO_FINGERS, PINKY] -> Extra Trees")
    print("    - Taxonomy        : Locked 14 Canonical Gestures (No ONE_FINGER)")
    print("    - Transport       : NG1|<COMMAND> (NO_COMMAND blocked)")
    print("==========================================================================")
    print("  Controls:")
    print("    Press 'q' or 'ESC' to quit cleanly.")
    print("    Press 'SPACE' to toggle/arm software STOP safety path.")
    print("==========================================================================")

    transport = DisplayConsoleTransport()

    with NeuroGripPipeline(config=cfg, transport=transport) as pipeline:
        # Pre-arm STOP safety detection for production live operation
        pipeline.arm_stop()

        window_name = "NeuroGrip — Final Production Hybrid Live CV Pipeline"
        if not args.no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, args.width, args.height)

        frames_processed = 0
        try:
            while pipeline.is_running:
                frame_container, rendered, viz_state = pipeline.process_frame()

                if args.diagnostic and (frames_processed % 10 == 0):
                    print(f"[DIAGNOSTIC F{frames_processed:05d}] {pipeline.format_diagnostic_line()}")

                if not args.no_gui and rendered is not None:
                    cv2.imshow(window_name, rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        logger.info("Quit key pressed ('%c'). Exiting live pipeline.", chr(key) if key < 128 else "ESC")
                        break
                    elif key == ord(" "):
                        pipeline.toggle_stop_arm()

                frames_processed += 1
                if args.max_frames is not None and frames_processed >= args.max_frames:
                    logger.info("Max frames limit reached (%d). Exiting loop.", args.max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Live pipeline interrupted by KeyboardInterrupt.")
        finally:
            if not args.no_gui:
                cv2.destroyAllWindows()

    print("\nLive CV Pipeline shutdown complete. Total frames processed:", frames_processed)
    return 0


if __name__ == "__main__":
    sys.exit(run_live_pipeline())
