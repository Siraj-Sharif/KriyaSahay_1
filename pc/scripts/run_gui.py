"""
scripts/run_gui.py
──────────────────
Phase 3.5 — Production Live Control GUI Dashboard Launcher.

Launches the dedicated High-Contrast Dark Production GUI Console for live development,
testing, demonstration, and robotic control inspection.

Usage:
    python pc/scripts/run_gui.py
    python pc/scripts/run_gui.py --camera 0
    python pc/scripts/run_gui.py --serial-port COM3 --baudrate 115200 --serial real
    python pc/scripts/run_gui.py --no-gui --max-frames 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.gui.dashboard import NeuroGripDashboardApp

logger = logging.getLogger("neurogrip.run_gui")


def create_gui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3.5 Production GUI Live Control Dashboard Launcher."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default=None,
        help="USB serial COM port (e.g. COM3 or /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial communication baud rate (default: 115200).",
    )
    parser.add_argument(
        "--serial",
        type=str,
        choices=["enabled", "disabled", "mock", "real"],
        default="mock",
        help="Serial transport mode: 'mock', 'real'/'enabled', or 'disabled' (default: mock).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without desktop GUI window (useful for headless CI/tests).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (diagnostic/test option).",
    )
    return parser


def run_gui_app(argv: Sequence[str] | None = None) -> int:
    parser = create_gui_parser()
    args = parser.parse_args(argv)

    cfg = AppConfig.default()
    cfg.camera.index = args.camera

    if args.serial_port is not None:
        cfg.serial.port = args.serial_port
    if args.baudrate is not None:
        cfg.serial.baud_rate = args.baudrate

    if args.serial in ("disabled", "off"):
        cfg.serial.enabled = False
        cfg.serial.mock = True
    elif args.serial in ("mock", "test"):
        cfg.serial.enabled = True
        cfg.serial.mock = True
    elif args.serial in ("enabled", "real"):
        cfg.serial.enabled = True
        cfg.serial.mock = False

    configure_logging(cfg.logging)

    print("==========================================================================")
    print(" NEUROGRIP — PRODUCTION CONTROL GUI DASHBOARD (PHASE 3.5)")
    print("==========================================================================")
    print(f"  Camera Device Index : {args.camera}")
    print(f"  Serial Port         : {cfg.serial.port}")
    print(f"  Serial Mode         : {'REAL' if (cfg.serial.enabled and not cfg.serial.mock) else ('MOCK' if cfg.serial.enabled else 'DISABLED')}")
    print(f"  GUI Window Mode     : {'HEADLESS' if args.no_gui else 'DESKTOP WINDOW'}")
    print("==========================================================================")

    if args.no_gui:
        logger.info("Running pipeline in headless diagnostic mode...")
        pipeline = NeuroGripPipeline(config=cfg)
        pipeline.arm_stop()
        frames_processed = 0

        def cb(container, rendered, viz):
            nonlocal frames_processed
            frames_processed += 1
            if args.max_frames and frames_processed >= args.max_frames:
                pipeline.stop()

        pipeline.run_loop(max_frames=args.max_frames, callback=cb)
        print(f"Headless pipeline execution finished. Total frames: {frames_processed}")
        return 0

    # Launch Desktop GUI App
    app = NeuroGripDashboardApp(config=cfg)

    if args.max_frames and app.root:
        # Schedule auto-shutdown for diagnostic/automated test runs
        app.root.after(int(args.max_frames * 33.3), app.on_close)

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(run_gui_app())
