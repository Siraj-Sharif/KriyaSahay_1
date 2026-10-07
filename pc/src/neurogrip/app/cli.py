"""
neurogrip.app.cli
-----------------
Production command-line entry point for the NeuroGrip PC-side application.

The CLI owns application lifecycle concerns only. All frame-processing logic
remains inside NeuroGripPipeline.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.visualization.overlay import OverlayWindow

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the NeuroGrip application argument parser."""
    parser = argparse.ArgumentParser(
        prog="neurogrip",
        description="NeuroGrip PC-side real-time hand gesture control system.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (default: built-in config/default.yaml).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without opening the OpenCV visualization window.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the dedicated Tkinter Production Dashboard GUI.",
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
        default=None,
        help="Serial communication baud rate (e.g. 115200).",
    )
    parser.add_argument(
        "--serial",
        type=str,
        choices=["enabled", "disabled", "mock"],
        default=None,
        help="Serial transport mode: 'enabled' (hardware serial), 'disabled', or 'mock'.",
    )
    parser.add_argument(
        "--serial-disabled",
        action="store_true",
        help="Disable serial hardware communication mode.",
    )
    return parser


def _load_config(
    config_path: Path | None,
    camera_idx: int = 0,
    serial_port: str | None = None,
    baudrate: int | None = None,
    serial_mode: str | None = None,
    serial_disabled: bool = False,
) -> AppConfig:
    """Load either the supplied YAML file or the project's default config."""
    if config_path is None:
        cfg = AppConfig.default()
    else:
        cfg = AppConfig.from_yaml(config_path)
    cfg.camera.index = camera_idx

    if serial_port is not None:
        cfg.serial.port = serial_port
    if baudrate is not None:
        cfg.serial.baud_rate = baudrate

    if serial_disabled or serial_mode == "disabled":
        cfg.serial.enabled = False
        cfg.serial.mock = True
    elif serial_mode == "mock":
        cfg.serial.enabled = True
        cfg.serial.mock = True
    elif serial_mode == "enabled":
        cfg.serial.enabled = True
        cfg.serial.mock = False

    return cfg



def run(config: AppConfig, *, no_gui: bool = False, use_gui: bool = False) -> int:
    """
    Run the NeuroGrip application using an already-loaded configuration.

    Returns a process-style exit code. Pipeline processing and component
    lifecycle remain delegated to NeuroGripPipeline.
    """
    if use_gui:
        from neurogrip.gui.dashboard import NeuroGripDashboardApp
        app = NeuroGripDashboardApp(config=config)
        app.run()
        return 0

    if no_gui:
        # Do not mutate the caller's configuration; make a shallow dataclass
        # replacement for the visualization section only.
        import dataclasses

        config = dataclasses.replace(
            config,
            visualization=dataclasses.replace(config.visualization, enabled=False),
        )

    pipeline = NeuroGripPipeline(config=config)
    overlay = OverlayWindow(config.visualization) if config.visualization.enabled else None

    def on_frame(frame_container, rendered, state) -> None:
        # rendered is intentionally available from the pipeline for headless
        # consumers. GUI display is the only responsibility of this callback.
        if overlay is None:
            return

        frame = frame_container.frame
        if frame is None:
            return

        key = overlay.show(frame, state)
        if key in (ord("q"), 27):
            logger.info("Exit requested by user.")
            pipeline.stop()
        elif key == ord(" "):
            pipeline.toggle_stop_arm()

    try:
        logger.info("Starting NeuroGrip CV application...")
        if not pipeline.initialize():
            logger.error("NeuroGrip initialization failed.")
            return 1

        logger.info("NeuroGrip ready. Press Q or ESC to exit.")
        if config.stop.arm_key.lower() == "space" and overlay is not None:
            logger.info("Press SPACE to arm/disarm software STOP detection.")

        pipeline.run_loop(callback=on_frame)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        pipeline.stop()
        return 0
    except Exception:
        logger.exception("Unhandled NeuroGrip application error.")
        return 1
    finally:
        if overlay is not None:
            overlay.close()
        # run_loop already shuts down in its finally block. This second call is
        # intentionally idempotent and protects early returns/exceptions.
        pipeline.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_config(
        args.config,
        camera_idx=args.camera,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        serial_mode=args.serial,
        serial_disabled=args.serial_disabled,
    )
    configure_logging(config.logging)
    return run(config, no_gui=args.no_gui, use_gui=getattr(args, "gui", False))



if __name__ == "__main__":
    raise SystemExit(main())
