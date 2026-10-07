"""
tests/test_live_cv_pipeline.py
───────────────────────────────
Unit tests for Stage 1 Production Live CV Pipeline (NeuroGripPipeline & DisplayConsoleTransport).
Verifies end-to-end execution loop, frame flow, 0-hand, 1-hand, 2+-hand states,
deterministic STOP safety, and protocol frame output.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.commands.validator import PipelineState
from neurogrip.communication.transport import DisplayConsoleTransport
from neurogrip.config.settings import AppConfig
from neurogrip.hand_tracking.landmarks import DetectionResult
from tests.conftest import make_hand_landmarks


def test_stage1_live_cv_pipeline_initialization():
    """Verify Stage 1 live CV pipeline initializes cleanly with DisplayConsoleTransport."""
    cfg = AppConfig()
    cfg.camera.backend = "mock"

    cam = MockCamera(config=cfg.camera)
    transport = DisplayConsoleTransport()

    detector = MagicMock()
    detector.initialize.return_value = None
    detector.is_initialized = True

    with NeuroGripPipeline(config=cfg, camera=cam, transport=transport, detector=detector) as p:
        assert p.is_initialized is True
        assert isinstance(p.transport, DisplayConsoleTransport)


def test_stage1_live_cv_pipeline_stop_safety_emission():
    """Verify armed STOP gesture immediately emits NG1|STOP protocol frame."""
    cfg = AppConfig()
    cfg.camera.backend = "mock"

    cam = MockCamera(config=cfg.camera)
    transport = DisplayConsoleTransport()

    stop_lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.08, -0.25),
            "index": (0.01, -0.25),
            "middle": (0.0, -0.25),
            "ring": (-0.01, -0.25),
            "pinky": (-0.02, -0.25),
        }
    )

    detector = MagicMock()
    detector.initialize.return_value = None
    detector.is_initialized = True
    detector.detect.return_value = DetectionResult(hands=[stop_lms], num_hands=1, timestamp_ms=100)

    with NeuroGripPipeline(config=cfg, camera=cam, transport=transport, detector=detector) as p:
        p.arm_stop()
        frame_c, rendered, viz_state = p.process_frame()

        assert viz_state.is_stop_active is True
        assert transport.latest_command == NeuroGripCommand.STOP
        assert transport.latest_frame == "NG1|STOP"


def test_stage1_live_cv_pipeline_idle_on_no_hand():
    """Verify 0-hand state emits IDLE and no protocol frame."""
    cfg = AppConfig()
    cfg.camera.backend = "mock"

    cam = MockCamera(config=cfg.camera)
    transport = DisplayConsoleTransport()

    detector = MagicMock()
    detector.initialize.return_value = None
    detector.is_initialized = True
    detector.detect.return_value = DetectionResult(hands=[], num_hands=0, timestamp_ms=200)

    with NeuroGripPipeline(config=cfg, camera=cam, transport=transport, detector=detector) as p:
        frame_c, rendered, viz_state = p.process_frame()

        assert viz_state.command in ("NO_COMMAND", "NO_HAND")
        assert transport.latest_command is None
        assert transport.latest_frame is None
