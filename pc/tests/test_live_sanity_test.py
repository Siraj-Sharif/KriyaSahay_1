"""
tests/test_live_sanity_test.py
───────────────────────────────
Unit tests for live_sanity_test.py CLI script.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)
from scripts.live_sanity_test import create_parser, run_sanity_test, main


def test_live_sanity_test_cli_parser_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.no_gui is False
    assert args.mock_camera is False
    assert args.max_frames is None


def test_live_sanity_test_execution_mock_camera():
    mock_cam = MockCamera()
    ret = run_sanity_test(
        no_gui=True,
        mock_camera=True,
        max_frames=5,
        camera_override=mock_cam,
    )
    assert ret == 0


def test_live_sanity_test_main_cli_args():
    ret = main(["--no-gui", "--mock-camera", "--max-frames", "3"])
    assert ret == 0
