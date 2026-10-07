"""
tests/test_camera.py
─────────────────────
Unit tests for Camera Abstraction Layer.
Covers CameraConfig defaults, FrameContainer properties, MockCamera behavior,
and OpenCVCamera lifecycle (using mocked cv2.VideoCapture).
"""
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from neurogrip.camera.base import CameraInterface, CameraState, FrameContainer
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.config.settings import AppConfig, CameraConfig


# ─────────────────────────────────────────────────────────
# Configuration & FrameContainer Tests
# ─────────────────────────────────────────────────────────

def test_camera_config_defaults():
    """Verify CameraConfig default values."""
    cfg = CameraConfig()
    assert cfg.index == 0
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.fps == 30
    assert cfg.backend == "opencv"


def test_frame_container_validity():
    """Verify FrameContainer.is_valid property."""
    # Valid non-empty array
    valid_frame = FrameContainer(frame=np.zeros((720, 1280, 3), dtype=np.uint8), timestamp_ms=100, frame_index=0)
    assert valid_frame.is_valid is True
    assert valid_frame.frame_index == 0

    # None frame -> Invalid
    none_frame = FrameContainer(frame=None, timestamp_ms=100, frame_index=1)
    assert none_frame.is_valid is False

    # Zero-size array -> Invalid
    empty_frame = FrameContainer(frame=np.array([]), timestamp_ms=100, frame_index=2)
    assert empty_frame.is_valid is False


# ─────────────────────────────────────────────────────────
# MockCamera Tests
# ─────────────────────────────────────────────────────────

def test_mock_camera_lifecycle_and_delivery():
    """Test MockCamera connection, frame generation, timestamps, and closing."""
    cam = MockCamera()
    assert cam.is_opened is False
    assert cam.state == CameraState.DISCONNECTED

    # Open camera
    assert cam.open() is True
    assert cam.is_opened is True
    assert cam.state == CameraState.MOCK

    # Read frame 0
    f0 = cam.read_frame()
    assert f0.is_valid is True
    assert f0.frame_index == 0
    assert f0.frame.shape == (720, 1280, 3)  # Default config resolution

    # Read frame 1
    f1 = cam.read_frame()
    assert f1.is_valid is True
    assert f1.frame_index == 1
    assert f1.timestamp_ms >= f0.timestamp_ms

    # Close
    cam.close()
    assert cam.is_opened is False
    assert cam.state == CameraState.DISCONNECTED


def test_mock_camera_custom_frames():
    """Test MockCamera using pre-supplied custom frame sequences."""
    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)

    cam = MockCamera(frames=[img1, img2])
    cam.open()

    f0 = cam.read_frame()
    assert f0.frame[0, 0, 0] == 50

    f1 = cam.read_frame()
    assert f1.frame[0, 0, 0] == 200

    # Wraps around to image 0
    f2 = cam.read_frame()
    assert f2.frame[0, 0, 0] == 50


def test_mock_camera_end_of_stream():
    """Verify max_frames limit returns invalid frame simulating end-of-stream."""
    cam = MockCamera(max_frames=2)
    cam.open()

    f0 = cam.read_frame()
    assert f0.is_valid is True

    f1 = cam.read_frame()
    assert f1.is_valid is True

    # 3rd read exceeds max_frames=2 -> invalid frame (End of stream)
    f2 = cam.read_frame()
    assert f2.is_valid is False
    assert f2.frame is None


def test_mock_camera_simulated_read_failure():
    """Verify simulate_failure_at returns invalid frame for configured index."""
    cam = MockCamera(simulate_failure_at=1)
    cam.open()

    f0 = cam.read_frame()
    assert f0.is_valid is True
    assert f0.frame_index == 0

    # Frame 1 fails
    f1 = cam.read_frame()
    assert f1.is_valid is False
    assert f1.frame_index == 1

    # Frame 2 succeeds again
    f2 = cam.read_frame()
    assert f2.is_valid is True
    assert f2.frame_index == 2


def test_mock_camera_read_while_disconnected():
    """Verify reading while disconnected returns invalid FrameContainer."""
    cam = MockCamera()
    f = cam.read_frame()
    assert f.is_valid is False
    assert f.frame is None


# ─────────────────────────────────────────────────────────
# OpenCVCamera Tests (Mocked cv2.VideoCapture)
# ─────────────────────────────────────────────────────────

@patch("cv2.VideoCapture")
def test_opencv_camera_successful_open_and_read(mock_video_capture):
    """Test successful OpenCVCamera initialization, parameter setting, and frame reading."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    # Simulate get properties returning width=1280, height=720, fps=30.0
    def mock_get(prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    mock_cap.get.side_effect = mock_get

    # Simulate read frame returning synthetic array
    dummy_img = np.ones((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_img)

    mock_video_capture.return_value = mock_cap

    cfg = CameraConfig(index=0, width=1280, height=720, fps=30)
    cam = OpenCVCamera(config=cfg)

    # Open camera
    assert cam.open() is True
    assert cam.is_opened is True
    assert cam.state == CameraState.CONNECTED
    assert cam.actual_width == 1280
    assert cam.actual_height == 720
    assert cam.actual_fps == 30.0

    # Read frame
    container = cam.read_frame()
    assert container.is_valid is True
    assert container.frame_index == 0
    assert container.frame.shape == (720, 1280, 3)

    # Close camera
    cam.close()
    assert cam.is_opened is False
    assert cam.state == CameraState.DISCONNECTED
    mock_cap.release.assert_called_once()


@patch("cv2.VideoCapture")
def test_opencv_camera_open_failure(mock_video_capture):
    """Verify OpenCVCamera handles open failure safely and sets state ERROR."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False  # Camera failed to open
    mock_video_capture.return_value = mock_cap

    cam = OpenCVCamera()
    assert cam.open() is False
    assert cam.is_opened is False
    assert cam.state == CameraState.ERROR


@patch("cv2.VideoCapture")
def test_opencv_camera_frame_read_failure(mock_video_capture):
    """Verify OpenCVCamera handles frame read failure (ret=False) safely."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)  # Read failed
    mock_video_capture.return_value = mock_cap

    cam = OpenCVCamera()
    cam.open()

    container = cam.read_frame()
    assert container.is_valid is False
    assert container.frame is None


@patch("cv2.VideoCapture")
def test_opencv_camera_context_manager(mock_video_capture):
    """Verify OpenCVCamera context manager opens and releases resources cleanly."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_video_capture.return_value = mock_cap

    with OpenCVCamera() as cam:
        assert cam.is_opened is True

    assert cam.is_opened is False
    mock_cap.release.assert_called_once()


@patch("cv2.VideoCapture")
def test_opencv_camera_auto_selection_fallback(mock_video_capture):
    """Verify OpenCVCamera automatically selects index 1 when index 0 is disabled in Windows."""
    mock_cap_0 = MagicMock()
    mock_cap_0.isOpened.return_value = False  # Index 0 disabled

    mock_cap_1 = MagicMock()
    mock_cap_1.isOpened.return_value = True   # Index 1 enabled

    def mock_vc(index):
        if index == 0:
            return mock_cap_0
        elif index == 1:
            return mock_cap_1
        return mock_cap_0

    mock_video_capture.side_effect = mock_vc

    cfg = CameraConfig(index=0)
    cam = OpenCVCamera(config=cfg)

    assert cam.open() is True
    assert cam.is_opened is True
    assert cam.config.index == 1

