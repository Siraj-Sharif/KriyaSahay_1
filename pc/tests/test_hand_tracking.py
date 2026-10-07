"""
tests/test_hand_tracking.py
────────────────────────────
Unit tests for hand tracking data structures and detector wrapper.
Uses synthetic landmarks and mocked MediaPipe API calls (no webcam or model download required).
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from neurogrip.config.settings import HandTrackingConfig
from neurogrip.hand_tracking.detector import HandDetector, HandDetectorError
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_21_landmarks():
    """Generate 21 synthetic normalized landmarks."""
    return [NormalizedLandmark(x=i * 0.04, y=i * 0.03, z=i * 0.01) for i in range(21)]


# ─────────────────────────────────────────────────────────
# Landmarks Data Structures Tests
# ─────────────────────────────────────────────────────────

def test_handedness_parsing():
    """Test Handedness string parsing logic."""
    assert Handedness.from_str("Left") == Handedness.LEFT
    assert Handedness.from_str("RIGHT") == Handedness.RIGHT
    assert Handedness.from_str("left_hand") == Handedness.LEFT
    assert Handedness.from_str("unknown") == Handedness.UNKNOWN
    assert Handedness.from_str("") == Handedness.UNKNOWN


def test_hand_landmarks_valid(synthetic_21_landmarks):
    """Test creating a valid 21-landmark HandLandmarks object."""
    hl = HandLandmarks(landmarks=synthetic_21_landmarks, handedness=Handedness.RIGHT, score=0.95)
    assert len(hl.landmarks) == 21
    assert hl.handedness == Handedness.RIGHT
    assert hl.score == 0.95
    assert hl.world_landmarks is None


def test_hand_landmarks_invalid_count():
    """Test that HandLandmarks rejects landmark lists that do not contain exactly 21 points."""
    with pytest.raises(ValueError, match="requires exactly 21 landmarks"):
        HandLandmarks(landmarks=[NormalizedLandmark(x=0, y=0, z=0)] * 10)


def test_detection_result_helpers(synthetic_21_landmarks):
    """Test DetectionResult helper properties for single-hand, multi-hand, and no-hand states."""
    hl1 = HandLandmarks(landmarks=synthetic_21_landmarks, handedness=Handedness.RIGHT)
    hl2 = HandLandmarks(landmarks=synthetic_21_landmarks, handedness=Handedness.LEFT)

    # 1. Empty result
    res_empty = DetectionResult(hands=[], num_hands=0)
    assert res_empty.is_empty is True
    assert res_empty.has_hand is False
    assert res_empty.is_ambiguous is False
    assert res_empty.primary_hand is None

    # 2. Single hand result
    res_single = DetectionResult(hands=[hl1], num_hands=1)
    assert res_single.is_empty is False
    assert res_single.has_hand is True
    assert res_single.is_ambiguous is False
    assert res_single.primary_hand == hl1

    # 3. Multi-hand (ambiguous) result
    res_multi = DetectionResult(hands=[hl1, hl2], num_hands=2)
    assert res_multi.is_empty is False
    assert res_multi.has_hand is False
    assert res_multi.is_ambiguous is True
    assert res_multi.primary_hand is None


# ─────────────────────────────────────────────────────────
# HandDetector Tests
# ─────────────────────────────────────────────────────────

def test_hand_detector_missing_model_file():
    """Verify HandDetector raises HandDetectorError if the model file is not found."""
    cfg = HandTrackingConfig(model_path="non_existent_path/hand_landmarker.task")
    detector = HandDetector(config=cfg)
    with pytest.raises(HandDetectorError, match="model file not found"):
        detector.initialize()


@patch("pathlib.Path.exists", return_value=True)
@patch("neurogrip.hand_tracking.detector.vision.HandLandmarker.create_from_options")
def test_hand_detector_initialization(mock_create, mock_exists):
    """Verify HandDetector initializes MediaPipe HandLandmarker with correct options."""
    detector = HandDetector()
    detector.initialize()
    assert detector.is_initialized is True
    assert mock_create.called is True


@patch("pathlib.Path.exists", return_value=True)
@patch("neurogrip.hand_tracking.detector.vision.HandLandmarker.create_from_options")
def test_hand_detector_detect_empty_frame(mock_create, mock_exists):
    """Verify HandDetector handles empty or zero-size frames safely."""
    detector = HandDetector()
    detector.initialize()

    res = detector.detect(frame=np.array([]), timestamp_ms=100)
    assert res.is_empty is True
    assert res.timestamp_ms == 100


@patch("pathlib.Path.exists", return_value=True)
@patch("neurogrip.hand_tracking.detector.vision.HandLandmarker.create_from_options")
def test_hand_detector_convert_result(mock_create, mock_exists):
    """Verify result conversion from MediaPipe HandLandmarkerResult to DetectionResult."""
    detector = HandDetector()
    detector.initialize()

    # Mock MediaPipe landmark object
    mock_lm = MagicMock()
    mock_lm.x = 0.5
    mock_lm.y = 0.5
    mock_lm.z = 0.0

    mock_category = MagicMock()
    mock_category.category_name = "Right"
    mock_category.score = 0.98

    mock_mp_result = MagicMock()
    mock_mp_result.hand_landmarks = [[mock_lm] * 21]
    mock_mp_result.handedness = [[mock_category]]
    mock_mp_result.hand_world_landmarks = None

    converted = detector._convert_result(mock_mp_result, timestamp_ms=500)
    assert converted.num_hands == 1
    assert converted.has_hand is True
    assert converted.timestamp_ms == 500
    hand = converted.primary_hand
    assert hand is not None
    assert hand.handedness == Handedness.RIGHT
    assert hand.score == 0.98
    assert len(hand.landmarks) == 21
    assert hand.landmarks[0].x == 0.5


@patch("pathlib.Path.exists", return_value=True)
@patch("neurogrip.hand_tracking.detector.vision.HandLandmarker.create_from_options")
def test_hand_detector_context_manager(mock_create, mock_exists):
    """Verify HandDetector context manager opens and closes landmarker cleanly."""
    mock_landmarker_instance = MagicMock()
    mock_create.return_value = mock_landmarker_instance

    with HandDetector() as detector:
        assert detector.is_initialized is True

    assert detector.is_initialized is False
    assert mock_landmarker_instance.close.called is True


def test_resolve_model_path():
    """Verify resolve_model_path handles absolute paths and relative pc/ directory fallbacks."""
    from pathlib import Path
    from neurogrip.config.settings import resolve_model_path

    # Absolute path test
    abs_p = Path("/tmp/fake_model.task").resolve()
    assert resolve_model_path(abs_p) == abs_p

    # Existing model test: "models/mediapipe/hand_landmarker.task"
    resolved = resolve_model_path("models/mediapipe/hand_landmarker.task")
    assert resolved.name == "hand_landmarker.task"
    assert resolved.exists()

