"""
tests/test_pretrained_gesture.py
───────────────────────────────────
Isolated Unit Tests for Phase 10K Pretrained Gesture Recognizer Adapter & Mapping Layer.
"""
from pathlib import Path
import numpy as np
import pytest

from neurogrip.recognition.pretrained_gesture import (
    HaGRIDNeuroGripMapper,
    PretrainedGestureRecognizer,
    PretrainedRecognitionResult,
)


def test_hagrid_neurogrip_mapper_direct():
    """Verify direct correspondence mapping from HaGRID/MediaPipe to NeuroGrip commands."""
    cmd, corr = HaGRIDNeuroGripMapper.map_class("Pointing_Up")
    assert cmd == "INDEX"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("Victory")
    assert cmd == "TWO_FINGER"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("three")
    assert cmd == "THREE_FINGER"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("four")
    assert cmd == "FOUR_FINGERS"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("Closed_Fist")
    assert cmd == "CLOSE"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("Unrecognized")
    assert cmd == "REST"
    assert corr == "DIRECT"


def test_hagrid_neurogrip_mapper_approximate():
    """Verify approximate correspondence mapping."""
    cmd, corr = HaGRIDNeuroGripMapper.map_class("Thumb_Up")
    assert cmd == "THUMB_ONLY"
    assert corr == "APPROXIMATE"

    cmd, corr = HaGRIDNeuroGripMapper.map_class("ILoveYou")
    assert cmd == "INDEX_PINKY"
    assert corr == "APPROXIMATE"


def test_hagrid_neurogrip_mapper_unsupported():
    """Verify unsupported gestures map to UNKNOWN with NONE correspondence."""
    cmd, corr = HaGRIDNeuroGripMapper.map_class("RandomUnknownGesture")
    assert cmd == "UNKNOWN"
    assert corr == "NONE"


def test_pretrained_recognizer_initialization():
    """Test PretrainedGestureRecognizer lifecycle and properties."""
    recognizer = PretrainedGestureRecognizer()
    assert recognizer.is_ready is True

    # Test empty frame prediction
    empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = recognizer.predict_frame(empty_frame, timestamp_ms=100)

    assert isinstance(res, PretrainedRecognitionResult)
    assert res.is_hand_detected is False or res.raw_pretrained_class in ("No_Hand", "Unrecognized")
    assert res.mapped_neurogrip_command in ("UNKNOWN", "REST")

    recognizer.close()
    assert recognizer.is_ready is False
