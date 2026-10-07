"""Unit Tests for HaGRID Phase 2.3 Raw Prediction Diagnostic Tool.

Tests cover:
1. Top-5 prediction sorting in descending order of probability.
2. Preservation of raw HaGRID class names.
3. Availability of raw top-1 class before taxonomy mapping.
4. Target gesture expected raw class specifications.
5. Summary statistics calculation (hit rate %, confidences, confusions).
6. Raw prediction switch count tracking.
7. Full floating-point probability preservation in JSON logging.
8. Non-mutation of production configuration.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from neurogrip.hagrid.config import HaGRIDConfig
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY, NO_COMMAND
from tools.hagrid_raw_prediction_diagnostic import (
    KEY_TO_GESTURE_KEY,
    TARGET_GESTURES,
    DiagnosticFrameRecord,
    compute_raw_predictions_unfiltered,
    compute_trial_summary_stats,
)


def test_target_gesture_expected_raw_class_specifications():
    """Verify target gestures map strictly to specified official HaGRID raw classes."""
    assert len(TARGET_GESTURES) == 14
    assert TARGET_GESTURES["INDEX_FINGER"]["expected_raw"] == ["point"]
    assert TARGET_GESTURES["TWO_FINGERS"]["expected_raw"] == ["two_up", "two_up_inverted"]
    assert TARGET_GESTURES["PINKY"]["expected_raw"] == ["little_finger"]
    assert TARGET_GESTURES["INDEX_FINGER"]["key"] == "0"
    assert TARGET_GESTURES["TWO_FINGERS"]["key"] == "["
    assert TARGET_GESTURES["PINKY"]["key"] == "7"


def test_all_14_canonical_gestures_and_key_mappings():
    """Verify all 14 canonical gestures are present and KEY_TO_GESTURE_KEY maps primary and alt keys."""
    for canonical in NEUROGRIP_TAXONOMY:
        assert canonical in TARGET_GESTURES, f"{canonical} must be present in TARGET_GESTURES"

    assert KEY_TO_GESTURE_KEY["1"] == "CALL"
    assert KEY_TO_GESTURE_KEY["2"] == "CLOSED_FIST"
    assert KEY_TO_GESTURE_KEY["3"] == "FOUR_FINGERS"
    assert KEY_TO_GESTURE_KEY["4"] == "GRABBING"
    assert KEY_TO_GESTURE_KEY["5"] == "GRIP"
    assert KEY_TO_GESTURE_KEY["6"] == "THUMBS_UP"
    assert KEY_TO_GESTURE_KEY["7"] == "PINKY"
    assert KEY_TO_GESTURE_KEY["8"] == "MIDDLE_FINGER"
    assert KEY_TO_GESTURE_KEY["9"] == "OK"
    assert KEY_TO_GESTURE_KEY["0"] == "INDEX_FINGER"
    assert KEY_TO_GESTURE_KEY["-"] == "INDEX_PINKY"
    assert KEY_TO_GESTURE_KEY["="] == "STOP"
    assert KEY_TO_GESTURE_KEY["["] == "TWO_FINGERS"
    assert KEY_TO_GESTURE_KEY["]"] == "THREE_FINGERS"

    # Test shift key variants
    assert KEY_TO_GESTURE_KEY[")"] == "INDEX_FINGER"
    assert KEY_TO_GESTURE_KEY["{"] == "TWO_FINGERS"
    assert KEY_TO_GESTURE_KEY["&"] == "PINKY"


def test_top5_prediction_sorting_and_unfiltered_probabilities():
    """Verify top-5 predictions are sorted descending and retain raw float probabilities without cutoff."""
    mock_classifier = MagicMock(spec=HaGRIDClassifier)
    mock_classifier.model = MagicMock()

    # Mock logits tensor of shape (1, 34)
    dummy_logits = torch.zeros((1, 34), dtype=torch.float32)
    # Set high logit for index 3 (point = 0.60) and index 19 (one = 0.20)
    dummy_logits[0, 3] = 3.0   # point
    dummy_logits[0, 19] = 2.0  # one
    dummy_logits[0, 23] = 1.0  # rock

    mock_classifier.model.return_value = dummy_logits
    mock_classifier.preprocess.return_value = torch.zeros((1, 3, 224, 224))

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = compute_raw_predictions_unfiltered(mock_classifier, dummy_frame, top_k=5)

    assert "raw_top1_class" in res
    assert "raw_top1_probability" in res
    assert "top5" in res
    assert len(res["top5"]) == 5

    # Verify top-1 is 'point'
    assert res["raw_top1_class"] == "point"
    assert res["canonical_command"] == "INDEX_FINGER"

    # Verify descending probability order
    probs = [p["probability"] for p in res["top5"]]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], "Top-5 probabilities must be sorted descending"

    # Verify un-gated float precision (no rounding to 0 or premature threshold cutoff)
    assert isinstance(res["raw_top1_probability"], float)
    assert 0.0 < res["raw_top1_probability"] <= 1.0


def test_summary_stats_hit_rate_and_confusions():
    """Verify compute_trial_summary_stats hit rate, switch count, and confusion calculations."""
    frames = [
        DiagnosticFrameRecord(0, "ts1", "point", 0.70, [], "INDEX_FINGER"),
        DiagnosticFrameRecord(1, "ts2", "point", 0.65, [], "INDEX_FINGER"),
        DiagnosticFrameRecord(2, "ts3", "one", 0.40, [], NO_COMMAND),
        DiagnosticFrameRecord(3, "ts4", "rock", 0.35, [], "INDEX_PINKY"),
        DiagnosticFrameRecord(4, "ts5", "point", 0.80, [], "INDEX_FINGER"),
    ]

    stats = compute_trial_summary_stats("INDEX_FINGER", ["point"], frames)

    assert stats["total_frames"] == 5
    # 3 out of 5 frames had raw_top1_class == 'point' -> 60.0%
    assert stats["hit_rate_pct"] == pytest.approx(60.0)
    assert stats["unsupported_pct"] == pytest.approx(40.0)

    # Switches: point->point (0), ->one (1), ->rock (2), ->point (3)
    assert stats["switch_count"] == 3

    # Confusions should list 'one' (20%) and 'rock' (20%)
    top_conf_dict = dict(stats["top_confusions"])
    assert "one" in top_conf_dict
    assert "rock" in top_conf_dict


def test_summary_stats_two_fingers_multi_raw_hit_rate():
    """Verify TWO_FINGERS counts both two_up and two_up_inverted as hits."""
    frames = [
        DiagnosticFrameRecord(0, "ts1", "two_up", 0.80, [], "TWO_FINGERS"),
        DiagnosticFrameRecord(1, "ts2", "two_up_inverted", 0.75, [], "TWO_FINGERS"),
        DiagnosticFrameRecord(2, "ts3", "three", 0.30, [], "THREE_FINGERS"),
    ]

    stats = compute_trial_summary_stats("TWO_FINGERS", ["two_up", "two_up_inverted"], frames)

    # 2 out of 3 frames hit expected raw -> 66.67%
    assert stats["hit_rate_pct"] == pytest.approx(66.67, rel=1e-2)


def test_json_serialization_preserves_full_float_precision():
    """Verify JSON dumping retains full float precision without loss of data."""
    frame = DiagnosticFrameRecord(
        frame_index=0,
        timestamp="2026-09-14T12:00:00",
        raw_top1_class="point",
        raw_top1_probability=0.384721123456789,
        top5=[{"class": "point", "probability": 0.384721123456789}],
        canonical_command="INDEX_FINGER",
    )

    data = [frame.__dict__]
    json_str = json.dumps(data)

    # Verify probability string contains full decimal representation
    assert "0.384721123456789" in json_str

    parsed = json.loads(json_str)
    assert parsed[0]["raw_top1_probability"] == pytest.approx(0.384721123456789)


def test_no_mutation_of_production_config():
    """Verify HaGRIDConfig defaults remain unchanged."""
    config = HaGRIDConfig()
    assert config.model_name == "ResNet18"
    assert config.confidence_threshold == 0.5
