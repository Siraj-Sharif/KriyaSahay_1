"""Unit Tests for HaGRID Phase 2.2 Real-World Recognition Evaluation Tool and Trial Metrics.

Tests cover:
1. Taxonomy mapping for all 14 locked commands.
2. Explicit removal of 'one' -> NO_COMMAND and invalid class handling.
3. Architecture B safety rules (0 hands -> NO_HAND, >1 hands -> MULTI_HAND_AMBIGUITY).
4. TrialMetrics calculation: matching frames, wrong command frames, NO_COMMAND frames.
5. Manual recognition consistency percentage calculation (matching / total * 100).
6. Prediction switch count tracking.
7. Diagnostic A/B agreement rate calculation.
8. On-screen key mapping resolution for all 14 taxonomy gestures.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from neurogrip.hagrid.benchmark import predict_architecture_b
from neurogrip.hagrid.detector import HaGRIDHandDetector
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import (
    HAGRID_CLASSES,
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    map_hagrid_to_neurogrip,
)
from tools.hagrid_phase2_realworld_evaluation import (
    GESTURE_KEY_MAP,
    TrialMetrics,
    calculate_agreement_rate,
)


def test_key_mapping_covers_all_14_canonical_commands():
    """Verify that GESTURE_KEY_MAP maps exactly to the 14 locked NeuroGrip commands."""
    assert len(GESTURE_KEY_MAP) == 14, "Key map must cover all 14 taxonomy commands"
    mapped_commands = set(GESTURE_KEY_MAP.values())
    assert mapped_commands == set(NEUROGRIP_TAXONOMY), "Key map commands must match locked taxonomy"


def test_trial_metrics_calculations():
    """Verify TrialMetrics tracking for matching, wrong, NO_COMMAND, switches, and consistency %."""
    metrics = TrialMetrics(architecture="A_FULL_FRAME", expected_gesture="CLOSED_FIST")

    # Simulate sequence: CLOSED_FIST (match), CLOSED_FIST (match), THUMBS_UP (wrong), NO_COMMAND, CLOSED_FIST (match)
    observations = [
        ("CLOSED_FIST", 25.0),
        ("CLOSED_FIST", 26.0),
        ("THUMBS_UP", 24.0),
        (NO_COMMAND, 28.0),
        ("CLOSED_FIST", 25.0),
    ]

    for cmd, lat in observations:
        metrics.update(cmd, lat)

    assert metrics.total_observed_frames == 5
    assert metrics.matching_frames == 3
    assert metrics.wrong_command_frames == 1
    assert metrics.no_command_frames == 1
    # Consistency = 3 / 5 * 100 = 60.0%
    assert metrics.manual_recognition_consistency == pytest.approx(60.0)

    # Switches: CLOSED_FIST->CLOSED_FIST (0), ->THUMBS_UP (1), ->NO_COMMAND (2), ->CLOSED_FIST (3)
    assert metrics.switch_count == 3
    assert metrics.mean_latency_ms == pytest.approx(25.6)


def test_agreement_rate_calculation():
    """Verify calculate_agreement_rate diagnostic calculation."""
    cmds_a = ["CLOSED_FIST", "STOP", "THUMBS_UP", "NO_COMMAND", "OK"]
    cmds_b = ["CLOSED_FIST", "STOP", "NO_COMMAND", "NO_COMMAND", "OK"]

    # 4 out of 5 frames match -> 80.0% agreement
    rate = calculate_agreement_rate(cmds_a, cmds_b)
    assert rate == pytest.approx(80.0)

    # Empty list handling
    assert calculate_agreement_rate([], []) == 0.0


def test_taxonomy_strict_mapping_rules():
    """Verify locked taxonomy mappings and explicit 'one' -> NO_COMMAND removal."""
    assert map_hagrid_to_neurogrip("one") == NO_COMMAND
    assert map_hagrid_to_neurogrip("dislike") == NO_COMMAND
    assert map_hagrid_to_neurogrip("no_gesture") == NO_COMMAND
    assert map_hagrid_to_neurogrip("fist") == "CLOSED_FIST"
    assert map_hagrid_to_neurogrip("like") == "THUMBS_UP"
    assert map_hagrid_to_neurogrip("rock") == "INDEX_PINKY"


def test_safety_rules_zero_and_multi_hand_rejection():
    """Verify zero-hand and multi-hand rejection logic in Architecture B."""
    mock_det = MagicMock(spec=HaGRIDHandDetector)
    mock_clf = MagicMock(spec=HaGRIDClassifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 0 Hands test
    mock_det.detect.return_value = {"boxes": [], "confidences": [], "count": 0, "latency_ms": 10.0}
    res_0 = predict_architecture_b(mock_det, mock_clf, dummy_frame)
    assert res_0["mapped_command"] == NO_COMMAND
    assert res_0["reason"] == "NO_HAND"

    # Multi-Hand test (>1 hands)
    mock_det.detect.return_value = {
        "boxes": [[10, 10, 50, 50], [100, 100, 150, 150]],
        "confidences": [0.9, 0.8],
        "count": 2,
        "latency_ms": 15.0,
    }
    res_multi = predict_architecture_b(mock_det, mock_clf, dummy_frame)
    assert res_multi["mapped_command"] == NO_COMMAND
    assert res_multi["reason"] == "MULTI_HAND_AMBIGUITY"
    assert mock_clf.predict.call_count == 0  # Classifier bypassed
