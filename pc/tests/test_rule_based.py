"""
tests/test_rule_based.py
────────────────────────
Unit tests for RuleBasedRecognizer.
Verifies command classification for all 13 official ML commands, GRAB calibration,
THUMB_ONLY lateral extension rules, and STOP recognition.
"""
import numpy as np
import pytest

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import FingerStateDetector
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from tests.conftest import make_hand_landmarks


# ─────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────

@pytest.fixture
def recognizer():
    return RuleBasedRecognizer()


@pytest.fixture
def extractor():
    return FeatureExtractor()


def get_prediction(recognizer, extractor, landmarks, is_stop_armed=False):
    features = extractor.extract(landmarks)
    return recognizer.predict(features, hand_landmarks=landmarks, is_stop_armed=is_stop_armed)


# ─────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────

def test_rule_based_is_ready(recognizer):
    """Verify RuleBasedRecognizer is always ready."""
    assert recognizer.is_ready is True


def test_rule_based_none_landmarks(recognizer):
    """Verify rule-based recognizer handles None landmarks safely by returning UNKNOWN."""
    res = recognizer.predict(features=np.zeros(67, dtype=np.float32), hand_landmarks=None)
    assert res.label == "UNKNOWN"
    assert res.confidence == 0.0


def test_rule_based_open(recognizer, extractor, open_hand_landmarks):
    """Verify 5 fingers extended gesture returns STOP regardless of is_stop_armed flag."""
    res_armed = get_prediction(recognizer, extractor, open_hand_landmarks, is_stop_armed=True)
    assert res_armed.label == NeuroGripCommand.STOP.value
    assert res_armed.confidence == 1.0

    res_disarmed = get_prediction(recognizer, extractor, open_hand_landmarks, is_stop_armed=False)
    assert res_disarmed.label == NeuroGripCommand.STOP.value
    assert res_disarmed.confidence == 1.0


def test_rule_based_close(recognizer, extractor, closed_hand_landmarks):
    """Verify CLOSE gesture prediction when all 5 fingers are fully closed fist."""
    res = get_prediction(recognizer, extractor, closed_hand_landmarks, is_stop_armed=False)
    assert res.label == NeuroGripCommand.CLOSE.value
    assert res.confidence == 1.0


def test_rule_based_index_only(recognizer, extractor, index_only_landmarks):
    """Verify INDEX gesture prediction when only index finger is extended."""
    res = get_prediction(recognizer, extractor, index_only_landmarks, is_stop_armed=False)
    assert res.label == NeuroGripCommand.INDEX.value
    assert res.confidence == 1.0


def test_rule_based_middle_only(recognizer, extractor):
    """Verify MIDDLE gesture prediction when only middle finger is extended."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, 0.05),
            "middle": (0.0, -0.27),
            "ring": (0.0, 0.05),
            "pinky": (0.0, 0.05),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.MIDDLE.value
    assert res.confidence == 1.0


def test_rule_based_ring_only(recognizer, extractor):
    """Verify RING gesture prediction when only ring finger is extended."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, 0.05),
            "middle": (0.0, 0.05),
            "ring": (0.0, -0.25),
            "pinky": (0.0, 0.05),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.RING.value
    assert res.confidence == 1.0


def test_rule_based_pinky_only(recognizer, extractor):
    """Verify PINKY gesture prediction when only pinky finger is extended."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, 0.05),
            "middle": (0.0, 0.05),
            "ring": (0.0, 0.05),
            "pinky": (0.0, -0.22),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.PINKY.value
    assert res.confidence == 1.0


def test_rule_based_thumb_only(recognizer, extractor, thumb_only_landmarks):
    """Verify THUMB_ONLY gesture prediction when thumb is laterally extended."""
    res = get_prediction(recognizer, extractor, thumb_only_landmarks, is_stop_armed=False)
    assert res.label == NeuroGripCommand.THUMB_ONLY.value
    assert res.confidence == 1.0


def test_rule_based_two_finger(recognizer, extractor):
    """Verify TWO_FINGER gesture prediction (Index + Middle extended)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, 0.05),
            "pinky": (0.0, 0.05),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.TWO_FINGER.value
    assert res.confidence == 1.0


def test_rule_based_three_finger(recognizer, extractor):
    """Verify THREE_FINGER gesture prediction (Index + Middle + Ring extended)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, 0.05),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.THREE_FINGER.value
    assert res.confidence == 1.0


def test_rule_based_index_pinky(recognizer, extractor):
    """Verify INDEX_PINKY gesture prediction (Index + Pinky extended)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, 0.05),
            "ring": (0.0, 0.05),
            "pinky": (0.0, -0.22),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.INDEX_PINKY.value
    assert res.confidence == 1.0


def test_rule_based_four_fingers(recognizer, extractor):
    """Verify FOUR_FINGERS gesture prediction (Four non-thumb fingers extended)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value
    assert res.confidence == 1.0


def test_rule_based_grab(recognizer, extractor):
    """Verify GRAB gesture prediction for partial-curl claw posture."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.08),
            "middle": (0.0, -0.08),
            "ring": (0.0, -0.08),
            "pinky": (0.0, -0.08),
        }
    )
    feats_grab = np.zeros(67, dtype=np.float32)
    feats_grab[57:62] = 0.35  # Extension ratios in [0.20, 0.55]
    feats_grab[47:57] = 0.40  # Joint cosines in [0.10, 0.60]
    res = recognizer.predict(feats_grab, hand_landmarks=landmarks)
    assert res.label == NeuroGripCommand.GRAB.value
    assert res.confidence == 0.90


def test_rule_based_rest(recognizer, extractor):
    """Verify REST gesture prediction for relaxed idle hand posture."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.08),
            "middle": (0.0, -0.08),
            "ring": (0.0, -0.08),
            "pinky": (0.0, -0.08),
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.REST.value
    assert res.confidence == 0.70


def test_rule_based_stop_recognition_independent_of_arming(recognizer, extractor, open_hand_landmarks):
    """Verify STOP recognition is independent of is_stop_armed at RuleBasedRecognizer level."""
    # 1. Unarmed -> Returns STOP
    res_unarmed = get_prediction(recognizer, extractor, open_hand_landmarks, is_stop_armed=False)
    assert res_unarmed.label == NeuroGripCommand.STOP.value
    assert res_unarmed.confidence == 1.0

    # 2. Armed -> Returns STOP
    res_armed = get_prediction(recognizer, extractor, open_hand_landmarks, is_stop_armed=True)
    assert res_armed.label == NeuroGripCommand.STOP.value
    assert res_armed.confidence == 1.0


# ─────────────────────────────────────────────────────────
# Regression Tests for Loose / UNKNOWN Non-Target Fingers
# ─────────────────────────────────────────────────────────

def test_rule_based_three_finger_loose_pinky(recognizer, extractor):
    """Regression test: THREE_FINGER classification succeeds when pinky is UNKNOWN (loose pinky)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),    # CLOSED
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, -0.25),    # OPEN
            "pinky": (0.0, -0.08),   # UNKNOWN (loose pinky)
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.THREE_FINGER.value
    assert res.confidence == 1.0


def test_rule_based_four_fingers_loose_thumb(recognizer, extractor):
    """Regression test: FOUR_FINGERS classification succeeds when thumb is UNKNOWN (tucked/loose thumb)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.05, -0.08),   # UNKNOWN (loose/tucked thumb)
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, -0.25),    # OPEN
            "pinky": (0.0, -0.22),   # OPEN
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value
    assert res.confidence == 1.0


def test_rule_based_index_pinky_loose_middle(recognizer, extractor):
    """Regression test: INDEX_PINKY classification succeeds when middle finger is UNKNOWN (loose middle)."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),    # CLOSED
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.08),  # UNKNOWN (loose middle)
            "ring": (0.0, 0.05),     # CLOSED
            "pinky": (0.0, -0.22),   # OPEN
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label == NeuroGripCommand.INDEX_PINKY.value
    assert res.confidence == 1.0


def test_rule_based_index_pinky_prevents_rest_fallthrough(recognizer, extractor):
    """Regression test: Proves INDEX_PINKY with a loose middle finger does NOT return REST."""
    landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),    # CLOSED
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.08),  # UNKNOWN (loose middle)
            "ring": (0.0, 0.05),     # CLOSED
            "pinky": (0.0, -0.22),   # OPEN
        }
    )
    res = get_prediction(recognizer, extractor, landmarks)
    assert res.label != NeuroGripCommand.REST.value
    assert res.label == NeuroGripCommand.INDEX_PINKY.value


