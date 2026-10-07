"""
pc/tests/test_phase10h_recognition_fixes.py
─────────────────────────────────────────────
Comprehensive unit tests for Phase 10H Recognition & Validation Fixes.
Verifies robust recognition for relaxed-thumb gestures, 3D landmark geometry,
handedness invariance, palm orientation filtering, and strict validator ground-truth alignment.
"""

import numpy as np
import pytest

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import FingerState, FingerStateDetector, FingerStateVector
from neurogrip.hand_tracking.detector import DetectionResult
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import RecognitionResult
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from scripts.collect_dataset import DatasetSampleValidator
from tests.conftest import make_hand_landmarks


@pytest.fixture
def recognizer():
    return RuleBasedRecognizer()


@pytest.fixture
def extractor():
    return FeatureExtractor()


def test_two_finger_with_relaxed_thumb(recognizer, extractor):
    """1. TWO_FINGER palm-facing pose with relaxed/ambiguous thumb."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.10, -0.05),   # Relaxed thumb (lateral extension low)
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, 0.05),     # CLOSED
            "pinky": (0.0, 0.05),    # CLOSED
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.TWO_FINGER.value


def test_three_finger_with_relaxed_thumb(recognizer, extractor):
    """2. THREE_FINGER palm-facing pose with relaxed/ambiguous thumb."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.08, -0.05),   # Relaxed thumb
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, -0.25),    # OPEN
            "pinky": (0.0, 0.05),    # CLOSED
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.THREE_FINGER.value


def test_four_fingers_with_relaxed_thumb(recognizer, extractor):
    """3. FOUR_FINGERS palm-facing pose with relaxed/tucked thumb does NOT become STOP."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.05, -0.05),   # Tucked/relaxed thumb
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, -0.25),    # OPEN
            "pinky": (0.0, -0.22),   # OPEN
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value


def test_stop_with_clearly_open_thumb(recognizer, extractor):
    """4. STOP requires all 5 fingers clearly extended with wide thumb spread."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.22, -0.15),   # Wide open thumb spread
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, -0.27),  # OPEN
            "ring": (0.0, -0.25),    # OPEN
            "pinky": (0.0, -0.22),   # OPEN
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.STOP.value


def test_index_pinky_with_relaxed_thumb(recognizer, extractor):
    """5. INDEX_PINKY palm-facing pose with relaxed thumb."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.05, -0.05),   # Relaxed thumb
            "index": (0.0, -0.25),   # OPEN
            "middle": (0.0, 0.05),   # CLOSED
            "ring": (0.0, 0.05),     # CLOSED
            "pinky": (0.0, -0.22),   # OPEN
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.INDEX_PINKY.value


def test_handedness_invariance(recognizer, extractor):
    """6. Left and right hand poses yield identical recognition labels."""
    offsets = {
        "thumb": (0.08, -0.05),
        "index": (0.0, -0.25),
        "middle": (0.0, -0.27),
        "ring": (0.0, 0.05),
        "pinky": (0.0, 0.05),
    }
    lms_right = make_hand_landmarks(tip_offsets=offsets, handedness=Handedness.RIGHT)
    lms_left = make_hand_landmarks(tip_offsets=offsets, handedness=Handedness.LEFT)

    res_right = recognizer.predict(extractor.extract(lms_right), hand_landmarks=lms_right)
    res_left = recognizer.predict(extractor.extract(lms_left), hand_landmarks=lms_left)

    assert res_right.label == res_left.label == NeuroGripCommand.TWO_FINGER.value


def test_sideways_pose_rejection():
    """7. Strongly sideways 90-degree edge-on hand is rejected as UNKNOWN."""
    detector = FingerStateDetector()
    # Construct artificial landmarks where x-coordinates align edge-on
    lms_data = [NormalizedLandmark(x=0.5, y=0.8, z=0.0)]
    for i in range(1, 21):
        lms_data.append(NormalizedLandmark(x=0.5, y=0.8 - (i * 0.02), z=float(i) * 0.05))
    hlms = HandLandmarks(landmarks=lms_data, handedness=Handedness.RIGHT)

    states = detector.detect(hlms)
    assert states.index == FingerState.UNKNOWN


def test_validator_rejects_four_fingers_misrecognized_as_stop():
    """8. FOUR_FINGERS incorrectly recognized as STOP -> validator rejects."""
    hand = make_hand_landmarks(tip_offsets={"thumb": (0.2, -0.15), "index": (0.0, -0.25)}, handedness=Handedness.RIGHT)
    det_res = DetectionResult(hands=[hand], num_hands=1)
    feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="STOP", confidence=1.0)
    target_cmd = NeuroGripCommand.FOUR_FINGERS

    is_valid, reason = DatasetSampleValidator.validate(det_res, feats, rec_res, target_cmd)
    assert is_valid is False
    assert "REJECT_MISMATCH" in reason


def test_validator_rejects_three_finger_misrecognized_as_four_fingers():
    """9. THREE_FINGER incorrectly recognized as FOUR_FINGERS -> validator rejects."""
    hand = make_hand_landmarks(tip_offsets={"thumb": (0.2, -0.15), "index": (0.0, -0.25)}, handedness=Handedness.RIGHT)
    det_res = DetectionResult(hands=[hand], num_hands=1)
    feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="FOUR_FINGERS", confidence=1.0)
    target_cmd = NeuroGripCommand.THREE_FINGER

    is_valid, reason = DatasetSampleValidator.validate(det_res, feats, rec_res, target_cmd)
    assert is_valid is False
    assert "REJECT_MISMATCH" in reason


def test_validator_accepts_exact_target_match():
    """10. Exact target recognition -> validator accepts."""
    hand = make_hand_landmarks(tip_offsets={"thumb": (0.2, -0.15), "index": (0.0, -0.25)}, handedness=Handedness.RIGHT)
    det_res = DetectionResult(hands=[hand], num_hands=1)
    feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="TWO_FINGER", confidence=1.0)
    target_cmd = NeuroGripCommand.TWO_FINGER

    is_valid, reason = DatasetSampleValidator.validate(det_res, feats, rec_res, target_cmd)
    assert is_valid is True
    assert reason == "VALID"


def test_four_fingers_with_thumb_unknown(recognizer, extractor):
    """FOUR_FINGERS recognized when thumb state is UNKNOWN."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.12, -0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value


def test_four_fingers_with_thumb_closed(recognizer, extractor):
    """FOUR_FINGERS recognized when thumb is CLOSED/tucked."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.0, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value


def test_four_fingers_noisy_thumb_open_narrow_spread(recognizer, extractor):
    """FOUR_FINGERS must NOT become STOP merely because thumb is classified OPEN if spread is narrow."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.08, -0.12),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == NeuroGripCommand.FOUR_FINGERS.value
    assert res.reason == "FOUR_NON_THUMB_OPEN_THUMB_NARROW"


def test_four_fingers_wide_spread_thumb_closed(recognizer, extractor):
    """FOUR_FINGERS recognized when thumb has wide spread offset but finger state is NOT OPEN."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.22, -0.15),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    metric = recognizer.get_thumb_spread_metric(lms)
    assert metric >= 0.45
    fs = recognizer.finger_detector.detect(lms)
    fs_closed_thumb = FingerStateVector(
        thumb=FingerState.CLOSED,
        index=fs.index,
        middle=fs.middle,
        ring=fs.ring,
        pinky=fs.pinky,
        is_palm_facing=fs.is_palm_facing,
    )
    feats = extractor.extract(lms)
    label, conf, reason = recognizer._classify_finger_states(fs_closed_thumb, lms, feats, False)
    assert label == NeuroGripCommand.FOUR_FINGERS.value
    assert reason == "FOUR_NON_THUMB_OPEN_THUMB_NOT_OPEN"


def test_palmfacing_false_returns_unknown_not_rest(recognizer):
    """PalmFacing=False (all finger states UNKNOWN) must return UNKNOWN, not REST."""
    detector = FingerStateDetector()
    lms_data = [NormalizedLandmark(x=0.5, y=0.8, z=0.0)]
    for i in range(1, 21):
        lms_data.append(NormalizedLandmark(x=0.5, y=0.8 - (i * 0.02), z=float(i) * 0.05))
    hlms = HandLandmarks(landmarks=lms_data, handedness=Handedness.RIGHT)

    states = detector.detect(hlms)
    assert states.thumb == FingerState.UNKNOWN
    assert states.index == FingerState.UNKNOWN
    assert states.middle == FingerState.UNKNOWN
    assert states.ring == FingerState.UNKNOWN
    assert states.pinky == FingerState.UNKNOWN

    feats = np.zeros(68, dtype=np.float32)
    res = recognizer.predict(feats, hand_landmarks=hlms)
    assert res.label == "UNKNOWN"
    assert res.confidence == 0.0
    assert res.reason == "NOT_PALM_FACING"


def test_near_boundary_thumb_geometry_returns_unknown(recognizer, extractor):
    """Near-boundary thumb spread geometry (0.35 <= metric < 0.45) returns UNKNOWN."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.11, -0.15),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    metric = recognizer.get_thumb_spread_metric(lms)
    assert 0.35 <= metric < 0.45, f"Expected metric in ambiguity range [0.35, 0.45), got {metric}"
    feats = extractor.extract(lms)
    res = recognizer.predict(feats, hand_landmarks=lms)
    assert res.label == "UNKNOWN"
    assert res.confidence == 0.0
    assert res.reason == "FOUR_NON_THUMB_OPEN_THUMB_AMBIGUOUS"


def test_deterministic_recognition_same_landmarks(recognizer, extractor):
    """Repeated predictions on identical landmarks return identical labels deterministically."""
    lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.22, -0.15),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )
    feats = extractor.extract(lms)
    res1 = recognizer.predict(feats, hand_landmarks=lms)
    res2 = recognizer.predict(feats, hand_landmarks=lms)
    res3 = recognizer.predict(feats, hand_landmarks=lms)
    assert res1.label == res2.label == res3.label == NeuroGripCommand.STOP.value
    assert res1.reason == res2.reason == res3.reason == "FOUR_NON_THUMB_OPEN_THUMB_WIDE_OPEN"
