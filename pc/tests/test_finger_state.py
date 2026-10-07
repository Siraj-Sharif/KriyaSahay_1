"""
tests/test_finger_state.py
───────────────────────────
Unit tests for FingerStateDetector and FingerStateVector.
"""
from neurogrip.features.finger_state import (
    Finger,
    FingerState,
    FingerStateDetector,
    FingerStateVector,
)


def test_finger_state_vector_accessors():
    """Test dictionary and property accessors on FingerStateVector."""
    vec = FingerStateVector(
        thumb=FingerState.OPEN,
        index=FingerState.OPEN,
        middle=FingerState.CLOSED,
        ring=FingerState.CLOSED,
        pinky=FingerState.UNKNOWN,
    )

    assert vec.thumb == FingerState.OPEN
    assert vec.index == FingerState.OPEN
    assert vec.middle == FingerState.CLOSED
    assert vec.ring == FingerState.CLOSED
    assert vec.pinky == FingerState.UNKNOWN

    assert vec[Finger.THUMB] == FingerState.OPEN
    assert vec["index"] == FingerState.OPEN
    assert vec["PINKY"] == FingerState.UNKNOWN

    d = vec.as_dict()
    assert d[Finger.THUMB] == FingerState.OPEN
    assert d[Finger.MIDDLE] == FingerState.CLOSED


def test_finger_state_detector_all_open(open_hand_landmarks):
    """Verify all 5 fingers detected as OPEN for open hand pose."""
    detector = FingerStateDetector()
    states = detector.detect(open_hand_landmarks)

    assert states.thumb == FingerState.OPEN
    assert states.index == FingerState.OPEN
    assert states.middle == FingerState.OPEN
    assert states.ring == FingerState.OPEN
    assert states.pinky == FingerState.OPEN


def test_finger_state_detector_all_closed(closed_hand_landmarks):
    """Verify all 5 fingers detected as CLOSED for closed fist pose."""
    detector = FingerStateDetector()
    states = detector.detect(closed_hand_landmarks)

    assert states.thumb == FingerState.CLOSED
    assert states.index == FingerState.CLOSED
    assert states.middle == FingerState.CLOSED
    assert states.ring == FingerState.CLOSED
    assert states.pinky == FingerState.CLOSED


def test_finger_state_detector_index_only(index_only_landmarks):
    """Verify index finger OPEN while other fingers CLOSED for index-only pose."""
    detector = FingerStateDetector()
    states = detector.detect(index_only_landmarks)

    assert states.index == FingerState.OPEN
    assert states.middle == FingerState.CLOSED
    assert states.ring == FingerState.CLOSED
    assert states.pinky == FingerState.CLOSED


def test_finger_state_detector_thumb_only(thumb_only_landmarks):
    """Verify thumb detected as OPEN (lateral extension) while remaining fingers CLOSED."""
    detector = FingerStateDetector()
    states = detector.detect(thumb_only_landmarks)

    assert states.thumb == FingerState.OPEN
    assert states.index == FingerState.CLOSED
    assert states.middle == FingerState.CLOSED
    assert states.ring == FingerState.CLOSED
    assert states.pinky == FingerState.CLOSED
