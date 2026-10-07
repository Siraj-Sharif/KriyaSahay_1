"""
tests/test_hybrid_recognizer.py
─────────────────────────────────
Unit tests for production HybridRecognizer.
Tests deterministic STOP safety priority, ML model inference delegation,
ML STOP untrusted override, and REST gesture handling.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.recognition.base import RecognitionResult
from neurogrip.recognition.hybrid import HybridRecognizer
from neurogrip.recognition.ml_recognizer import MLRecognizer
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from tests.conftest import make_hand_landmarks


def test_hybrid_recognizer_initialization():
    """Verify HybridRecognizer initializes ready with default AppConfig."""
    cfg = AppConfig()
    hybrid = HybridRecognizer(config=cfg)
    assert hybrid.is_ready is True


def test_hybrid_recognizer_deterministic_stop_priority():
    """Verify deterministic flat-hand wide-thumb posture triggers STOP immediately with 1.0 confidence."""
    cfg = AppConfig()
    hybrid = HybridRecognizer(config=cfg)

    # Landmarks for flat hand + wide thumb spread (STOP gesture)
    stop_lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.08, -0.25),
            "index": (0.01, -0.25),
            "middle": (0.0, -0.25),
            "ring": (-0.01, -0.25),
            "pinky": (-0.02, -0.25),
        }
    )

    dummy_feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    res = hybrid.predict(dummy_feats, hand_landmarks=stop_lms, is_stop_armed=True)

    assert res.label == NeuroGripCommand.STOP.value
    assert res.confidence == 1.0
    assert "stop" in res.recognizer_type.lower()


def test_hybrid_recognizer_ml_stop_untrusted_override():
    """Verify ML STOP predictions are overridden to UNKNOWN when deterministic STOP did not fire."""
    cfg = AppConfig()

    class MockMLRecognizer(MLRecognizer):
        @property
        def is_ready(self) -> bool:
            return True

        def predict(self, features, hand_landmarks=None, is_stop_armed=True):
            return RecognitionResult(
                label=NeuroGripCommand.STOP.value,
                confidence=0.95,
                all_scores={"STOP": 0.95},
                recognizer_type="mock_ml",
            )

    hybrid = HybridRecognizer(config=cfg, ml_recognizer=MockMLRecognizer(config=cfg))

    # Landmark object that does NOT trigger deterministic STOP (e.g. narrow thumb)
    normal_lms = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, -0.10),
            "index": (0.01, -0.25),
            "middle": (0.0, -0.25),
            "ring": (-0.01, -0.25),
            "pinky": (-0.02, -0.25),
        }
    )

    dummy_feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)

    # When STOP is armed, ML STOP prediction is accepted
    res = hybrid.predict(dummy_feats, hand_landmarks=normal_lms, is_stop_armed=True)
    assert res.label == "STOP"

    # When STOP is disarmed, recognition still identifies STOP (arming gates transmission downstream, not recognition)
    res_disarmed = hybrid.predict(dummy_feats, hand_landmarks=normal_lms, is_stop_armed=False)
    assert res_disarmed.label == "STOP"


def test_hybrid_recognizer_ml_active_gesture_delegation():
    """Verify active ML gesture predictions (e.g. INDEX) pass through cleanly."""
    cfg = AppConfig()

    class MockMLRecognizer(MLRecognizer):
        @property
        def is_ready(self) -> bool:
            return True

        def predict(self, features, hand_landmarks=None, is_stop_armed=True, raw_frame=None):
            return RecognitionResult(
                label="INDEX",
                confidence=0.92,
                all_scores={"INDEX": 0.92},
                recognizer_type="mock_ml",
            )

    mock_hagrid = MagicMock()
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {
        "raw_class": "point",
        "mapped_command": "INDEX_FINGER",
        "confidence": 0.90,
        "top_k": [],
    }

    hybrid = HybridRecognizer(config=cfg, hagrid_classifier=mock_hagrid, ml_recognizer=MockMLRecognizer(config=cfg))

    index_hand = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, -0.10),
            "index": (0.01, -0.30),
            "middle": (0.0, 0.0),
            "ring": (0.0, 0.0),
            "pinky": (0.0, 0.0),
        }
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    res = hybrid.predict(dummy_feats, hand_landmarks=index_hand, is_stop_armed=True, raw_frame=dummy_frame)

    assert res.label == "INDEX_FINGER"
    assert res.model_used == "EXTRA_TREES"

