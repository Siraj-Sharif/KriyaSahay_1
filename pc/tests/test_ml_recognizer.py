"""
tests/test_ml_recognizer.py
────────────────────────────
Unit tests for MLRecognizer.
Uses synthetic scikit-learn models saved to temporary files to test model bundle validation,
feature version/dimension checking, confidence gating, and STOP security overrides.
"""
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, RecognitionConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.recognition.ml_recognizer import MLRecognizer


@pytest.fixture
def dummy_model_bundle():
    """Create a temporary trained RandomForest model bundle dictionary."""
    # Create synthetic dataset with FeatureExtractor.FEATURE_DIM features and 3 classes ("INDEX", "CLOSE", "STOP")
    X = np.random.randn(30, FeatureExtractor.FEATURE_DIM).astype(np.float32)
    y = np.array(["INDEX"] * 10 + ["CLOSE"] * 10 + ["STOP"] * 10)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    bundle = {
        "classifier": clf,
        "preprocessor": None,
        "classes": list(clf.classes_),
        "feature_version": FeatureExtractor.FEATURE_VERSION,
        "feature_dim": FeatureExtractor.FEATURE_DIM,
        "trained_at": "2026-09-04T00:00:00Z",
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        bundle_path = Path(tmp.name)
        joblib.dump(bundle, bundle_path)

    yield bundle_path

    if bundle_path.exists():
        bundle_path.unlink()


def test_ml_recognizer_missing_model_file():
    """Verify MLRecognizer handles missing model files gracefully without crashing."""
    cfg = AppConfig()
    cfg.recognition.model_path = "non_existent_model_dir/model.pkl"

    recognizer = MLRecognizer(config=cfg)
    assert recognizer.is_ready is False

    # Predict on uninitialized model -> UNKNOWN
    res = recognizer.predict(np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32))
    assert res.label == "UNKNOWN"
    assert res.confidence == 0.0
    assert res.recognizer_type == "ml_uninitialized"


def test_ml_recognizer_valid_model_loading(dummy_model_bundle):
    """Verify MLRecognizer loads and predicts cleanly from a valid model bundle."""
    cfg = AppConfig()
    cfg.recognition.model_path = str(dummy_model_bundle)

    recognizer = MLRecognizer(config=cfg)
    assert recognizer.is_ready is True

    feat = np.random.randn(FeatureExtractor.FEATURE_DIM).astype(np.float32)
    res = recognizer.predict(feat)

    assert isinstance(res.label, str)
    assert 0.0 <= res.confidence <= 1.0
    assert res.recognizer_type == "ml_random_forest"
    assert len(res.all_scores) > 0


def test_ml_recognizer_feature_version_mismatch(dummy_model_bundle):
    """Verify MLRecognizer rejects a model bundle with incompatible feature version."""
    bundle = joblib.load(dummy_model_bundle)
    bundle["feature_version"] = "v0_incompatible"

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        bad_path = Path(tmp.name)
        joblib.dump(bundle, bad_path)

    try:
        cfg = AppConfig()
        cfg.recognition.model_path = str(bad_path)
        recognizer = MLRecognizer(config=cfg)

        assert recognizer.is_ready is False
    finally:
        if bad_path.exists():
            bad_path.unlink()


def test_ml_recognizer_feature_dim_mismatch(dummy_model_bundle):
    """Verify MLRecognizer rejects a model bundle with incompatible feature dimension."""
    bundle = joblib.load(dummy_model_bundle)
    bundle["feature_dim"] = 50  # Bad dimension

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        bad_path = Path(tmp.name)
        joblib.dump(bundle, bad_path)

    try:
        cfg = AppConfig()
        cfg.recognition.model_path = str(bad_path)
        recognizer = MLRecognizer(config=cfg)

        assert recognizer.is_ready is False
    finally:
        if bad_path.exists():
            bad_path.unlink()


def test_ml_recognizer_confidence_gating(dummy_model_bundle):
    """Verify low-confidence predictions below threshold return UNKNOWN."""
    cfg = AppConfig()
    cfg.recognition.model_path = str(dummy_model_bundle)
    cfg.recognition.normal_min_confidence = 0.9999  # Unreachably high threshold

    recognizer = MLRecognizer(config=cfg)
    assert recognizer.is_ready is True

    feat = np.random.randn(FeatureExtractor.FEATURE_DIM).astype(np.float32)
    res = recognizer.predict(feat)

    assert res.label == "UNKNOWN"
    assert res.is_unknown is True


def test_ml_recognizer_stop_armed_security_override(dummy_model_bundle):
    """Verify predicted STOP is overridden to UNKNOWN when is_stop_armed=False."""
    cfg = AppConfig()
    cfg.recognition.model_path = str(dummy_model_bundle)
    cfg.recognition.normal_min_confidence = 0.01  # Accept any prediction

    recognizer = MLRecognizer(config=cfg)

    # Force classifier to output "STOP" by mocking predict_proba
    recognizer._classifier.predict_proba = lambda x: np.array([[0.0, 0.0, 1.0]])  # STOP is 3rd class

    # When is_stop_armed=False -> overridden to UNKNOWN
    res_unarmed = recognizer.predict(np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32), is_stop_armed=False)
    assert res_unarmed.label == "UNKNOWN"

    # When is_stop_armed=True -> allowed to be STOP
    res_armed = recognizer.predict(np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32), is_stop_armed=True)
    assert res_armed.label == NeuroGripCommand.STOP.value
