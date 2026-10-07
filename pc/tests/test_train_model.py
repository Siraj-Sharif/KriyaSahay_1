"""
tests/test_train_model.py
───────────────────────────
Unit tests for train_model script.
Tests data validation, feature dimension checks, missing/invalid feature handling,
saved model bundle structure, reloading, and MLRecognizer integration.
"""
import csv
import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.recognition.ml_recognizer import MLRecognizer
from scripts.train_model import (
    OFFICIAL_CLASSES,
    evaluate_predictions,
    load_and_validate_split,
    train_and_evaluate,
)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory containing valid train.csv, val.csv, test.csv for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        header = ["label", "handedness", "session_id", "sample_id"] + [
            f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
        ]

        for split_name in ["train.csv", "val.csv", "test.csv"]:
            csv_file = tmp_path / split_name
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # Create 2 samples for each of the 13 official classes
                for cls_name in OFFICIAL_CLASSES:
                    for i in range(2):
                        feat_row = [float(np.random.randn()) for _ in range(FeatureExtractor.FEATURE_DIM)]
                        row = [cls_name, "RIGHT", f"sess_{cls_name}", f"sample_{i}"] + feat_row
                        writer.writerow(row)

        yield tmp_path


def test_load_and_validate_split_valid(temp_dataset_dir):
    """Verify load_and_validate_split loads a valid dataset CSV correctly."""
    train_path = temp_dataset_dir / "train.csv"
    X, y = load_and_validate_split(train_path)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (26, FeatureExtractor.FEATURE_DIM)  # 13 classes * 2 samples
    assert len(y) == 26
    assert set(y) == set(OFFICIAL_CLASSES)


def test_load_and_validate_split_missing_file():
    """Verify load_and_validate_split raises FileNotFoundError when CSV is missing."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_split(Path("non_existent_directory/train.csv"))


def test_load_and_validate_split_invalid_header_col_count(temp_dataset_dir):
    """Verify load_and_validate_split detects header column count mismatches."""
    bad_csv = temp_dataset_dir / "bad_header.csv"
    bad_header = ["label", "handedness", "session_id", "sample_id", "feature_0", "feature_1"]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(bad_header)

    with pytest.raises(ValueError, match="Header column count mismatch"):
        load_and_validate_split(bad_csv)


def test_load_and_validate_split_unauthorized_label(temp_dataset_dir):
    """Verify load_and_validate_split detects labels outside official 13 classes."""
    bad_csv = temp_dataset_dir / "unauthorized_label.csv"
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(["OPEN", "RIGHT", "sess1", "samp1"] + [0.1] * FeatureExtractor.FEATURE_DIM)

    with pytest.raises(ValueError, match="unauthorized label"):
        load_and_validate_split(bad_csv)


def test_load_and_validate_split_non_numeric_feature(temp_dataset_dir):
    """Verify load_and_validate_split detects non-numeric feature values."""
    bad_csv = temp_dataset_dir / "non_numeric.csv"
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = ["INDEX", "RIGHT", "sess1", "samp1"] + [0.1] * (FeatureExtractor.FEATURE_DIM - 1) + ["NOT_A_FLOAT"]
        writer.writerow(row)

    with pytest.raises(ValueError, match="non-numeric feature value"):
        load_and_validate_split(bad_csv)


def test_load_and_validate_split_nan_inf_feature(temp_dataset_dir):
    """Verify load_and_validate_split detects NaN and Inf values in features."""
    bad_csv = temp_dataset_dir / "nan_feature.csv"
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = ["INDEX", "RIGHT", "sess1", "samp1"] + [0.1] * (FeatureExtractor.FEATURE_DIM - 1) + ["NaN"]
        writer.writerow(row)

    with pytest.raises(ValueError, match="invalid feature value \\(NaN/Inf\\)"):
        load_and_validate_split(bad_csv)


def test_load_and_validate_split_missing_class(temp_dataset_dir):
    """Verify load_and_validate_split raises error when split is missing required official classes."""
    incomplete_csv = temp_dataset_dir / "incomplete.csv"
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]
    with open(incomplete_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Only write 1 class instead of all 13
        writer.writerow(["INDEX", "RIGHT", "sess1", "samp1"] + [0.1] * FeatureExtractor.FEATURE_DIM)

    with pytest.raises(ValueError, match="missing expected official classes"):
        load_and_validate_split(incomplete_csv)


def test_evaluate_predictions_metrics():
    """Verify evaluation metric calculations."""
    y_true = np.array(["INDEX", "INDEX", "CLOSE", "STOP"])
    y_pred = np.array(["INDEX", "CLOSE", "CLOSE", "STOP"])
    classes = ["CLOSE", "INDEX", "STOP"]

    metrics = evaluate_predictions(y_true, y_pred, classes)

    assert "accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.75
    assert len(metrics["confusion_matrix"]) == 3


def test_train_and_evaluate_pipeline(temp_dataset_dir):
    """Verify complete training, model bundle saving, JSON report, and MLRecognizer integration."""
    with tempfile.TemporaryDirectory() as model_dir_str:
        model_dir = Path(model_dir_str)

        # Run training with tiny n_estimators for fast execution
        exit_code = train_and_evaluate(
            processed_dir=temp_dataset_dir,
            model_dir=model_dir,
            seed=42,
            n_estimators=5,
            n_jobs=1,
        )

        assert exit_code == 0

        model_pkl = model_dir / "neurogrip_rf_v2.pkl" if (model_dir / "neurogrip_rf_v2.pkl").exists() else model_dir / "neurogrip_rf_v1.pkl"
        report_json = model_dir / "neurogrip_rf_v2_report.json" if (model_dir / "neurogrip_rf_v2_report.json").exists() else model_dir / "neurogrip_rf_v1_report.json"

        assert model_pkl.exists()
        assert report_json.exists()

        # 1. Verify saved model bundle (.pkl) structure
        bundle = joblib.load(model_pkl)
        assert isinstance(bundle, dict)
        assert "classifier" in bundle
        assert isinstance(bundle["classifier"], RandomForestClassifier)
        assert bundle["preprocessor"] is None
        assert "classes" in bundle
        assert len(bundle["classes"]) == 13
        assert bundle["classes"] == OFFICIAL_CLASSES
        assert bundle["feature_version"] == FeatureExtractor.FEATURE_VERSION
        assert bundle["feature_dim"] == FeatureExtractor.FEATURE_DIM
        assert "trained_at" in bundle

        # 2. Verify JSON report content
        with open(report_json, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["model_type"] == "RandomForestClassifier"
        assert report["feature_version"] == FeatureExtractor.FEATURE_VERSION
        assert report["feature_dim"] == FeatureExtractor.FEATURE_DIM
        assert report["training_sample_count"] == 26
        assert report["validation_sample_count"] == 26
        assert report["test_sample_count"] == 26
        assert len(report["class_names"]) == 13
        assert "validation_metrics" in report
        assert "final_test_metrics" in report
        assert "per_class_metrics" in report
        assert "confusion_matrix" in report

        # 3. Verify MLRecognizer reloads and performs inference successfully
        cfg = AppConfig()
        cfg.recognition.model_path = str(model_pkl)
        recognizer = MLRecognizer(config=cfg)

        assert recognizer.is_ready is True

        dummy_feat = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
        result = recognizer.predict(dummy_feat, is_stop_armed=True)

        assert result.label in OFFICIAL_CLASSES or result.label == "UNKNOWN"
        assert result.recognizer_type == "ml_random_forest"
