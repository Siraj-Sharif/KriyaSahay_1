"""
scripts/train_model.py
───────────────────────
Phase 10C: ML Model Training and Validation Script.
Loads preprocessed train, validation, and test datasets from pc/data/processed/,
validates dataset schema and 67-D feature vectors, trains a baseline RandomForestClassifier
on train.csv, evaluates on val.csv, performs final test evaluation on test.csv,
saves the model bundle (.pkl) and JSON training report to pc/models/.

Usage:
    python pc/scripts/train_model.py
    python pc/scripts/train_model.py --processed-dir pc/data/processed --model-dir pc/models --seed 42
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure src directory is in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.recognition.ml_recognizer import MLRecognizer

logger = logging.getLogger("neurogrip.train_model")

# Canonical 13 ML classes sorted deterministically
OFFICIAL_CLASSES: list[str] = sorted([c.value for c in NeuroGripCommand])


def load_and_validate_split(
    csv_path: Path,
    expected_classes: Sequence[str] = OFFICIAL_CLASSES,
    expected_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single dataset split CSV file and perform data validation.

    Validates:
    - File existence
    - Header column names & feature count (must equal expected_dim)
    - Metadata columns (label, handedness, session_id, sample_id) exist
    - Feature values are non-empty, numeric, and finite (no NaN or Inf)
    - Labels belong to expected_classes (13 official classes)
    - All expected_classes are present in the split
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split file not found: '{path}'")

    exp_classes_set = set(expected_classes)
    features_list: list[list[float]] = []
    labels_list: list[str] = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header:
            raise ValueError(f"Empty CSV file: '{path}'")

        if len(header) < 4 or header[0:4] != ["label", "handedness", "session_id", "sample_id"]:
            raise ValueError(f"Invalid header structure in '{path}'. First 4 columns must be label, handedness, session_id, sample_id.")

        det_dim = len(header) - 4
        if expected_dim is not None:
            dim = expected_dim
        else:
            if det_dim in (67, 68):
                dim = det_dim
            else:
                dim = FeatureExtractor.FEATURE_DIM

        expected_total_cols = 4 + dim

        if len(header) != expected_total_cols:
            raise ValueError(
                f"Header column count mismatch in '{path}': "
                f"expected {expected_total_cols} columns (4 metadata + {dim} features), "
                f"got {len(header)}."
            )

        feature_col_names = header[4:]
        expected_feat_names = [f"feature_{i}" for i in range(dim)]
        if feature_col_names != expected_feat_names:
            raise ValueError(
                f"Feature column header mismatch in '{path}'. "
                f"Expected feature_0 through feature_{dim - 1}."
            )

        for line_num, row in enumerate(reader, start=2):
            if len(row) != expected_total_cols:
                raise ValueError(
                    f"Row {line_num} in '{path}' has column count mismatch: "
                    f"expected {expected_total_cols}, got {len(row)}."
                )

            label = row[0].strip()
            if not label:
                raise ValueError(f"Row {line_num} in '{path}' has empty label.")

            if label not in exp_classes_set:
                raise ValueError(
                    f"Row {line_num} in '{path}' has unauthorized label '{label}'. "
                    f"Must be one of official classes."
                )

            feat_vals: list[float] = []
            for col_idx, val_str in enumerate(row[4:], start=4):
                val_str = val_str.strip()
                if not val_str:
                    raise ValueError(
                        f"Row {line_num} col {col_idx} in '{path}' has empty feature value."
                    )
                try:
                    val = float(val_str)
                except ValueError:
                    raise ValueError(
                        f"Row {line_num} col {col_idx} in '{path}' has non-numeric feature value '{val_str}'."
                    )

                if np.isnan(val) or np.isinf(val):
                    raise ValueError(
                        f"Row {line_num} col {col_idx} in '{path}' has invalid feature value (NaN/Inf): {val}."
                    )

                feat_vals.append(val)

            features_list.append(feat_vals)
            labels_list.append(label)

    if not features_list:
        raise ValueError(f"No valid data rows found in '{path}'.")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=str)

    if X.shape[1] != dim:
        raise ValueError(
            f"Feature matrix dimension mismatch for '{path}': "
            f"expected ({X.shape[0]}, {dim}), got {X.shape}."
        )

    present_classes = set(y)
    missing_classes = exp_classes_set - present_classes
    if missing_classes:
        raise ValueError(
            f"Dataset split '{path}' is missing expected official classes: {sorted(missing_classes)}."
        )

    return X, y


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Calculate classification metrics against a specified class order."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    per_class: dict[str, dict[str, Any]] = {}
    for cls_name in classes:
        if cls_name in report_dict:
            per_class[cls_name] = {
                "precision": float(report_dict[cls_name]["precision"]),
                "recall": float(report_dict[cls_name]["recall"]),
                "f1_score": float(report_dict[cls_name]["f1-score"]),
                "support": int(report_dict[cls_name]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def build_train_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for train_model.py."""
    data_root = Path(__file__).resolve().parent.parent / "data"
    model_root = Path(__file__).resolve().parent.parent / "models"
    default_processed = data_root / "processed_v2" if (data_root / "processed_v2").exists() else data_root / "processed"

    parser = argparse.ArgumentParser(
        prog="train_model",
        description="NeuroGrip ML Model Training & Validation.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help=f"Directory containing processed train.csv, val.csv, test.csv (default: {default_processed}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=model_root,
        help=f"Output directory for model bundle (.pkl) and training report (.json) (default: {model_root}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of decision trees in RandomForestClassifier (default: 300).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for training (-1 uses all processors, default: -1).",
    )
    return parser


def train_and_evaluate(
    processed_dir: Path,
    model_dir: Path,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> int:
    """
    Train baseline RandomForestClassifier model and evaluate on validation & test sets.
    """
    logger.info("Starting NeuroGrip ML Training Pipeline...")
    logger.info("  Processed Data Dir : %s", processed_dir)
    logger.info("  Model Output Dir   : %s", model_dir)
    logger.info("  Random Seed        : %d", seed)
    logger.info("  N Estimators       : %d", n_estimators)
    logger.info("  N Jobs             : %d", n_jobs)

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    # 1. Load and validate all three splits
    logger.info("Loading and validating dataset splits...")
    X_train, y_train = load_and_validate_split(train_csv)
    X_val, y_val = load_and_validate_split(val_csv)
    X_test, y_test = load_and_validate_split(test_csv)

    feat_dim = X_train.shape[1]
    feat_version = "v2" if feat_dim == 68 else ("v1" if feat_dim == 67 else f"v_{feat_dim}")
    is_v2 = (feat_version == "v2")

    model_filename = "neurogrip_rf_v2.pkl" if is_v2 else "neurogrip_rf_v1.pkl"
    report_filename = "neurogrip_rf_v2_report.json" if is_v2 else "neurogrip_rf_v1_report.json"

    logger.info(
        "Dataset Loaded Successfully: Train=%d samples, Val=%d samples, Test=%d samples. Feature Dim=%d (%s).",
        len(X_train),
        len(X_val),
        len(X_test),
        feat_dim,
        feat_version,
    )

    # 2. Train baseline RandomForestClassifier using ONLY train.csv
    logger.info("Training RandomForestClassifier baseline on train split...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    classes_list = list(clf.classes_)
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    logger.info("Model fitted. Discovered %d classes: %s", len(classes_list), classes_list)

    # 3. Evaluate model on Validation set
    y_val_pred = clf.predict(X_val)
    val_metrics = evaluate_predictions(y_val, y_val_pred, classes_list)

    # Print Validation Results to Console
    print("\n==================================================")
    print(f"Validation Results ({feat_version.upper()} - val.csv)")
    print("==================================================")
    print(f"Accuracy        : {val_metrics['accuracy']:.4f}")
    print(f"Macro Precision : {val_metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {val_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score  : {val_metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {val_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Validation Metrics:")
    for cls_name, m in val_metrics["per_class"].items():
        print(f"  {cls_name:<14} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['support']}")

    # 4. Evaluate finalized model on Test set
    y_test_pred = clf.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_test_pred, classes_list)

    # Print Final Test Results to Console
    print("\n==================================================")
    print(f"Final Test Results ({feat_version.upper()} - test.csv)")
    print("==================================================")
    print(f"Accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision : {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score  : {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {test_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Test Metrics:")
    for cls_name, m in test_metrics["per_class"].items():
        print(f"  {cls_name:<14} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['support']}")

    # 5. Critical Pair Confusion Analysis
    cm_test = np.array(test_metrics["confusion_matrix"])
    critical_pairs = [
        ("STOP", "FOUR_FINGERS"),
        ("INDEX", "MIDDLE"),
        ("GRAB", "CLOSE"),
        ("REST", "FOUR_FINGERS"),
        ("TWO_FINGER", "INDEX_PINKY"),
        ("THREE_FINGER", "FOUR_FINGERS"),
    ]

    pair_confusions: dict[str, dict[str, int]] = {}
    print("\n--------------------------------------------------")
    print("Critical Pair Confusion Analysis (Test Split)")
    print("--------------------------------------------------")
    for c1, c2 in critical_pairs:
        if c1 in class_to_idx and c2 in class_to_idx:
            idx1, idx2 = class_to_idx[c1], class_to_idx[c2]
            c1_pred_as_c2 = int(cm_test[idx1, idx2])
            c2_pred_as_c1 = int(cm_test[idx2, idx1])
            pair_key = f"{c1} vs {c2}"
            pair_confusions[pair_key] = {
                f"{c1}_actual_predicted_as_{c2}": c1_pred_as_c2,
                f"{c2}_actual_predicted_as_{c1}": c2_pred_as_c1,
            }
            print(f"  {c1:12} vs {c2:12} | {c1} -> {c2}: {c1_pred_as_c2:3d} | {c2} -> {c1}: {c2_pred_as_c1:3d}")

    # 6. Baseline Comparison (v2 vs Frozen v1 Baseline)
    v1_baseline = {
        "val_accuracy": 0.6004,
        "val_macro_f1": 0.5480,
        "test_accuracy": 0.7362,
        "test_macro_f1": 0.7021,
        "stop_test_f1": 0.0000,
        "index_test_f1": 0.1121,
        "four_fingers_test_f1": 0.5908,
    }

    v2_stop_f1 = test_metrics["per_class"]["STOP"]["f1_score"] if "STOP" in test_metrics["per_class"] else 0.0
    v2_index_f1 = test_metrics["per_class"]["INDEX"]["f1_score"] if "INDEX" in test_metrics["per_class"] else 0.0
    v2_four_f1 = test_metrics["per_class"]["FOUR_FINGERS"]["f1_score"] if "FOUR_FINGERS" in test_metrics["per_class"] else 0.0

    print("\n--------------------------------------------------")
    print("Baseline Comparison (v2 vs Frozen v1 Baseline)")
    print("--------------------------------------------------")
    print(f"  Val Accuracy     | v1: {v1_baseline['val_accuracy']:.4f}  ->  v2: {val_metrics['accuracy']:.4f}  (Delta: {val_metrics['accuracy'] - v1_baseline['val_accuracy']:+.4f})")
    print(f"  Val Macro F1     | v1: {v1_baseline['val_macro_f1']:.4f}  ->  v2: {val_metrics['macro_f1']:.4f}  (Delta: {val_metrics['macro_f1'] - v1_baseline['val_macro_f1']:+.4f})")
    print(f"  Test Accuracy    | v1: {v1_baseline['test_accuracy']:.4f}  ->  v2: {test_metrics['accuracy']:.4f}  (Delta: {test_metrics['accuracy'] - v1_baseline['test_accuracy']:+.4f})")
    print(f"  Test Macro F1    | v1: {v1_baseline['test_macro_f1']:.4f}  ->  v2: {test_metrics['macro_f1']:.4f}  (Delta: {test_metrics['macro_f1'] - v1_baseline['test_macro_f1']:+.4f})")
    print(f"  STOP Test F1     | v1: {v1_baseline['stop_test_f1']:.4f}  ->  v2: {v2_stop_f1:.4f}  (Delta: {v2_stop_f1 - v1_baseline['stop_test_f1']:+.4f})")
    print(f"  INDEX Test F1    | v1: {v1_baseline['index_test_f1']:.4f}  ->  v2: {v2_index_f1:.4f}  (Delta: {v2_index_f1 - v1_baseline['index_test_f1']:+.4f})")
    print(f"  FOUR_FINGERS F1  | v1: {v1_baseline['four_fingers_test_f1']:.4f}  ->  v2: {v2_four_f1:.4f}  (Delta: {v2_four_f1 - v1_baseline['four_fingers_test_f1']:+.4f})")
    print("==================================================\n")

    # 7. Save Model Bundle (.pkl)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename
    trained_at_iso = datetime.now(timezone.utc).isoformat()

    bundle = {
        "classifier": clf,
        "preprocessor": None,
        "classes": classes_list,
        "feature_version": feat_version,
        "feature_dim": feat_dim,
        "trained_at": trained_at_iso,
    }

    joblib.dump(bundle, model_path)
    logger.info("Saved model bundle to '%s'.", model_path)

    # 8. Save JSON Training Report
    report_path = model_dir / report_filename
    report_data = {
        "trained_at": trained_at_iso,
        "model_type": "RandomForestClassifier",
        "model_parameters": {
            "n_estimators": clf.n_estimators,
            "random_state": clf.random_state,
            "class_weight": clf.class_weight,
            "n_jobs": clf.n_jobs,
        },
        "feature_version": feat_version,
        "feature_dim": feat_dim,
        "training_sample_count": len(X_train),
        "validation_sample_count": len(X_val),
        "test_sample_count": len(X_test),
        "class_names": classes_list,
        "validation_metrics": {
            "accuracy": val_metrics["accuracy"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        },
        "final_test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
        "per_class_metrics": {
            "validation": val_metrics["per_class"],
            "test": test_metrics["per_class"],
        },
        "confusion_matrix": {
            "validation": val_metrics["confusion_matrix"],
            "test": test_metrics["confusion_matrix"],
        },
        "critical_pair_confusions": pair_confusions,
        "v1_baseline_comparison": {
            "v1": v1_baseline,
            "v2": {
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "stop_test_f1": v2_stop_f1,
                "index_test_f1": v2_index_f1,
                "four_fingers_test_f1": v2_four_f1,
            },
        },
    }

    with open(report_path, mode="w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved JSON training report to '%s'.", report_path)

    # 9. Integration Verification with MLRecognizer
    logger.info("Performing integration verification with MLRecognizer...")
    loaded_bundle = joblib.load(model_path)
    if "classifier" not in loaded_bundle:
        raise RuntimeError("Saved bundle missing 'classifier' key.")
    if loaded_bundle["feature_dim"] != feat_dim:
        raise RuntimeError(f"Saved bundle dimension mismatch: {loaded_bundle['feature_dim']}.")
    if loaded_bundle["feature_version"] != feat_version:
        raise RuntimeError(f"Saved bundle version mismatch: {loaded_bundle['feature_version']}.")
    if len(loaded_bundle["classes"]) != 13:
        raise RuntimeError(f"Saved bundle class count mismatch: expected 13, got {len(loaded_bundle['classes'])}.")

    app_cfg = AppConfig()
    app_cfg.recognition.model_path = str(model_path)
    recognizer = MLRecognizer(config=app_cfg)

    if not recognizer.is_ready:
        raise RuntimeError("MLRecognizer failed to initialize with saved model bundle.")

    sample_feat = X_test[0]
    result = recognizer.predict(sample_feat)
    logger.info(
        "Integration Verification SUCCESS: MLRecognizer is_ready=%s, predicted label='%s', confidence=%.4f.",
        recognizer.is_ready,
        result.label,
        result.confidence,
    )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_train_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return train_and_evaluate(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    sys.exit(main())
