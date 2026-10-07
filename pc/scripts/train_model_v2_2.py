"""
scripts/train_model_v2_2.py
───────────────────────────
Phase 10I.2: ML Model Training and Evaluation on NeuroGrip v2.2 Dataset.
Loads processed_v2_2 dataset splits (train: 2,800, val: 1,600, test: 1,700),
trains four candidate models on train.csv, evaluates on val.csv, selects the best candidate
based on validation macro F1, exports the model bundle to pc/models/, and performs a single
final evaluation on test.csv.

Usage:
    python pc/scripts/train_model_v2_2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure src and scripts directories are in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.recognition.ml_recognizer import MLRecognizer
from train_model import OFFICIAL_CLASSES, evaluate_predictions, load_and_validate_split

logger = logging.getLogger("neurogrip.train_model_v2_2")

V2_1_BASELINE = {
    "val_accuracy": 0.7814,
    "val_macro_f1": 0.7324,
    "test_accuracy": 0.8033,
    "test_macro_f1": 0.7822,
    "per_class_test_f1": {
        "CLOSE": 0.9848,
        "FOUR_FINGERS": 0.9755,
        "GRAB": 0.9950,
        "INDEX": 0.9071,
        "INDEX_PINKY": 1.0000,
        "MIDDLE": 0.3223,
        "PINKY": 1.0000,
        "REST": 0.0198,
        "RING": 0.6087,
        "STOP": 0.6667,
        "THREE_FINGER": 0.8824,
        "THUMB_ONLY": 0.9802,
        "TWO_FINGER": 0.8264,
    },
    "per_class_test_recall": {
        "CLOSE": 0.9700,
        "FOUR_FINGERS": 0.9950,
        "GRAB": 0.9900,
        "INDEX": 0.8300,
        "INDEX_PINKY": 1.0000,
        "MIDDLE": 0.1950,
        "PINKY": 1.0000,
        "REST": 0.0100,
        "RING": 0.9800,
        "STOP": 1.0000,
        "THREE_FINGER": 0.9000,
        "THUMB_ONLY": 0.9900,
        "TWO_FINGER": 1.0000,
    },
}


def build_candidate_models_v2_2(
    seed: int = 42,
    n_jobs: int = -1,
) -> dict[str, dict[str, Any]]:
    """
    Construct the 4 model candidates specified for Phase 10I.2.
    """
    return {
        "extra_trees": {
            "name": "Extra Trees",
            "classifier": ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            ),
            "requires_scaling": False,
        },
        "random_forest": {
            "name": "Random Forest",
            "classifier": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            ),
            "requires_scaling": False,
        },
        "knn_1": {
            "name": "1-NN",
            "classifier": KNeighborsClassifier(
                n_neighbors=1,
                n_jobs=n_jobs,
            ),
            "requires_scaling": True,
        },
        "rbf_svm": {
            "name": "RBF SVM",
            "classifier": SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
            "requires_scaling": True,
        },
    }


def create_v2_2_train_parser() -> argparse.ArgumentParser:
    data_root = Path(__file__).resolve().parent.parent / "data"
    model_root = Path(__file__).resolve().parent.parent / "models"
    default_processed = data_root / "processed_v2_2"

    parser = argparse.ArgumentParser(
        prog="train_model_v2_2",
        description="NeuroGrip v2.2 Model Selection & Evaluation Pipeline.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help=f"Directory containing processed v2.2 splits (default: {default_processed}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=model_root,
        help=f"Output directory for model bundle and JSON report (default: {model_root}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for estimators (-1 uses all cores, default: -1).",
    )
    return parser


def evaluate_unseen_middle_session(
    test_csv: Path,
    clf: Any,
    scaler: Any | None,
    target_session_id: str = "cf9b7593",
) -> dict[str, Any]:
    """
    Specifically evaluate model predictions on the new unseen MIDDLE test session cf9b7593.
    """
    labels = []
    features = []
    sess_ids = []

    with open(test_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            labels.append(row[0].strip())
            sess_ids.append(row[2].strip())
            features.append([float(x) for x in row[4:]])

    X_all = np.array(features, dtype=np.float32)
    y_all = np.array(labels, dtype=str)

    target_mask = (np.array(sess_ids) == target_session_id)
    if not np.any(target_mask):
        return {"session_id": target_session_id, "found": False}

    X_target = X_all[target_mask]
    y_target = y_all[target_mask]

    if scaler is not None:
        X_target_proc = scaler.transform(X_target)
    else:
        X_target_proc = X_target

    y_pred = clf.predict(X_target_proc)
    correct = int(np.sum(y_pred == y_target))
    total = int(len(y_target))

    from collections import Counter
    pred_counts = dict(Counter(y_pred))

    return {
        "session_id": target_session_id,
        "found": True,
        "total_samples": total,
        "correct_samples": correct,
        "recall": float(correct / total) if total > 0 else 0.0,
        "predicted_distribution": pred_counts,
    }


def run_phase10i_2_training(
    processed_dir: Path,
    model_dir: Path,
    seed: int = 42,
    n_jobs: int = -1,
) -> int:
    """
    Execute Phase 10I.2 v2.2 model training, candidate selection based ONLY on validation macro F1,
    export selected model bundle & report, evaluate once on test.csv, and compare against v2.1 baseline.
    """
    logger.info("================================================================================")
    logger.info("NeuroGrip Phase 10I.2 Model Training & Selection Pipeline (Dataset v2.2)")
    logger.info("================================================================================")

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    # 1. Load splits
    X_train, y_train = load_and_validate_split(train_csv, expected_dim=68)
    X_val, y_val = load_and_validate_split(val_csv, expected_dim=68)
    X_test, y_test = load_and_validate_split(test_csv, expected_dim=68)

    logger.info(
        "Dataset v2.2 Loaded: Train=%d samples, Val=%d samples, Test=%d samples. Feature Dim=68.",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    candidates = build_candidate_models_v2_2(seed=seed, n_jobs=n_jobs)
    candidate_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, dict[str, Any]] = {}

    best_cand_key: str | None = None
    best_val_macro_f1 = -1.0

    print("\n==========================================================================")
    print("Phase 10I.2 Model Candidates Validation Benchmarking (v2.2 val.csv)")
    print("==========================================================================")
    print(f"{'Candidate':<16} | {'Val Acc':<8} | {'Val Macro F1':<13} | {'Val W-F1':<9}")
    print("--------------------------------------------------------------------------")

    # 2. Train candidates ONLY on train.csv and evaluate on val.csv
    for cand_key, cand in candidates.items():
        name = cand["name"]
        raw_clf = cand["classifier"]
        requires_scaling = cand["requires_scaling"]

        if requires_scaling:
            scaler = StandardScaler()
            X_tr_proc = scaler.fit_transform(X_train)
            X_val_proc = scaler.transform(X_val)
            X_te_proc = scaler.transform(X_test)
        else:
            scaler = None
            X_tr_proc = X_train
            X_val_proc = X_val
            X_te_proc = X_test

        raw_clf.fit(X_tr_proc, y_train)
        classes_list = list(raw_clf.classes_)

        y_val_pred = raw_clf.predict(X_val_proc)
        val_metrics = evaluate_predictions(y_val, y_val_pred, classes_list)

        candidate_results[cand_key] = {
            "name": name,
            "requires_scaling": requires_scaling,
            "classes": classes_list,
            "val_metrics": val_metrics,
        }

        fitted_models[cand_key] = {
            "classifier": raw_clf,
            "preprocessor": scaler,
            "X_te_proc": X_te_proc,
            "classes": classes_list,
        }

        val_macro_f1 = val_metrics["macro_f1"]
        print(
            f"{name:<16} | {val_metrics['accuracy']:<8.4f} | {val_macro_f1:<13.4f} | {val_metrics['weighted_f1']:<9.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_cand_key = cand_key

    assert best_cand_key is not None
    best_cand_info = candidate_results[best_cand_key]
    selected_model_name = best_cand_info["name"]
    logger.info("Selected Best Candidate based on Validation Macro F1: '%s' (Macro F1 = %.4f)", selected_model_name, best_val_macro_f1)

    print("\n==========================================================================")
    print(f"SELECTED MODEL: {selected_model_name}")
    print(f"Selection Criterion: Highest Validation Macro F1 ({best_val_macro_f1:.4f})")
    print("==========================================================================")

    # 3. Perform ONE final evaluation on test.csv for selected candidate
    best_fitted = fitted_models[best_cand_key]
    clf_final = best_fitted["classifier"]
    scaler_final = best_fitted["preprocessor"]
    X_te_proc_final = best_fitted["X_te_proc"]
    classes_final = best_fitted["classes"]

    y_test_pred = clf_final.predict(X_te_proc_final)
    test_metrics = evaluate_predictions(y_test, y_test_pred, classes_final)

    # Evaluate specifically on unseen new MIDDLE test session cf9b7593
    cf9b7593_analysis = evaluate_unseen_middle_session(
        test_csv, clf_final, scaler_final, target_session_id="cf9b7593"
    )

    print("\nFinal Test Metrics (v2.2 test.csv - 1,700 samples):")
    print(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print("\nPer-Class Test Metrics (v2.2):")
    for cls_name in classes_final:
        m = test_metrics["per_class"][cls_name]
        print(
            f"  {cls_name:<14} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['support']}"
        )

    # 4. Save Selected Model Bundle (.pkl) -> models/NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl
    model_filename = "NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl"
    model_path = model_dir / model_filename
    trained_at_iso = datetime.now(timezone.utc).isoformat()

    bundle = {
        "classifier": clf_final,
        "preprocessor": scaler_final,
        "classes": classes_final,
        "feature_version": "v2",
        "feature_dim": 68,
        "dataset": "v2.2",
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "selected_by": "validation_macro_f1",
        "trained_at": trained_at_iso,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    logger.info("Saved selected v2.2 model bundle to '%s'.", model_path)

    # 5. Save JSON Training Report
    report_filename = "neurogrip_v2_2_report.json"
    report_path = model_dir / report_filename
    val_metrics_selected = best_cand_info["val_metrics"]

    report_data = {
        "trained_at": trained_at_iso,
        "dataset": "v2.2",
        "feature_version": "v2",
        "feature_dim": 68,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "selected_candidate_key": best_cand_key,
        "selected_model_name": selected_model_name,
        "selected_by": "validation_macro_f1",
        "saved_model_file": model_filename,
        "candidates_validation_benchmark": {
            k: {
                "name": v["name"],
                "val_accuracy": v["val_metrics"]["accuracy"],
                "val_macro_precision": v["val_metrics"]["macro_precision"],
                "val_macro_recall": v["val_metrics"]["macro_recall"],
                "val_macro_f1": v["val_metrics"]["macro_f1"],
                "val_weighted_f1": v["val_metrics"]["weighted_f1"],
            }
            for k, v in candidate_results.items()
        },
        "selected_model_validation_metrics": {
            "accuracy": val_metrics_selected["accuracy"],
            "macro_precision": val_metrics_selected["macro_precision"],
            "macro_recall": val_metrics_selected["macro_recall"],
            "macro_f1": val_metrics_selected["macro_f1"],
            "weighted_f1": val_metrics_selected["weighted_f1"],
            "per_class": val_metrics_selected["per_class"],
            "confusion_matrix": val_metrics_selected["confusion_matrix"],
        },
        "selected_model_test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "v2_1_vs_v2_2_comparison": {
            "v2_1_baseline": V2_1_BASELINE,
            "v2_2_selected": {
                "val_accuracy": val_metrics_selected["accuracy"],
                "val_macro_f1": val_metrics_selected["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
        },
        "unseen_middle_session_cf9b7593_analysis": cf9b7593_analysis,
    }

    with open(report_path, mode="w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved JSON training report to '%s'.", report_path)

    # 6. Integration verification with MLRecognizer
    logger.info("Verifying MLRecognizer integration with newly saved v2.2 model...")
    app_cfg = AppConfig()
    app_cfg.recognition.model_path = str(model_path)
    recognizer = MLRecognizer(config=app_cfg)

    if not recognizer.is_ready:
        raise RuntimeError(f"MLRecognizer failed to load v2.2 model bundle from '{model_path}'.")

    res = recognizer.predict(X_test[0])
    logger.info(
        "MLRecognizer Integration SUCCESS: label='%s', confidence=%.4f, recognizer_type='%s'.",
        res.label,
        res.confidence,
        recognizer._recognizer_type,
    )

    logger.info("Phase 10I.2 Training & Evaluation Pipeline Completed Successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_v2_2_train_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return run_phase10i_2_training(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    sys.exit(main())
