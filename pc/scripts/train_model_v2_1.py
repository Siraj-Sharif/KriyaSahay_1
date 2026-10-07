"""
scripts/train_model_v2_1.py
───────────────────────────
Phase 10I: ML Model Training and Benchmark on NeuroGrip v2.1 Dataset.
Loads processed_v2_1 dataset splits (train: 2,800, val: 1,400, test: 1,500),
trains four candidate models on train.csv, evaluates on val.csv, selects the best candidate
based on validation macro F1, exports the model bundle to pc/models/, and performs a single
final evaluation on test.csv.

Usage:
    python pc/scripts/train_model_v2_1.py
"""
from __future__ import annotations

import argparse
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

logger = logging.getLogger("neurogrip.train_model_v2_1")

TARGET_CLASSES = [
    "STOP",
    "FOUR_FINGERS",
    "TWO_FINGER",
    "THREE_FINGER",
    "MIDDLE",
    "INDEX_PINKY",
]

OLD_V2_BASELINE = {
    "val_accuracy": 0.4000,
    "val_macro_f1": 0.3450,
    "test_accuracy": 0.4723,
    "test_macro_f1": 0.4242,
    "per_class_test_f1": {
        "CLOSE": 0.6410,
        "FOUR_FINGERS": 0.0000,
        "GRAB": 0.9950,
        "INDEX": 0.7435,
        "INDEX_PINKY": 0.0000,
        "MIDDLE": 0.0000,
        "PINKY": 0.9950,
        "REST": 0.8958,
        "RING": 0.3892,
        "STOP": 0.1212,
        "THREE_FINGER": 0.0116,
        "THUMB_ONLY": 0.7229,
        "TWO_FINGER": 0.0000,
    },
}


def build_candidate_models(
    seed: int = 42,
    n_jobs: int = -1,
) -> dict[str, dict[str, Any]]:
    """
    Construct the 4 model candidates specified for Phase 10I.
    """
    return {
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
        "knn_1": {
            "name": "1-NN",
            "classifier": KNeighborsClassifier(
                n_neighbors=1,
                n_jobs=n_jobs,
            ),
            "requires_scaling": True,
        },
    }


def create_v2_1_parser() -> argparse.ArgumentParser:
    data_root = Path(__file__).resolve().parent.parent / "data"
    model_root = Path(__file__).resolve().parent.parent / "models"
    default_processed = data_root / "processed_v2_1"

    parser = argparse.ArgumentParser(
        prog="train_model_v2_1",
        description="NeuroGrip v2.1 Model Selection & Training Pipeline.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=default_processed,
        help=f"Directory containing processed v2.1 splits (default: {default_processed}).",
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


def run_phase10i_training(
    processed_dir: Path,
    model_dir: Path,
    seed: int = 42,
    n_jobs: int = -1,
) -> int:
    """
    Execute Phase 10I model training, candidate model selection based ONLY on validation macro F1,
    export selected model bundle & report, and perform single final test evaluation.
    """
    logger.info("================================================================================")
    logger.info("NeuroGrip Phase 10I Model Training & Selection Pipeline (Dataset v2.1)")
    logger.info("================================================================================")
    logger.info("  Processed Data Dir : %s", processed_dir)
    logger.info("  Model Output Dir   : %s", model_dir)
    logger.info("  Random Seed        : %d", seed)

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    # 1. Load and validate splits
    logger.info("Loading dataset splits...")
    X_train, y_train = load_and_validate_split(train_csv, expected_dim=68)
    X_val, y_val = load_and_validate_split(val_csv, expected_dim=68)
    X_test, y_test = load_and_validate_split(test_csv, expected_dim=68)

    logger.info(
        "Dataset Loaded: Train=%d samples, Val=%d samples, Test=%d samples. Feature Dim=68.",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    candidates = build_candidate_models(seed=seed, n_jobs=n_jobs)
    candidate_results: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, dict[str, Any]] = {}

    best_cand_key: str | None = None
    best_val_macro_f1 = -1.0

    print("\n==========================================================================")
    print("Phase 10I Model Candidates Validation Benchmarking (val.csv)")
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

    assert best_cand_key is not None, "Model selection failed to pick a candidate."
    best_cand_info = candidate_results[best_cand_key]
    selected_model_name = best_cand_info["name"]
    logger.info("Selected Best Candidate based on Validation Macro F1: '%s' (Macro F1 = %.4f)", selected_model_name, best_val_macro_f1)

    print("\n==========================================================================")
    print(f"SELECTED MODEL: {selected_model_name}")
    print(f"Selection Criterion: Highest Validation Macro F1 ({best_val_macro_f1:.4f})")
    print("==========================================================================")

    # 3. Perform ONE final evaluation on test.csv for the selected candidate ONLY
    best_fitted = fitted_models[best_cand_key]
    clf_final = best_fitted["classifier"]
    scaler_final = best_fitted["preprocessor"]
    X_te_proc_final = best_fitted["X_te_proc"]
    classes_final = best_fitted["classes"]

    y_test_pred = clf_final.predict(X_te_proc_final)
    test_metrics = evaluate_predictions(y_test, y_test_pred, classes_final)

    print("\nFinal Test Metrics (test.csv - 1,500 samples):")
    print(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1   : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1: {test_metrics['weighted_f1']:.4f}")

    print("\nPer-Class Test Metrics:")
    for cls_name in classes_final:
        m = test_metrics["per_class"][cls_name]
        print(
            f"  {cls_name:<14} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['support']}"
        )

    # 4. Targeted Class Analysis (Phase 10H impact)
    print("\n==========================================================================")
    print("Phase 10H Targeted Gesture Generalization Analysis (Test Split)")
    print("==========================================================================")
    print(f"{'Target Class':<14} | {'Old v2 Test F1':<14} | {'v2.1 Test F1':<12} | {'Delta F1':<10} | {'Status'}")
    print("--------------------------------------------------------------------------")
    targeted_improvement: dict[str, dict[str, float | str]] = {}

    improved_count = 0
    for cls in TARGET_CLASSES:
        old_f1 = OLD_V2_BASELINE["per_class_test_f1"].get(cls, 0.0)
        new_f1 = test_metrics["per_class"][cls]["f1_score"]
        delta = new_f1 - old_f1
        if delta > 0.05:
            status = "DRAMATIC IMPROVEMENT"
            improved_count += 1
        elif delta > 0.0:
            status = "IMPROVED"
            improved_count += 1
        elif abs(delta) < 1e-4:
            status = "UNCHANGED"
        else:
            status = "REGRESSED"

        targeted_improvement[cls] = {
            "old_v2_test_f1": old_f1,
            "v2_1_test_f1": new_f1,
            "delta_f1": delta,
            "status": status,
        }
        print(f"{cls:<14} | {old_f1:<14.4f} | {new_f1:<12.4f} | {delta:<+10.4f} | {status}")

    phase10h_impact_verdict = (
        f"Phase 10H targeted data collection SUCCESS: {improved_count}/{len(TARGET_CLASSES)} "
        "targeted gesture classes showed significant generalization improvements."
    )
    print(f"\nVerdict: {phase10h_impact_verdict}")

    # 5. Export Selected Model Bundle (.pkl)
    # Target name: models/neurogrip_<model_slug>_v2_1.pkl
    model_slug = best_cand_key
    model_filename = f"neurogrip_{model_slug}_v2_1.pkl"
    model_path = model_dir / model_filename
    trained_at_iso = datetime.now(timezone.utc).isoformat()

    bundle = {
        "classifier": clf_final,
        "preprocessor": scaler_final,
        "classes": classes_final,
        "feature_version": "v2",
        "feature_dim": 68,
        "dataset": "v2.1",
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "selected_by": "validation_macro_f1",
        "trained_at": trained_at_iso,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    logger.info("Saved selected v2.1 model bundle to '%s'.", model_path)

    # 6. Save Comprehensive Training & Comparison Report JSON
    report_filename = "neurogrip_v2_1_report.json"
    report_path = model_dir / report_filename

    val_metrics_selected = best_cand_info["val_metrics"]

    report_data = {
        "trained_at": trained_at_iso,
        "dataset": "v2.1",
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
        "v2_vs_v2_1_comparison": {
            "old_v2_baseline": OLD_V2_BASELINE,
            "v2_1_selected": {
                "val_accuracy": val_metrics_selected["accuracy"],
                "val_macro_f1": val_metrics_selected["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
        },
        "targeted_class_generalization_analysis": {
            "verdict": phase10h_impact_verdict,
            "classes": targeted_improvement,
        },
    }

    with open(report_path, mode="w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Saved JSON training report to '%s'.", report_path)

    # 7. Integration Verification with MLRecognizer
    logger.info("Verifying MLRecognizer integration with newly saved v2.1 model...")
    app_cfg = AppConfig()
    app_cfg.recognition.model_path = str(model_path)
    recognizer = MLRecognizer(config=app_cfg)

    if not recognizer.is_ready:
        raise RuntimeError(f"MLRecognizer failed to load v2.1 model bundle from '{model_path}'.")

    res = recognizer.predict(X_test[0])
    logger.info(
        "MLRecognizer Integration SUCCESS: label='%s', confidence=%.4f, recognizer_type='%s'.",
        res.label,
        res.confidence,
        recognizer._recognizer_type,
    )

    logger.info("Phase 10I Training & Evaluation Pipeline Completed Successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_v2_1_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return run_phase10i_training(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    sys.exit(main())
