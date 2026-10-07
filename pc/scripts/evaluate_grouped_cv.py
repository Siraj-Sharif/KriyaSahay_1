"""
scripts/evaluate_grouped_cv.py
──────────────────────────────
Phase 10G: Session-Aware Cross-Validation for NeuroGrip v2 Gesture Recognizer.

Loads pc/data/processed_v2/train.csv ONLY.
Uses session_id as the grouping variable so no session appears in both training and validation folds.
Evaluates 6 candidate gesture recognition models using 5-fold GroupKFold CV.

Save artifacts:
  - pc/models/neurogrip_v2_grouped_cv_report.json
  - pc/models/neurogrip_v2_grouped_cv_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure src directory is accessible
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.commands.definitions import NeuroGripCommand

FEATURE_DIM = 68
GEOMETRY_FEATURE_INDICES = list(range(40, 68))  # 28-D: [40..67]
OFFICIAL_CLASSES: list[str] = sorted([c.value for c in NeuroGripCommand])


@dataclass
class TrainRecord:
    label: str
    handedness: str
    session_id: str
    sample_id: int
    features: np.ndarray  # Shape (68,)


def load_train_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TrainRecord]]:
    """Load train.csv and return X (68-D), y (labels), groups (session_id), and records list."""
    records: list[TrainRecord] = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            hnd = row["handedness"]
            sess = row["session_id"]
            samp_id = int(row["sample_id"])
            feats = np.array([float(row[f"feature_{i}"]) for i in range(FEATURE_DIM)], dtype=np.float32)
            records.append(TrainRecord(label=lbl, handedness=hnd, session_id=sess, sample_id=samp_id, features=feats))

    X = np.array([r.features for r in records], dtype=np.float32)
    y = np.array([r.label for r in records])
    groups = np.array([r.session_id for r in records])
    return X, y, groups, records


def create_candidate_model(candidate_name: str) -> Any:
    """Factory function for creating fresh candidate model instances."""
    if candidate_name == "Random Forest — full 68D":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    elif candidate_name == "Extra Trees — full 68D":
        return ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    elif candidate_name == "RBF SVM — full 68D":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42, class_weight="balanced"))
    elif candidate_name == "Random Forest — geometry 28D [40..67]":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    elif candidate_name == "RBF SVM — geometry 28D [40..67]":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42, class_weight="balanced"))
    elif candidate_name == "1-NN — full 68D":
        return KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    else:
        raise ValueError(f"Unknown candidate model: {candidate_name}")


def run_session_aware_cv(train_csv_path: Path | str, n_splits: int = 5) -> dict[str, Any]:
    t_path = Path(train_csv_path)
    X, y, groups, _ = load_train_csv(t_path)

    candidates = [
        "Random Forest — full 68D",
        "Extra Trees — full 68D",
        "RBF SVM — full 68D",
        "Random Forest — geometry 28D [40..67]",
        "RBF SVM — geometry 28D [40..67]",
        "1-NN — full 68D",
    ]

    gkf = GroupKFold(n_splits=n_splits)

    results: dict[str, Any] = {}

    for cand_name in candidates:
        use_geom = "geometry 28D" in cand_name
        X_in = X[:, GEOMETRY_FEATURE_INDICES] if use_geom else X

        fold_accuracies: list[float] = []
        fold_macro_f1s: list[float] = []
        fold_per_class_f1s: list[dict[str, float]] = []
        fold_details: list[dict[str, Any]] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_in, y, groups=groups), start=1):
            X_tr_f, y_tr_f = X_in[tr_idx], y[tr_idx]
            X_val_f, y_val_f = X_in[val_idx], y[val_idx]
            val_groups_f = sorted(list(set(groups[val_idx])))

            model = create_candidate_model(cand_name)
            model.fit(X_tr_f, y_tr_f)
            y_pred_f = model.predict(X_val_f)

            fold_acc = float(accuracy_score(y_val_f, y_pred_f))
            fold_f1 = float(f1_score(y_val_f, y_pred_f, average="macro", zero_division=0))

            # Per-class F1 for this fold
            classes_in_val = sorted(list(set(y_val_f)))
            per_class_f1 = f1_score(y_val_f, y_pred_f, average=None, labels=classes_in_val, zero_division=0)
            pc_f1_dict = {cls_name: float(per_class_f1[i]) for i, cls_name in enumerate(classes_in_val)}

            fold_accuracies.append(fold_acc)
            fold_macro_f1s.append(fold_f1)
            fold_per_class_f1s.append(pc_f1_dict)

            fold_details.append(
                {
                    "fold": fold_idx,
                    "val_sessions": val_groups_f,
                    "accuracy": fold_acc,
                    "macro_f1": fold_f1,
                    "per_class_f1": pc_f1_dict,
                }
            )

        mean_acc = float(np.mean(fold_accuracies))
        std_acc = float(np.std(fold_accuracies))
        mean_f1 = float(np.mean(fold_macro_f1s))
        std_f1 = float(np.std(fold_macro_f1s))

        # Mean per-class F1 across all folds where the class appeared
        mean_per_class_f1: dict[str, float] = {}
        for cls_name in OFFICIAL_CLASSES:
            cls_f1_vals = [f_dict[cls_name] for f_dict in fold_per_class_f1s if cls_name in f_dict]
            mean_per_class_f1[cls_name] = float(np.mean(cls_f1_vals)) if cls_f1_vals else 0.0

        results[cand_name] = {
            "candidate_name": cand_name,
            "feature_dim": 28 if use_geom else 68,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "fold_accuracies": fold_accuracies,
            "fold_macro_f1s": fold_macro_f1s,
            "mean_per_class_f1": mean_per_class_f1,
            "fold_details": fold_details,
        }

    # Primary ranking by mean_macro_f1 desc, then mean_accuracy desc
    ranked_candidates = sorted(
        results.keys(),
        key=lambda k: (results[k]["mean_macro_f1"], results[k]["mean_accuracy"]),
        reverse=True,
    )

    return {
        "ranked_candidates": ranked_candidates,
        "results": results,
        "n_splits": n_splits,
        "total_unique_sessions": len(set(groups)),
        "total_train_samples": len(X),
    }


def export_cv_reports(
    cv_output: dict[str, Any],
    json_path: Path | str = "pc/models/neurogrip_v2_grouped_cv_report.json",
    csv_path: Path | str = "pc/models/neurogrip_v2_grouped_cv_summary.csv",
) -> None:
    j_path = Path(json_path)
    c_path = Path(csv_path)

    j_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    with open(j_path, mode="w", encoding="utf-8") as f:
        json.dump(cv_output, f, indent=2)

    with open(c_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "candidate_name",
            "feature_dim",
            "mean_macro_f1",
            "std_macro_f1",
            "mean_accuracy",
            "std_accuracy",
            "fold_1_f1",
            "fold_2_f1",
            "fold_3_f1",
            "fold_4_f1",
            "fold_5_f1",
            "stop_mean_f1",
            "four_fingers_mean_f1",
            "middle_mean_f1",
            "two_finger_mean_f1",
        ])

        for rank_idx, cand_name in enumerate(cv_output["ranked_candidates"], start=1):
            c_data = cv_output["results"][cand_name]
            f1s = c_data["fold_macro_f1s"]
            pc = c_data["mean_per_class_f1"]

            writer.writerow([
                rank_idx,
                cand_name,
                c_data["feature_dim"],
                f"{c_data['mean_macro_f1']:.4f}",
                f"{c_data['std_macro_f1']:.4f}",
                f"{c_data['mean_accuracy']:.4f}",
                f"{c_data['std_accuracy']:.4f}",
                f"{f1s[0]:.4f}",
                f"{f1s[1]:.4f}",
                f"{f1s[2]:.4f}",
                f"{f1s[3]:.4f}",
                f"{f1s[4]:.4f}",
                f"{pc.get('STOP', 0.0):.4f}",
                f"{pc.get('FOUR_FINGERS', 0.0):.4f}",
                f"{pc.get('MIDDLE', 0.0):.4f}",
                f"{pc.get('TWO_FINGER', 0.0):.4f}",
            ])


def print_cv_report(cv_output: dict[str, Any]) -> None:
    print("=" * 110)
    print("NeuroGrip Phase 10G — Session-Aware GroupKFold Cross-Validation Report")
    print("=" * 110)

    print("\nCANDIDATE RANKING (GroupKFold n_splits=5 on train.csv session_ids)")
    print(f"  {'Rank':4} | {'Candidate Model':38} | {'Feature Dim':11} | {'Mean Macro F1 ± Std':22} | {'Mean Accuracy ± Std':22}")
    print("  " + "-" * 105)

    for rank_idx, cand_name in enumerate(cv_output["ranked_candidates"], start=1):
        c_data = cv_output["results"][cand_name]
        f1_str = f"{c_data['mean_macro_f1']:.4f} ± {c_data['std_macro_f1']:.4f}"
        acc_str = f"{c_data['mean_accuracy']:.4f} ± {c_data['std_accuracy']:.4f}"
        print(f"  {rank_idx:4d} | {cand_name:38} | {c_data['feature_dim']:11d} | {f1_str:22} | {acc_str:22}")

    print("\nFOLD-BY-FOLD MACRO F1 SCORES")
    print(f"  {'Candidate Model':38} | {'Fold 1':7} | {'Fold 2':7} | {'Fold 3':7} | {'Fold 4':7} | {'Fold 5':7}")
    print("  " + "-" * 85)
    for cand_name in cv_output["ranked_candidates"]:
        f1s = cv_output["results"][cand_name]["fold_macro_f1s"]
        print(f"  {cand_name:38} | {f1s[0]:7.4f} | {f1s[1]:7.4f} | {f1s[2]:7.4f} | {f1s[3]:7.4f} | {f1s[4]:7.4f}")

    print("\nPER-CLASS MEAN MACRO F1 (TOP 3 MODELS)")
    top_3 = cv_output["ranked_candidates"][:3]
    print(f"  {'Class Label':15} | " + " | ".join([f"{m[:20]:20}" for m in top_3]))
    print("  " + "-" * 85)
    for cls in OFFICIAL_CLASSES:
        vals = [f"{cv_output['results'][m]['mean_per_class_f1'].get(cls, 0.0):.4f}" for m in top_3]
        print(f"  {cls:15} | " + " | ".join([f"{v:20}" for v in vals]))

    top_winner = cv_output["ranked_candidates"][0]
    print("\n" + "=" * 110)
    print(f"WINNING MODEL (Session-Aware GroupKFold CV): {top_winner}")
    print(f"  Mean Macro F1: {cv_output['results'][top_winner]['mean_macro_f1']:.4f} ± {cv_output['results'][top_winner]['std_macro_f1']:.4f}")
    print(f"  Mean Accuracy: {cv_output['results'][top_winner]['mean_accuracy']:.4f} ± {cv_output['results'][top_winner]['std_accuracy']:.4f}")
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroGrip Phase 10G Session-Aware GroupKFold Cross-Validation")
    parser.add_argument("--train-csv", type=str, default="pc/data/processed_v2/train.csv", help="Path to train.csv")
    parser.add_argument("--json-output", type=str, default="pc/models/neurogrip_v2_grouped_cv_report.json", help="JSON report output")
    parser.add_argument("--csv-output", type=str, default="pc/models/neurogrip_v2_grouped_cv_summary.csv", help="CSV report output")
    args = parser.parse_args()

    cv_output = run_session_aware_cv(args.train_csv)
    export_cv_reports(cv_output, args.json_output, args.csv_output)
    print_cv_report(cv_output)


if __name__ == "__main__":
    main()
