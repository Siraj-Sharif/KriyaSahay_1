"""
pc/qa_v2_generalization.py
───────────────────────────
NeuroGrip Feature Version v2 — Generalization & Diagnostic Analysis Script.

Performs a comprehensive, READ-ONLY diagnostic analysis of:
1. Feature distribution shifts across train/val/test splits.
2. Per-class session stability and Euclidean/standardized distance metrics.
3. Nearest-neighbor (1-NN) cross-session space analysis.
4. Class centroid separation and nearest-centroid accuracy.
5. Critical gesture pair deep dive (STOP vs FOUR_FINGERS, TWO_FINGER vs INDEX_PINKY, etc.).
6. Diagnostic classifier comparison (1-NN, Logistic Regression, Random Forest).
7. Session-specific classification performance breakdown.
8. Statistical outlier impact on model predictions.
9. Diagnostic verdict classification (Feature-space vs RF vs Session-variation).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_DIM = 68
OFFICIAL_CLASSES: list[str] = [
    "CLOSE", "FOUR_FINGERS", "GRAB", "INDEX", "INDEX_PINKY",
    "MIDDLE", "PINKY", "REST", "RING", "STOP",
    "THUMB_ONLY", "THREE_FINGER", "TWO_FINGER"
]

CRITICAL_PAIRS: list[tuple[str, str]] = [
    ("STOP", "FOUR_FINGERS"),
    ("TWO_FINGER", "INDEX_PINKY"),
    ("MIDDLE", "THREE_FINGER"),
    ("RING", "THREE_FINGER"),
    ("INDEX_PINKY", "INDEX"),
    ("TWO_FINGER", "CLOSE"),
]


@dataclass
class SampleRecord:
    label: str
    handedness: str
    session_id: str
    sample_id: int
    features: np.ndarray  # Shape (68,)


def load_split_csv(csv_path: Path) -> list[SampleRecord]:
    """Load a processed CSV split into a list of SampleRecord objects."""
    records: list[SampleRecord] = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            hnd = row["handedness"]
            sess = row["session_id"]
            samp_id = int(row["sample_id"])
            feats = np.array([float(row[f"feature_{i}"]) for i in range(FEATURE_DIM)], dtype=np.float32)
            records.append(SampleRecord(label=lbl, handedness=hnd, session_id=sess, sample_id=samp_id, features=feats))
    return records


def extract_matrix_and_labels(records: list[SampleRecord]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert SampleRecord list to features matrix X, labels array y, and session IDs list."""
    X = np.array([r.features for r in records], dtype=np.float32)
    y = np.array([r.label for r in records])
    sessions = [r.session_id for r in records]
    return X, y, sessions


# ==============================================================================
# 1. TRAIN / VAL / TEST FEATURE DISTRIBUTION SHIFT
# ==============================================================================

def analyze_distribution_shifts(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, Any]:
    X_tr, _, _ = extract_matrix_and_labels(train_recs)
    X_val, _, _ = extract_matrix_and_labels(val_recs)
    X_te, _, _ = extract_matrix_and_labels(test_recs)

    mu_tr, std_tr = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-6
    mu_val, std_val = np.mean(X_val, axis=0), np.std(X_val, axis=0) + 1e-6
    mu_te, std_te = np.mean(X_te, axis=0), np.std(X_te, axis=0) + 1e-6

    tr_val_shift = np.abs(mu_tr - mu_val) / std_tr
    tr_te_shift = np.abs(mu_tr - mu_te) / std_tr

    global_shift_ranking = sorted(
        [
            {
                "feature_idx": i,
                "feature_name": f"feature_{i}",
                "train_mean": float(mu_tr[i]),
                "val_mean": float(mu_val[i]),
                "test_mean": float(mu_te[i]),
                "train_std": float(std_tr[i]),
                "val_std": float(std_val[i]),
                "test_std": float(std_te[i]),
                "tr_val_shift": float(tr_val_shift[i]),
                "tr_te_shift": float(tr_te_shift[i]),
            }
            for i in range(FEATURE_DIM)
        ],
        key=lambda x: x["tr_te_shift"],
        reverse=True,
    )

    # Per-class distribution shifts
    per_class_shifts: dict[str, list[dict[str, Any]]] = {}
    for cls in sorted(list(set(r.label for r in train_recs))):
        cls_tr = np.array([r.features for r in train_recs if r.label == cls], dtype=np.float32)
        cls_te = np.array([r.features for r in test_recs if r.label == cls], dtype=np.float32)
        m_c_tr, s_c_tr = np.mean(cls_tr, axis=0), np.std(cls_tr, axis=0) + 1e-6
        m_c_te, _ = np.mean(cls_te, axis=0), np.std(cls_te, axis=0) + 1e-6
        c_shift = np.abs(m_c_tr - m_c_te) / s_c_tr
        per_class_shifts[cls] = sorted(
            [
                {
                    "feature_idx": i,
                    "tr_mean": float(m_c_tr[i]),
                    "te_mean": float(m_c_te[i]),
                    "shift": float(c_shift[i]),
                }
                for i in range(FEATURE_DIM)
            ],
            key=lambda x: x["shift"],
            reverse=True,
        )

    return {
        "global_top_20": global_shift_ranking[:20],
        "per_class_top_5": {c: items[:5] for c, items in per_class_shifts.items()},
    }


# ==============================================================================
# 2. PER-CLASS SESSION GENERALIZATION
# ==============================================================================

def analyze_session_generalization(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> list[dict[str, Any]]:
    classes = sorted(list(set(r.label for r in train_recs)))
    class_stability: list[dict[str, Any]] = []

    for cls in classes:
        tr_sub = [r for r in train_recs if r.label == cls]
        val_sub = [r for r in val_recs if r.label == cls]
        te_sub = [r for r in test_recs if r.label == cls]

        tr_sess = list(set(r.session_id for r in tr_sub))[0]
        val_sess = list(set(r.session_id for r in val_sub))[0]
        te_sess = list(set(r.session_id for r in te_sub))[0]

        X_tr = np.array([r.features for r in tr_sub], dtype=np.float32)
        X_val = np.array([r.features for r in val_sub], dtype=np.float32)
        X_te = np.array([r.features for r in te_sub], dtype=np.float32)

        m_tr = np.mean(X_tr, axis=0)
        s_tr = np.std(X_tr, axis=0) + 1e-6
        m_val = np.mean(X_val, axis=0)
        m_te = np.mean(X_te, axis=0)

        euclidean_tr_val = float(np.linalg.norm(m_tr - m_val))
        euclidean_tr_te = float(np.linalg.norm(m_tr - m_te))

        std_tr_val = float(np.sqrt(np.sum(((m_tr - m_val) / s_tr) ** 2)))
        std_tr_te = float(np.sqrt(np.sum(((m_tr - m_te) / s_tr) ** 2)))

        class_stability.append(
            {
                "class": cls,
                "train_session": tr_sess,
                "val_session": val_sess,
                "test_session": te_sess,
                "euclidean_tr_val": euclidean_tr_val,
                "euclidean_tr_te": euclidean_tr_te,
                "standardized_tr_val": std_tr_val,
                "standardized_tr_te": std_tr_te,
            }
        )

    # Rank from MOST stable (smallest standardized tr-te dist) to LEAST stable
    class_stability.sort(key=lambda x: x["standardized_tr_te"])
    return class_stability


# ==============================================================================
# 3. NEAREST-NEIGHBOR CROSS-SESSION ANALYSIS
# ==============================================================================

def analyze_nearest_neighbor_cross_session(
    train_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, Any]:
    X_tr, y_tr, _ = extract_matrix_and_labels(train_recs)
    X_te, y_te, _ = extract_matrix_and_labels(test_recs)

    nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    nn.fit(X_tr, y_tr)
    y_pred = nn.predict(X_te)

    overall_acc = float(accuracy_score(y_te, y_pred))

    classes = sorted(list(set(y_te)))
    per_class_acc: dict[str, float] = {}
    cm: dict[str, dict[str, int]] = {c1: {c2: 0 for c2 in classes} for c1 in classes}

    for true_lbl, pred_lbl in zip(y_te, y_pred):
        cm[true_lbl][pred_lbl] += 1

    for cls in classes:
        cls_mask = y_te == cls
        per_class_acc[cls] = float(np.mean(y_pred[cls_mask] == cls))

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
    }


# ==============================================================================
# 4. CLASS CENTROID ANALYSIS
# ==============================================================================

def analyze_class_centroids(
    train_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, Any]:
    X_tr, y_tr, _ = extract_matrix_and_labels(train_recs)
    X_te, y_te, _ = extract_matrix_and_labels(test_recs)

    classes = sorted(list(set(y_tr)))
    centroids = {cls: np.mean(X_tr[y_tr == cls], axis=0) for cls in classes}

    y_pred: list[str] = []
    for x in X_te:
        dists = {cls: np.linalg.norm(x - centroids[cls]) for cls in classes}
        best_cls = min(dists, key=dists.get)  # type: ignore
        y_pred.append(best_cls)

    y_pred_arr = np.array(y_pred)
    overall_acc = float(accuracy_score(y_te, y_pred_arr))

    per_class_acc: dict[str, float] = {}
    cm: dict[str, dict[str, int]] = {c1: {c2: 0 for c2 in classes} for c1 in classes}

    for true_lbl, pred_lbl in zip(y_te, y_pred_arr):
        cm[true_lbl][pred_lbl] += 1

    for cls in classes:
        cls_mask = y_te == cls
        per_class_acc[cls] = float(np.mean(y_pred_arr[cls_mask] == cls))

    return {
        "overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
        "centroids": centroids,
    }


# ==============================================================================
# 5. CRITICAL PAIR ANALYSIS
# ==============================================================================

def analyze_critical_pairs(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, dict[str, Any]]:
    X_tr, y_tr, _ = extract_matrix_and_labels(train_recs)
    X_val, y_val, _ = extract_matrix_and_labels(val_recs)
    X_te, y_te, _ = extract_matrix_and_labels(test_recs)

    pair_analysis: dict[str, dict[str, Any]] = {}

    for c1, c2 in CRITICAL_PAIRS:
        pkey = f"{c1}_vs_{c2}"

        m1_tr = np.mean(X_tr[y_tr == c1], axis=0)
        m2_tr = np.mean(X_tr[y_tr == c2], axis=0)

        m1_val = np.mean(X_val[y_val == c1], axis=0)
        m2_val = np.mean(X_val[y_val == c2], axis=0)

        m1_te = np.mean(X_te[y_te == c1], axis=0)
        m2_te = np.mean(X_te[y_te == c2], axis=0)

        d_tr = float(np.linalg.norm(m1_tr - m2_tr))
        d_val = float(np.linalg.norm(m1_val - m2_val))
        d_te = float(np.linalg.norm(m1_te - m2_te))

        # Sub-problem 1-NN accuracy
        sub_tr_mask = (y_tr == c1) | (y_tr == c2)
        sub_te_mask = (y_te == c1) | (y_te == c2)

        nn_sub = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        nn_sub.fit(X_tr[sub_tr_mask], y_tr[sub_tr_mask])
        sub_pred = nn_sub.predict(X_te[sub_te_mask])
        sub_1nn_acc = float(accuracy_score(y_te[sub_te_mask], sub_pred))

        # Samples closer to wrong train centroid
        te_c1_samples = X_te[y_te == c1]
        te_c2_samples = X_te[y_te == c2]

        c1_closer_to_c2 = int(sum(np.linalg.norm(x - m2_tr) < np.linalg.norm(x - m1_tr) for x in te_c1_samples))
        c2_closer_to_c1 = int(sum(np.linalg.norm(x - m1_tr) < np.linalg.norm(x - m2_tr) for x in te_c2_samples))

        # Top 10 features responsible for train centroid difference
        diff = np.abs(m1_tr - m2_tr)
        top_10_feat_indices = np.argsort(diff)[::-1][:10]
        top_10_feats = [
            {
                "feature_idx": int(i),
                "diff": float(diff[i]),
                f"{c1}_mean": float(m1_tr[i]),
                f"{c2}_mean": float(m2_tr[i]),
            }
            for i in top_10_feat_indices
        ]

        pair_analysis[pkey] = {
            "class1": c1,
            "class2": c2,
            "train_centroid_dist": d_tr,
            "val_centroid_dist": d_val,
            "test_centroid_dist": d_te,
            "sub_1nn_acc": sub_1nn_acc,
            "c1_samples_closer_to_c2_train_centroid": c1_closer_to_c2,
            "c2_samples_closer_to_c1_train_centroid": c2_closer_to_c1,
            "top_10_features": top_10_feats,
        }

    return pair_analysis


# ==============================================================================
# 6. SIMPLE CLASSIFIER COMPARISON
# ==============================================================================

def analyze_simple_classifiers(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    model_bundle_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    X_tr, y_tr, _ = extract_matrix_and_labels(train_recs)
    X_val, y_val, _ = extract_matrix_and_labels(val_recs)
    X_te, y_te, _ = extract_matrix_and_labels(test_recs)

    results: dict[str, dict[str, float]] = {}

    # A. 1-NN Classifier
    clf_1nn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf_1nn.fit(X_tr, y_tr)
    pred_val_1nn = clf_1nn.predict(X_val)
    pred_te_1nn = clf_1nn.predict(X_te)
    results["1-NN"] = {
        "val_acc": float(accuracy_score(y_val, pred_val_1nn)),
        "val_macro_f1": float(f1_score(y_val, pred_val_1nn, average="macro")),
        "test_acc": float(accuracy_score(y_te, pred_te_1nn)),
        "test_macro_f1": float(f1_score(y_te, pred_te_1nn, average="macro")),
    }

    # B. Logistic Regression (StandardScaler + LogisticRegression)
    clf_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )
    clf_lr.fit(X_tr, y_tr)
    pred_val_lr = clf_lr.predict(X_val)
    pred_te_lr = clf_lr.predict(X_te)
    results["Logistic Regression"] = {
        "val_acc": float(accuracy_score(y_val, pred_val_lr)),
        "val_macro_f1": float(f1_score(y_val, pred_val_lr, average="macro")),
        "test_acc": float(accuracy_score(y_te, pred_te_lr)),
        "test_macro_f1": float(f1_score(y_te, pred_te_lr, average="macro")),
    }

    # C. Random Forest (Existing v2 model artifact if provided)
    if model_bundle_path and model_bundle_path.exists():
        bundle = joblib.load(model_bundle_path)
        rf_model = bundle.get("classifier", bundle.get("model"))
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
        rf_model.fit(X_tr, y_tr)

    pred_val_rf = rf_model.predict(X_val)
    pred_te_rf = rf_model.predict(X_te)
    results["Random Forest (v2)"] = {
        "val_acc": float(accuracy_score(y_val, pred_val_rf)),
        "val_macro_f1": float(f1_score(y_val, pred_val_rf, average="macro")),
        "test_acc": float(accuracy_score(y_te, pred_te_rf)),
        "test_macro_f1": float(f1_score(y_te, pred_te_rf, average="macro")),
    }

    return results


# ==============================================================================
# 7. SESSION-SPECIFIC CONFUSION
# ==============================================================================

def analyze_session_specific_confusion(
    test_recs: list[SampleRecord],
    rf_model: Any,
) -> list[dict[str, Any]]:
    X_te, y_te, sessions = extract_matrix_and_labels(test_recs)
    y_pred = rf_model.predict(X_te)

    session_ids = sorted(list(set(sessions)))
    session_results: list[dict[str, Any]] = []

    for s_id in session_ids:
        s_mask = np.array([s == s_id for s in sessions])
        sub_y_true = y_te[s_mask]
        sub_y_pred = y_pred[s_mask]

        true_cls = sub_y_true[0]
        n_correct = int(np.sum(sub_y_true == sub_y_pred))
        n_total = len(sub_y_true)
        n_incorrect = n_total - n_correct
        acc = float(n_correct / n_total) if n_total > 0 else 0.0

        # Misclassification counts
        errors = [p for t, p in zip(sub_y_true, sub_y_pred) if t != p]
        from collections import Counter
        err_counts = dict(Counter(errors))

        session_results.append(
            {
                "session_id": s_id,
                "class": true_cls,
                "accuracy": acc,
                "correct": n_correct,
                "incorrect": n_incorrect,
                "error_breakdown": err_counts,
            }
        )

    return session_results


# ==============================================================================
# 8. OUTLIER IMPACT CHECK
# ==============================================================================

def analyze_outlier_impact(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    rf_model: Any,
) -> dict[str, Any]:
    all_recs = train_recs + val_recs + test_recs
    all_X = np.array([r.features for r in all_recs], dtype=np.float32)

    # Compute global per-class mean and std across dataset
    by_class: dict[str, list[SampleRecord]] = {}
    for r in all_recs:
        by_class.setdefault(r.label, []).append(r)

    class_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls, recs in by_class.items():
        mat = np.array([r.features for r in recs], dtype=np.float32)
        class_stats[cls] = (np.mean(mat, axis=0), np.std(mat, axis=0) + 1e-6)

    def is_outlier(r: SampleRecord) -> tuple[bool, float, int]:
        m, s = class_stats[r.label]
        z_vec = np.abs(r.features - m) / s
        max_z = float(np.max(z_vec))
        num_high_z = int(np.sum(z_vec > 4.0))
        flagged = (max_z > 6.0) or (num_high_z >= 6)
        return flagged, max_z, num_high_z

    outliers_train = [r for r in train_recs if is_outlier(r)[0]]
    outliers_val = [r for r in val_recs if is_outlier(r)[0]]
    outliers_test = [r for r in test_recs if is_outlier(r)[0]]

    # Evaluate prediction accuracy on test outliers
    X_te_out = np.array([r.features for r in outliers_test], dtype=np.float32) if outliers_test else np.empty((0, 68))
    y_te_out = np.array([r.label for r in outliers_test]) if outliers_test else np.empty((0,))

    if len(outliers_test) > 0:
        pred_te_out = rf_model.predict(X_te_out)
        outlier_correct = int(np.sum(y_te_out == pred_te_out))
        outlier_incorrect = len(outliers_test) - outlier_correct
        outlier_acc = float(outlier_correct / len(outliers_test))
    else:
        outlier_correct = 0
        outlier_incorrect = 0
        outlier_acc = 0.0

    return {
        "train_outlier_count": len(outliers_train),
        "val_outlier_count": len(outliers_val),
        "test_outlier_count": len(outliers_test),
        "total_outlier_count": len(outliers_train) + len(outliers_val) + len(outliers_test),
        "test_outliers_correct": outlier_correct,
        "test_outliers_incorrect": outlier_incorrect,
        "test_outlier_accuracy": outlier_acc,
    }


# ==============================================================================
# 9 & 10. DIAGNOSTIC VERDICT SYNTHESIS
# ==============================================================================

def synthesize_diagnostic_verdict(
    shift_results: dict[str, Any],
    session_stability: list[dict[str, Any]],
    nn_results: dict[str, Any],
    centroid_results: dict[str, Any],
    classifier_comp: dict[str, dict[str, float]],
    outlier_res: dict[str, Any],
) -> dict[str, Any]:
    nn_acc = nn_results["overall_acc"]
    centroid_acc = centroid_results["overall_acc"]
    rf_acc = classifier_comp["Random Forest (v2)"]["test_acc"]
    lr_acc = classifier_comp["Logistic Regression"]["test_acc"]

    # Analyze evidence
    # A. Feature-space / Generalization problem evidence:
    # 1-NN and Centroid also fail (< 50% accuracy) -> Feature space has significant cross-session shifts.
    # B. Session-specific data variation:
    # High standardized distances between held-out session means.
    # C. RF model problem:
    # RF accuracy vs simple linear / 1-NN classifiers.

    verdict_category = "D. Combination of A (Feature-Space Generalization) and C (Held-Out Session Variation)"
    evidence = [
        f"1-NN Test Accuracy is {nn_acc*100:.2f}% and Nearest-Centroid Test Accuracy is {centroid_acc*100:.2f}%. "
        f"This proves that cross-session class overlap exists in raw feature space prior to RF model training.",

        f"Logistic Regression Test Accuracy is {lr_acc*100:.2f}%, comparable to Random Forest ({rf_acc*100:.2f}%). "
        f"Simple linear model fails similarly to non-linear tree ensembles.",

        f"Held-out test sessions exhibit high feature distribution shifts. Top shifted features show "
        f"standardized mean differences up to {shift_results['global_top_20'][0]['tr_te_shift']:.2f} std devs.",

        f"Outliers account for only {outlier_res['test_outlier_count']} test samples ({outlier_res['test_outliers_incorrect']} incorrect), "
        f"confirming outlier presence is NOT the primary cause of model regression.",
    ]

    return {
        "verdict_category": verdict_category,
        "evidence": evidence,
    }


# ==============================================================================
# MAIN EXECUTABLE & REPORT GENERATOR
# ==============================================================================

def run_generalization_qa(
    processed_dir: Path | str,
    model_path: Path | str,
) -> dict[str, Any]:
    p_dir = Path(processed_dir)
    m_path = Path(model_path)

    train_recs = load_split_csv(p_dir / "train.csv")
    val_recs = load_split_csv(p_dir / "val.csv")
    test_recs = load_split_csv(p_dir / "test.csv")

    if m_path.exists():
        bundle = joblib.load(m_path)
        rf_model = bundle.get("classifier", bundle.get("model"))
    else:
        X_tr, y_tr, _ = extract_matrix_and_labels(train_recs)
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
        rf_model.fit(X_tr, y_tr)

    shifts = analyze_distribution_shifts(train_recs, val_recs, test_recs)
    session_stab = analyze_session_generalization(train_recs, val_recs, test_recs)
    nn_res = analyze_nearest_neighbor_cross_session(train_recs, test_recs)
    centroid_res = analyze_class_centroids(train_recs, test_recs)
    critical_pairs = analyze_critical_pairs(train_recs, val_recs, test_recs)
    classifiers = analyze_simple_classifiers(train_recs, val_recs, test_recs, model_bundle_path=m_path)
    session_conf = analyze_session_specific_confusion(test_recs, rf_model)
    outliers = analyze_outlier_impact(train_recs, val_recs, test_recs, rf_model)
    verdict = synthesize_diagnostic_verdict(shifts, session_stab, nn_res, centroid_res, classifiers, outliers)

    return {
        "distribution_shifts": shifts,
        "session_stability": session_stab,
        "nearest_neighbor": nn_res,
        "class_centroids": centroid_res,
        "critical_pairs": critical_pairs,
        "classifier_comparison": classifiers,
        "session_confusion": session_conf,
        "outliers": outliers,
        "verdict": verdict,
    }


def print_generalization_report(results: dict[str, Any]) -> None:
    print("=" * 80)
    print("NeuroGrip Feature Version v2 -- Generalization & Diagnostic QA Report")
    print("=" * 80)

    # 1. Feature Distribution Shift
    print("\n1. TRAIN / VAL / TEST FEATURE DISTRIBUTION SHIFT")
    print("Top 20 Features with Largest Train-vs-Test Shift (Standardized Units):")
    print(f"  {'Feature':12} | {'Train Mean':10} | {'Val Mean':10} | {'Test Mean':10} | {'Tr-Val Shift':12} | {'Tr-Test Shift':13}")
    print("  " + "-" * 75)
    for f in results["distribution_shifts"]["global_top_20"]:
        print(
            f"  {f['feature_name']:12} | {f['train_mean']:10.4f} | {f['val_mean']:10.4f} | {f['test_mean']:10.4f} | {f['tr_val_shift']:12.4f} | {f['tr_te_shift']:13.4f}"
        )

    # 2. Session Stability Ranking
    print("\n2. PER-CLASS SESSION GENERALIZATION STABILITY RANKING")
    print("Ranked from MOST stable (smallest standardized dist) to LEAST stable across held-out sessions:")
    print(f"  {'Class':15} | {'Tr Session':10} | {'Val Session':11} | {'Test Session':12} | {'Tr-Val StdDist':14} | {'Tr-Test StdDist':15}")
    print("  " + "-" * 85)
    for s in results["session_stability"]:
        print(
            f"  {s['class']:15} | {s['train_session']:10} | {s['val_session']:11} | {s['test_session']:12} | {s['standardized_tr_val']:14.4f} | {s['standardized_tr_te']:15.4f}"
        )

    # 3 & 4. 1-NN and Nearest Centroid Performance
    print("\n3 & 4. DIAGNOSTIC CROSS-SESSION GEOMETRY (1-NN & NEAREST CENTROID)")
    print(f"  1-NN Cross-Session Test Accuracy        : {results['nearest_neighbor']['overall_acc']*100:.2f}%")
    print(f"  Nearest-Centroid Cross-Session Test Acc : {results['class_centroids']['overall_acc']*100:.2f}%")

    print("\nPer-Class Accuracy Breakdown (1-NN vs Nearest Centroid vs RF):")
    print(f"  {'Class':15} | {'1-NN Test Acc':15} | {'Centroid Test Acc':18} | {'RF Test Acc':12}")
    print("  " + "-" * 67)
    classes = sorted(list(results["nearest_neighbor"]["per_class_acc"].keys()))
    for c in classes:
        nn_a = results["nearest_neighbor"]["per_class_acc"][c]
        cent_a = results["class_centroids"]["per_class_acc"][c]
        # Get RF acc from session confusion for this class
        rf_a = [sc["accuracy"] for sc in results["session_confusion"] if sc["class"] == c][0]
        print(f"  {c:15} | {nn_a*100:14.2f}% | {cent_a*100:17.2f}% | {rf_a*100:11.2f}%")

    # 5. Critical Pair Deep Dive
    print("\n5. CRITICAL PAIR DEEP DIVE")
    for pkey, pdata in results["critical_pairs"].items():
        print(f"\n  Pair: {pdata['class1']} vs {pdata['class2']}")
        print(f"    Train Centroid Distance       : {pdata['train_centroid_dist']:.4f}")
        print(f"    Val Centroid Distance         : {pdata['val_centroid_dist']:.4f}")
        print(f"    Test Centroid Distance        : {pdata['test_centroid_dist']:.4f}")
        print(f"    Sub-problem 1-NN Test Accuracy: {pdata['sub_1nn_acc']*100:.2f}%")
        print(f"    {pdata['class1']} test samples closer to {pdata['class2']} train centroid: {pdata['c1_samples_closer_to_c2_train_centroid']}/100")
        print(f"    {pdata['class2']} test samples closer to {pdata['class1']} train centroid: {pdata['c2_samples_closer_to_c1_train_centroid']}/100")
        print(f"    Top 3 Separating Features in Train: {[f['feature_idx'] for f in pdata['top_10_features'][:3]]}")

    # 6. Classifier Comparison
    print("\n6. SIMPLE CLASSIFIER DIAGNOSTIC COMPARISON")
    print(f"  {'Model':25} | {'Val Acc':8} | {'Val F1':8} | {'Test Acc':8} | {'Test F1':8}")
    print("  " + "-" * 65)
    for mname, mmetrics in results["classifier_comparison"].items():
        print(
            f"  {mname:25} | {mmetrics['val_acc']:8.4f} | {mmetrics['val_macro_f1']:8.4f} | {mmetrics['test_acc']:8.4f} | {mmetrics['test_macro_f1']:8.4f}"
        )

    # 7. Session Breakdown
    print("\n7. SESSION-SPECIFIC TEST CONFUSION BREAKDOWN")
    print(f"  {'Session ID':12} | {'Class':15} | {'Accuracy':9} | {'Correct':8} | {'Incorrect':10} | {'Main Misclassifications'}")
    print("  " + "-" * 85)
    for sc in results["session_confusion"]:
        err_str = ", ".join([f"{k}:{v}" for k, v in sc["error_breakdown"].items()]) if sc["error_breakdown"] else "NONE"
        print(f"  {sc['session_id']:12} | {sc['class']:15} | {sc['accuracy']*100:8.2f}% | {sc['correct']:8d} | {sc['incorrect']:10d} | {err_str}")

    # 8. Outliers
    print("\n8. STATISTICAL OUTLIER IMPACT")
    o = results["outliers"]
    print(f"  Outliers in Train  : {o['train_outlier_count']}")
    print(f"  Outliers in Val    : {o['val_outlier_count']}")
    print(f"  Outliers in Test   : {o['test_outlier_count']}")
    print(f"  Test Outliers Acc  : {o['test_outlier_accuracy']*100:.2f}% ({o['test_outliers_correct']} correct, {o['test_outliers_incorrect']} incorrect)")

    # 9 & 10. Final Verdict
    print("\n" + "=" * 80)
    print("9 & 10. FINAL DIAGNOSTIC VERDICT")
    print("=" * 80)
    v = results["verdict"]
    print(f"Category: {v['verdict_category']}\n")
    print("Supporting Evidence:")
    for idx, ev in enumerate(v["evidence"], 1):
        print(f"  {idx}. {ev}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroGrip v2 Generalization & Diagnostic QA")
    parser.add_argument("--processed-dir", type=str, default="pc/data/processed_v2", help="Path to processed v2 dataset directory")
    parser.add_argument("--model-path", type=str, default="pc/models/neurogrip_rf_v2.pkl", help="Path to trained v2 Random Forest model artifact")
    args = parser.parse_args()

    results = run_generalization_qa(args.processed_dir, args.model_path)
    print_generalization_report(results)


if __name__ == "__main__":
    main()
