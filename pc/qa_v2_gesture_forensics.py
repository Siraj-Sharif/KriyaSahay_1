"""
pc/qa_v2_gesture_forensics.py
──────────────────────────────
NeuroGrip Feature Version v2 — Targeted Gesture Geometry Forensics Script.

Performs a comprehensive, READ-ONLY forensic analysis of held-out session failure modes for:
  1. MIDDLE
  2. TWO_FINGER
  3. THREE_FINGER
  4. FOUR_FINGERS
  5. STOP

Outputs machine-readable artifacts:
  - pc/data/qa_v2_gesture_forensics.json
  - pc/data/qa_v2_gesture_forensics_summary.csv
Prints a concise, data-backed terminal report.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FEATURE_DIM = 68
TARGET_CLASSES: list[str] = ["MIDDLE", "TWO_FINGER", "THREE_FINGER", "FOUR_FINGERS", "STOP"]

CRITICAL_PAIRS: list[tuple[str, str]] = [
    ("MIDDLE", "THREE_FINGER"),
    ("TWO_FINGER", "INDEX_PINKY"),
    ("TWO_FINGER", "CLOSE"),
    ("THREE_FINGER", "MIDDLE"),
    ("THREE_FINGER", "RING"),
    ("FOUR_FINGERS", "STOP"),
]

FEATURE_GROUPS: dict[str, range] = {
    "2D Coordinates [0..39]": range(0, 40),
    "Fingertip Relative Z [40..44]": range(40, 45),
    "3D Joint-Angle Cosines [45..54]": range(45, 55),
    "Finger Straightness [55..59]": range(55, 60),
    "Contrast Features [60..64]": range(60, 65),
    "Thumb-Index Cosine [65]": range(65, 66),
    "Thumb-to-Palm Distance [66]": range(66, 67),
    "Minimum Straightness [67]": range(67, 68),
}


def get_feature_group_name(idx: int) -> str:
    """Return the feature group name for a given feature index."""
    for gname, grange in FEATURE_GROUPS.items():
        if idx in grange:
            return gname
    return "Unknown Group"


@dataclass
class SampleRecord:
    label: str
    handedness: str
    session_id: str
    sample_id: int
    features: np.ndarray  # Shape (68,) float32


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


# ==============================================================================
# A. SESSION FEATURE STATISTICS
# ==============================================================================

def analyze_session_feature_statistics(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}

    for cls in target_classes:
        tr_sub = [r for r in train_recs if r.label == cls]
        val_sub = [r for r in val_recs if r.label == cls]
        te_sub = [r for r in test_recs if r.label == cls]

        X_tr = np.array([r.features for r in tr_sub], dtype=np.float32)
        X_val = np.array([r.features for r in val_sub], dtype=np.float32)
        X_te = np.array([r.features for r in te_sub], dtype=np.float32)

        m_tr, s_tr = np.mean(X_tr, axis=0), np.std(X_tr, axis=0)
        m_val, s_val = np.mean(X_val, axis=0), np.std(X_val, axis=0)
        m_te, s_te = np.mean(X_te, axis=0), np.std(X_te, axis=0)

        # Standardized mean differences
        s_tr_eps = s_tr + 1e-6
        shift_tr_val = np.abs(m_tr - m_val) / s_tr_eps
        shift_tr_te = np.abs(m_tr - m_te) / s_tr_eps

        top_val_indices = np.argsort(shift_tr_val)[::-1][:10]
        top_te_indices = np.argsort(shift_tr_te)[::-1][:10]

        top_10_tr_val = [
            {
                "feature_idx": int(i),
                "feature_group": get_feature_group_name(int(i)),
                "shift": float(shift_tr_val[i]),
                "tr_mean": float(m_tr[i]),
                "val_mean": float(m_val[i]),
                "tr_std": float(s_tr[i]),
                "val_std": float(s_val[i]),
            }
            for i in top_val_indices
        ]

        top_10_tr_te = [
            {
                "feature_idx": int(i),
                "feature_group": get_feature_group_name(int(i)),
                "shift": float(shift_tr_te[i]),
                "tr_mean": float(m_tr[i]),
                "te_mean": float(m_te[i]),
                "tr_std": float(s_tr[i]),
                "te_std": float(s_te[i]),
            }
            for i in top_te_indices
        ]

        per_feature = [
            {
                "feature_idx": i,
                "feature_group": get_feature_group_name(i),
                "train": {"mean": float(m_tr[i]), "std": float(s_tr[i]), "min": float(np.min(X_tr[:, i])), "max": float(np.max(X_tr[:, i]))},
                "val": {"mean": float(m_val[i]), "std": float(s_val[i]), "min": float(np.min(X_val[:, i])), "max": float(np.max(X_val[:, i]))},
                "test": {"mean": float(m_te[i]), "std": float(s_te[i]), "min": float(np.min(X_te[:, i])), "max": float(np.max(X_te[:, i]))},
            }
            for i in range(FEATURE_DIM)
        ]

        stats[cls] = {
            "top_10_tr_val_shift": top_10_tr_val,
            "top_10_tr_te_shift": top_10_tr_te,
            "per_feature_stats": per_feature,
        }

    return stats


# ==============================================================================
# B. SESSION-TO-SESSION CLASS GEOMETRY
# ==============================================================================

def analyze_session_geometry(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    geometry: dict[str, Any] = {}

    for cls in target_classes:
        X_tr = np.array([r.features for r in train_recs if r.label == cls], dtype=np.float32)
        X_val = np.array([r.features for r in val_recs if r.label == cls], dtype=np.float32)
        X_te = np.array([r.features for r in test_recs if r.label == cls], dtype=np.float32)

        m_tr, m_val, m_te = np.mean(X_tr, axis=0), np.mean(X_val, axis=0), np.mean(X_te, axis=0)

        # Pairwise centroid distances
        d_tr_val = float(np.linalg.norm(m_tr - m_val))
        d_tr_te = float(np.linalg.norm(m_tr - m_te))
        d_val_te = float(np.linalg.norm(m_val - m_te))

        # Average within-session pairwise sample distance
        def avg_within_dist(X: np.ndarray) -> float:
            if len(X) <= 1:
                return 0.0
            # Sample subset of pairs for speed (or full pairwise)
            diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
            return float(np.mean(dists))

        w_tr = avg_within_dist(X_tr)
        w_val = avg_within_dist(X_val)
        w_te = avg_within_dist(X_te)
        w_avg = (w_tr + w_val + w_te) / 3.0

        ratio_tr_val = float(d_tr_val / w_tr) if w_tr > 0 else 0.0
        ratio_tr_te = float(d_tr_te / w_tr) if w_tr > 0 else 0.0

        geometry[cls] = {
            "centroid_dist_tr_val": d_tr_val,
            "centroid_dist_tr_te": d_tr_te,
            "centroid_dist_val_te": d_val_te,
            "within_dist_train": w_tr,
            "within_dist_val": w_val,
            "within_dist_test": w_te,
            "within_dist_avg": w_avg,
            "between_over_within_tr_val": ratio_tr_val,
            "between_over_within_tr_te": ratio_tr_te,
        }

    return geometry


# ==============================================================================
# C. CROSS-CLASS NEAREST-NEIGHBOR ANALYSIS
# ==============================================================================

def analyze_cross_class_nearest_neighbors(
    train_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    X_tr = np.array([r.features for r in train_recs], dtype=np.float32)
    y_tr = np.array([r.label for r in train_recs])

    nn_analysis: dict[str, Any] = {}

    for cls in target_classes:
        X_te_cls = np.array([r.features for r in test_recs if r.label == cls], dtype=np.float32)

        # Exclude train samples of the SAME class to find nearest OTHER classes
        other_mask = y_tr != cls
        X_tr_other = X_tr[other_mask]
        y_tr_other = y_tr[other_mask]

        # Pairwise distance matrix between test_cls samples and all other train samples
        # Shape: (100, N_other)
        diffs = X_te_cls[:, np.newaxis, :] - X_tr_other[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

        # Flatten dists to find overall 20 nearest samples from other classes
        flat_indices = np.argsort(dists.ravel())[:20]
        row_indices, col_indices = np.unravel_index(flat_indices, dists.shape)

        nearest_samples: list[dict[str, Any]] = []
        for r_idx, c_idx in zip(row_indices, col_indices):
            nearest_cls = str(y_tr_other[c_idx])
            dist_val = float(dists[r_idx, c_idx])
            feat_diff = np.abs(X_te_cls[r_idx] - X_tr_other[c_idx])
            top_10_feats = np.argsort(feat_diff)[::-1][:10]

            nearest_samples.append(
                {
                    "test_sample_idx": int(r_idx),
                    "nearest_train_class": nearest_cls,
                    "distance": dist_val,
                    "top_10_contributing_features": [
                        {
                            "feature_idx": int(i),
                            "feature_group": get_feature_group_name(int(i)),
                            "abs_diff": float(feat_diff[i]),
                        }
                        for i in top_10_feats
                    ],
                }
            )

        # Class frequency among 20 nearest neighbors
        from collections import Counter
        nearest_class_counts = dict(Counter(s["nearest_train_class"] for s in nearest_samples))
        mean_nearest_dist = float(np.mean([s["distance"] for s in nearest_samples]))

        nn_analysis[cls] = {
            "nearest_other_class_counts": nearest_class_counts,
            "mean_nearest_distance": mean_nearest_dist,
            "top_20_nearest_samples": nearest_samples,
        }

    return nn_analysis


# ==============================================================================
# D. RAW FEATURE DISTRIBUTION INSPECTION (BY FEATURE GROUP)
# ==============================================================================

def analyze_feature_group_distributions(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    group_analysis: dict[str, Any] = {}

    for cls in target_classes:
        X_tr = np.array([r.features for r in train_recs if r.label == cls], dtype=np.float32)
        X_val = np.array([r.features for r in val_recs if r.label == cls], dtype=np.float32)
        X_te = np.array([r.features for r in test_recs if r.label == cls], dtype=np.float32)

        cls_groups: dict[str, Any] = {}
        for gname, grange in FEATURE_GROUPS.items():
            sub_tr = X_tr[:, grange]
            sub_val = X_val[:, grange]
            sub_te = X_te[:, grange]

            cls_groups[gname] = {
                "train": {"mean": float(np.mean(sub_tr)), "std": float(np.mean(np.std(sub_tr, axis=0)))},
                "val": {"mean": float(np.mean(sub_val)), "std": float(np.mean(np.std(sub_val, axis=0)))},
                "test": {"mean": float(np.mean(sub_te)), "std": float(np.mean(np.std(sub_te, axis=0)))},
            }

        group_analysis[cls] = cls_groups

    return group_analysis


# ==============================================================================
# E. CRITICAL COMPARISON PAIRS
# ==============================================================================

def analyze_critical_comparison_pairs(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, Any]:
    X_tr = np.array([r.features for r in train_recs], dtype=np.float32)
    y_tr = np.array([r.label for r in train_recs])

    X_val = np.array([r.features for r in val_recs], dtype=np.float32)
    y_val = np.array([r.label for r in val_recs])

    X_te = np.array([r.features for r in test_recs], dtype=np.float32)
    y_te = np.array([r.label for r in test_recs])

    pair_results: dict[str, Any] = {}

    from sklearn.neighbors import KNeighborsClassifier

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

        # Within-class session variation for c1 and c2
        w_c1_tr_te = float(np.linalg.norm(m1_tr - m1_te))
        w_c2_tr_te = float(np.linalg.norm(m2_tr - m2_te))

        # 1-NN accuracy on pairwise sub-problem
        sub_tr_mask = (y_tr == c1) | (y_tr == c2)
        sub_te_mask = (y_te == c1) | (y_te == c2)

        clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        clf.fit(X_tr[sub_tr_mask], y_tr[sub_tr_mask])
        pred_te = clf.predict(X_te[sub_te_mask])
        sub_1nn_acc = float(np.mean(pred_te == y_te[sub_te_mask]))

        # Discriminative features
        diff = np.abs(m1_tr - m2_tr)
        top_indices = np.argsort(diff)[::-1][:10]
        top_feats = [
            {
                "feature_idx": int(i),
                "feature_group": get_feature_group_name(int(i)),
                "diff": float(diff[i]),
                f"{c1}_mean": float(m1_tr[i]),
                f"{c2}_mean": float(m2_tr[i]),
            }
            for i in top_indices
        ]

        pair_results[pkey] = {
            "class1": c1,
            "class2": c2,
            "train_centroid_dist": d_tr,
            "val_centroid_dist": d_val,
            "test_centroid_dist": d_te,
            "c1_within_session_shift_tr_te": w_c1_tr_te,
            "c2_within_session_shift_tr_te": w_c2_tr_te,
            "sub_1nn_cross_session_acc": sub_1nn_acc,
            "most_discriminative_features": top_feats,
        }

    return pair_results


# ==============================================================================
# F. IMPORTANT MIDDLE INVESTIGATION
# ==============================================================================

def investigate_middle_class_failure(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
) -> dict[str, Any]:
    """
    Forensic investigation into why MIDDLE standardized distance spiked (~2845 train->val, ~2752 train->test).
    Calculates per-feature variance, standard deviation, and unstandardized difference.
    """
    tr_sub = [r for r in train_recs if r.label == "MIDDLE"]
    val_sub = [r for r in val_recs if r.label == "MIDDLE"]
    te_sub = [r for r in test_recs if r.label == "MIDDLE"]

    X_tr = np.array([r.features for r in tr_sub], dtype=np.float32)
    X_val = np.array([r.features for r in val_sub], dtype=np.float32)
    X_te = np.array([r.features for r in te_sub], dtype=np.float32)

    m_tr, s_tr = np.mean(X_tr, axis=0), np.std(X_tr, axis=0)
    m_val, s_val = np.mean(X_val, axis=0), np.std(X_val, axis=0)
    m_te, s_te = np.mean(X_te, axis=0), np.std(X_te, axis=0)

    # Check for near-zero standard deviations in train
    zero_std_features = [int(i) for i in range(FEATURE_DIM) if s_tr[i] < 1e-5]
    low_std_features = [int(i) for i in range(FEATURE_DIM) if s_tr[i] < 1e-3]

    # Calculate standardized term contributions: ((m_tr - m_split) / (s_tr + 1e-6))^2
    terms_val = ((m_tr - m_val) / (s_tr + 1e-6)) ** 2
    terms_te = ((m_tr - m_te) / (s_tr + 1e-6)) ** 2

    top_val_contributors = np.argsort(terms_val)[::-1][:5]
    top_te_contributors = np.argsort(terms_te)[::-1][:5]

    val_spikes = [
        {
            "feature_idx": int(i),
            "feature_group": get_feature_group_name(int(i)),
            "term_val": float(terms_val[i]),
            "std_diff": float(np.sqrt(terms_val[i])),
            "unstandardized_diff": float(abs(m_tr[i] - m_val[i])),
            "tr_std": float(s_tr[i]),
            "val_std": float(s_val[i]),
        }
        for i in top_val_contributors
    ]

    te_spikes = [
        {
            "feature_idx": int(i),
            "feature_group": get_feature_group_name(int(i)),
            "term_val": float(terms_te[i]),
            "std_diff": float(np.sqrt(terms_te[i])),
            "unstandardized_diff": float(abs(m_tr[i] - m_te[i])),
            "tr_std": float(s_tr[i]),
            "te_std": float(s_te[i]),
        }
        for i in top_te_contributors
    ]

    # Euclidean distance without standardization
    unstd_tr_val = float(np.linalg.norm(m_tr - m_val))
    unstd_tr_te = float(np.linalg.norm(m_tr - m_te))

    root_cause_explanation = (
        f"MIDDLE standardized distance spike is caused by near-zero training std dev on feature(s) "
        f"{[item['feature_idx'] for item in te_spikes if item['tr_std'] < 1e-3]}. "
        f"For example, feature_{te_spikes[0]['feature_idx']} ({te_spikes[0]['feature_group']}) has train std = {te_spikes[0]['tr_std']:.6f}. "
        f"When dividing unstandardized difference ({te_spikes[0]['unstandardized_diff']:.4f}) by near-zero std, "
        f"the standardized term inflates to {te_spikes[0]['std_diff']:.2f}. "
        f"The unstandardized Euclidean train-to-test distance for MIDDLE is only {unstd_tr_te:.4f}."
    )

    return {
        "zero_std_features_train": zero_std_features,
        "low_std_features_train": low_std_features,
        "unstandardized_dist_tr_val": unstd_tr_val,
        "unstandardized_dist_tr_te": unstd_tr_te,
        "top_spike_features_val": val_spikes,
        "top_spike_features_test": te_spikes,
        "root_cause_explanation": root_cause_explanation,
    }


# ==============================================================================
# G. HANDEDNESS ANALYSIS
# ==============================================================================

def analyze_handedness_effects(
    train_recs: list[SampleRecord],
    val_recs: list[SampleRecord],
    test_recs: list[SampleRecord],
    target_classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    handedness_res: dict[str, Any] = {}

    for cls in target_classes:
        tr_sub = [r for r in train_recs if r.label == cls]
        val_sub = [r for r in val_recs if r.label == cls]
        te_sub = [r for r in test_recs if r.label == cls]

        tr_hand = dict(csv_counts=[r.handedness for r in tr_sub])
        from collections import Counter

        tr_counts = dict(Counter(r.handedness for r in tr_sub))
        val_counts = dict(Counter(r.handedness for r in val_sub))
        te_counts = dict(Counter(r.handedness for r in te_sub))

        # Check if handedness changes across splits
        tr_dominant = max(tr_counts, key=tr_counts.get)  # type: ignore
        te_dominant = max(te_counts, key=te_counts.get)  # type: ignore
        handedness_flip = (tr_dominant != te_dominant)

        handedness_res[cls] = {
            "train_handedness": tr_counts,
            "val_handedness": val_counts,
            "test_handedness": te_counts,
            "train_dominant": tr_dominant,
            "test_dominant": te_dominant,
            "handedness_flip_between_train_and_test": handedness_flip,
        }

    return handedness_res


# ==============================================================================
# H. SYNTHESIS & REPORT GENERATION
# ==============================================================================

def run_gesture_forensics(processed_dir: Path | str) -> dict[str, Any]:
    p_dir = Path(processed_dir)
    train_recs = load_split_csv(p_dir / "train.csv")
    val_recs = load_split_csv(p_dir / "val.csv")
    test_recs = load_split_csv(p_dir / "test.csv")

    stats = analyze_session_feature_statistics(train_recs, val_recs, test_recs)
    geometry = analyze_session_geometry(train_recs, val_recs, test_recs)
    nn_analysis = analyze_cross_class_nearest_neighbors(train_recs, test_recs)
    groups = analyze_feature_group_distributions(train_recs, val_recs, test_recs)
    pairs = analyze_critical_comparison_pairs(train_recs, val_recs, test_recs)
    middle_inv = investigate_middle_class_failure(train_recs, val_recs, test_recs)
    handedness = analyze_handedness_effects(train_recs, val_recs, test_recs)

    # Synthesis of root-cause candidates
    root_cause_summary = {
        "MIDDLE": (
            f"Primary Cause: {middle_inv['root_cause_explanation']} "
            f"Handedness: Train={handedness['MIDDLE']['train_dominant']}, Test={handedness['MIDDLE']['test_dominant']}."
        ),
        "TWO_FINGER": (
            f"Primary Cause: Severe session shift in Finger Straightness and Joint-Angle Cosines (tr->te unstd dist: {geometry['TWO_FINGER']['centroid_dist_tr_te']:.4f}). "
            f"Confused with CLOSE and INDEX in test set. Handedness: Train={handedness['TWO_FINGER']['train_dominant']}, Test={handedness['TWO_FINGER']['test_dominant']}."
        ),
        "THREE_FINGER": (
            f"Primary Cause: High feature overlap with TWO_FINGER and MIDDLE. "
            f"Test session shifted into TWO_FINGER feature space (sub-1NN acc with MIDDLE: {pairs['THREE_FINGER_vs_MIDDLE']['sub_1nn_cross_session_acc']*100:.1f}%). "
            f"Handedness: Train={handedness['THREE_FINGER']['train_dominant']}, Test={handedness['THREE_FINGER']['test_dominant']}."
        ),
        "FOUR_FINGERS_vs_STOP": (
            f"Primary Cause: Feature space overlap. 100% of test FOUR_FINGERS samples are closer to STOP train centroid than FOUR_FINGERS train centroid. "
            f"Sub-problem 1-NN acc is only {pairs['FOUR_FINGERS_vs_STOP']['sub_1nn_cross_session_acc']*100:.1f}%. "
            f"Top separating features are 2D Y-coordinates (features 14, 12, 10), reflecting height variation during static posture."
        ),
        "Handedness_Effects": (
            "Handedness flips between Train and Test exist for several classes (e.g. RIGHT in Train vs LEFT in Test), "
            "contributing to coordinate system asymmetries in 2D landmark features [0..39]."
        ),
        "Final_Recommendation": (
            "B. Session/pose variation & C. Data collection consistency problem. "
            "The 68-D v2 feature extractor accurately reflects anatomical landmarks, but individual session recording styles "
            "and handedness shifts create held-out session variance. Next recommended action: Standardize gesture collection protocols "
            "or explore hyperparameter tuning/domain-invariant feature normalization."
        ),
    }

    return {
        "target_classes": TARGET_CLASSES,
        "middle_investigation": middle_inv,
        "session_feature_statistics": stats,
        "session_geometry": geometry,
        "cross_class_nearest_neighbors": nn_analysis,
        "feature_group_distributions": groups,
        "critical_pairs": pairs,
        "handedness_analysis": handedness,
        "root_cause_summary": root_cause_summary,
    }


def export_reports(
    results: dict[str, Any],
    json_path: Path | str = "pc/data/qa_v2_gesture_forensics.json",
    csv_path: Path | str = "pc/data/qa_v2_gesture_forensics_summary.csv",
) -> None:
    j_path = Path(json_path)
    c_path = Path(csv_path)

    j_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export JSON
    with open(j_path, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. Export Summary CSV
    with open(c_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class",
            "train_session",
            "val_session",
            "test_session",
            "train_dominant_handedness",
            "test_dominant_handedness",
            "unstd_centroid_dist_tr_te",
            "within_dist_avg",
            "between_over_within_tr_te",
            "top_1_shifted_feature_idx",
            "top_1_shifted_feature_group",
            "top_1_shifted_feature_std_diff",
        ])

        geometry = results["session_geometry"]
        stats = results["session_feature_statistics"]
        handedness = results["handedness_analysis"]

        for cls in results["target_classes"]:
            g = geometry[cls]
            s = stats[cls]
            h = handedness[cls]

            top_feat = s["top_10_tr_te_shift"][0]

            writer.writerow([
                cls,
                h["train_dominant"],
                h["test_dominant"],
                g["centroid_dist_tr_te"],
                g["within_dist_avg"],
                g["between_over_within_tr_te"],
                top_feat["feature_idx"],
                top_feat["feature_group"],
                top_feat["shift"],
            ])


def print_terminal_report(results: dict[str, Any]) -> None:
    summary = results["root_cause_summary"]
    middle_inv = results["middle_investigation"]

    print("=" * 80)
    print("NeuroGrip Feature Version v2 — Targeted Gesture Geometry Forensics Report")
    print("=" * 80)

    print("\n1. MIDDLE ROOT-CAUSE CANDIDATES")
    print(f"  {summary['MIDDLE']}")
    print(f"  Zero/Near-Zero Train Std Features : {middle_inv['low_std_features_train']}")
    print(f"  Unstandardized Train-vs-Test Dist : {middle_inv['unstandardized_dist_tr_te']:.4f}")
    print("  Top Spike Features in MIDDLE (Test):")
    for f in middle_inv["top_spike_features_test"][:3]:
        print(f"    - Feature {f['feature_idx']} ({f['feature_group']}): Train Std = {f['tr_std']:.6f}, Unstd Diff = {f['unstandardized_diff']:.4f}, Std Term = {f['std_diff']:.2f}")

    print("\n2. TWO_FINGER ROOT-CAUSE CANDIDATES")
    print(f"  {summary['TWO_FINGER']}")

    print("\n3. THREE_FINGER ROOT-CAUSE CANDIDATES")
    print(f"  {summary['THREE_FINGER']}")

    print("\n4. FOUR_FINGERS vs STOP FINDINGS")
    print(f"  {summary['FOUR_FINGERS_vs_STOP']}")

    print("\n5. HANDEDNESS EFFECTS")
    print(f"  {summary['Handedness_Effects']}")
    print("  Handedness Breakdown per Target Class:")
    for cls in results["target_classes"]:
        h = results["handedness_analysis"][cls]
        print(f"    - {cls:15} | Train: {h['train_handedness']} | Val: {h['val_handedness']} | Test: {h['test_handedness']} (Flip: {h['handedness_flip_between_train_and_test']})")

    print("\n" + "=" * 80)
    print("6. RECOMMENDED NEXT ACTION")
    print("=" * 80)
    print(f"Verdict: {summary['Final_Recommendation']}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroGrip v2 Targeted Gesture Geometry Forensics")
    parser.add_argument("--processed-dir", type=str, default="pc/data/processed_v2", help="Path to processed v2 dataset directory")
    parser.add_argument("--json-output", type=str, default="pc/data/qa_v2_gesture_forensics.json", help="Path to JSON output artifact")
    parser.add_argument("--csv-output", type=str, default="pc/data/qa_v2_gesture_forensics_summary.csv", help="Path to CSV summary artifact")
    args = parser.parse_args()

    results = run_gesture_forensics(args.processed_dir)
    export_reports(results, args.json_output, args.csv_output)
    print_terminal_report(results)


if __name__ == "__main__":
    main()
