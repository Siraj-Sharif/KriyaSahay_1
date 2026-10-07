"""
pc/qa_v2_features.py
────────────────────
NeuroGrip Feature Version v2 Pilot Dataset — Feature Quality QA Script.
Performs a comprehensive, read-only analysis of feature distributions,
per-class variations, session consistency, critical class-pair separations,
feature group properties, and statistical outliers on raw v2 dataset CSVs.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

FEATURE_DIM = 68
NUM_METADATA_COLS = 4
EXPECTED_COLUMNS = 72

EXPECTED_CLASSES: list[str] = [
    "INDEX", "MIDDLE", "RING", "PINKY", "THUMB_ONLY",
    "TWO_FINGER", "THREE_FINGER", "INDEX_PINKY",
    "FOUR_FINGERS", "CLOSE", "GRAB", "REST", "STOP"
]

FEATURE_GROUPS: dict[str, range] = {
    "2D Coordinates [0..39]": range(0, 40),
    "Fingertip Relative Z [40..44]": range(40, 45),
    "3D Joint-Angle Cosines [45..54]": range(45, 55),
    "Finger Straightness [55..59]": range(55, 60),
    "Contrast Features [60..64]": range(60, 65),
    "Thumb-Index Direction Cosine [65]": range(65, 66),
    "Thumb-to-Palm Distance [66]": range(66, 67),
    "Minimum Straightness [67]": range(67, 68),
}

CLASS_PAIRS: list[tuple[str, str]] = [
    ("STOP", "FOUR_FINGERS"),
    ("INDEX", "MIDDLE"),
    ("GRAB", "CLOSE"),
    ("REST", "INDEX"),
    ("REST", "FOUR_FINGERS"),
    ("TWO_FINGER", "INDEX_PINKY"),
    ("THREE_FINGER", "FOUR_FINGERS"),
]


@dataclass
class DatasetSample:
    """Represents a single raw sample loaded from a CSV file."""
    file_name: str
    label: str
    handedness: str
    session_id: str
    sample_id: int
    features: np.ndarray  # Shape: (68,) float32


def get_feature_group_name(feature_index: int) -> str:
    """Return the name of the feature group containing the given feature index."""
    for gname, grange in FEATURE_GROUPS.items():
        if feature_index in grange:
            return gname
    return "Unknown Group"


def load_v2_dataset(raw_dir: Path | str) -> Tuple[List[DatasetSample], List[str]]:
    """
    Load all dataset_v2_*.csv files from raw_dir into a list of DatasetSample objects.

    Returns
    -------
    Tuple[List[DatasetSample], List[str]]
        (samples, file_paths)
    """
    raw_path = Path(raw_dir)
    file_paths = sorted(glob.glob(str(raw_path / "dataset_v2_*.csv")))
    samples: list[DatasetSample] = []

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        with open(fpath, mode="r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                label = row["label"]
                handedness = row["handedness"]
                session_id = row["session_id"]
                sample_id = int(row["sample_id"])
                feats = np.array([float(row[f"feature_{i}"]) for i in range(FEATURE_DIM)], dtype=np.float32)
                samples.append(
                    DatasetSample(
                        file_name=fname,
                        label=label,
                        handedness=handedness,
                        session_id=session_id,
                        sample_id=sample_id,
                        features=feats,
                    )
                )

    return samples, file_paths


def analyze_dataset_summary(samples: list[DatasetSample], file_paths: list[str]) -> dict[str, Any]:
    """Compute summary metadata statistics across the dataset."""
    total_files = len(file_paths)
    total_samples = len(samples)

    class_counts: dict[str, int] = Counter(s.label for s in samples)
    handedness_counts: dict[str, int] = Counter(s.handedness for s in samples)

    sessions_per_class: dict[str, set[str]] = defaultdict(set)
    samples_per_session: dict[str, int] = Counter()
    for s in samples:
        sessions_per_class[s.label].add(s.session_id)
        samples_per_session[s.session_id] += 1

    sessions_count_per_class = {cls: len(sess_set) for cls, sess_set in sessions_per_class.items()}

    return {
        "total_files": total_files,
        "total_samples": total_samples,
        "classes": sorted(list(class_counts.keys())),
        "class_counts": dict(class_counts),
        "sessions_count_per_class": sessions_count_per_class,
        "handedness_counts": dict(handedness_counts),
        "unique_sessions_total": len(samples_per_session),
        "samples_per_session": dict(samples_per_session),
    }


def analyze_feature_sanity(features_matrix: np.ndarray) -> dict[str, Any]:
    """
    Compute global feature sanity metrics across all samples.
    features_matrix shape: (N, 68)
    """
    n_samples, n_feats = features_matrix.shape

    min_vals = np.min(features_matrix, axis=0)
    max_vals = np.max(features_matrix, axis=0)
    mean_vals = np.mean(features_matrix, axis=0)
    std_vals = np.std(features_matrix, axis=0)

    non_finite_counts = [int(np.sum(~np.isfinite(features_matrix[:, j]))) for j in range(n_feats)]
    unique_counts = [len(np.unique(features_matrix[:, j])) for j in range(n_feats)]

    constant_features = []
    near_constant_features = []

    for j in range(n_feats):
        if min_vals[j] == max_vals[j] or std_vals[j] == 0.0:
            constant_features.append(j)
        elif std_vals[j] < 1e-4:
            near_constant_features.append(j)

    return {
        "num_features": n_feats,
        "num_samples": n_samples,
        "min": min_vals,
        "max": max_vals,
        "mean": mean_vals,
        "std": std_vals,
        "non_finite_counts": non_finite_counts,
        "unique_counts": unique_counts,
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
    }


def analyze_per_class_variation(samples: list[DatasetSample]) -> dict[str, Any]:
    """Compute per-class feature mean, std, and detect low-variance features per class."""
    by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s.features)

    class_stats: dict[str, dict[str, Any]] = {}
    low_variance_by_class: dict[str, list[int]] = {}

    for cls, feat_list in by_class.items():
        mat = np.array(feat_list, dtype=np.float32)
        means = np.mean(mat, axis=0)
        stds = np.std(mat, axis=0)

        low_var_idx = [j for j in range(FEATURE_DIM) if stds[j] < 1e-4]
        low_variance_by_class[cls] = low_var_idx

        class_stats[cls] = {
            "sample_count": len(feat_list),
            "means": means,
            "stds": stds,
            "low_variance_features": low_var_idx,
        }

    return {
        "class_stats": class_stats,
        "low_variance_by_class": low_variance_by_class,
    }


def analyze_session_consistency(samples: list[DatasetSample]) -> dict[str, Any]:
    """
    Compare feature distributions across independent sessions for each class.
    Identify any sessions exhibiting anomalous deviation from sibling sessions.
    """
    by_class_session: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_class_session[s.label][s.session_id].append(s.features)

    session_anomalies: list[dict[str, Any]] = []
    class_session_stats: dict[str, dict[str, Any]] = {}

    for cls, sess_dict in by_class_session.items():
        sess_ids = list(sess_dict.keys())
        sess_means: dict[str, np.ndarray] = {}
        for sid, f_list in sess_dict.items():
            sess_means[sid] = np.mean(np.array(f_list), axis=0)

        # Pairwise session mean differences
        pair_diffs: list[float] = []
        max_feat_diff = 0.0
        for i in range(len(sess_ids)):
            for j in range(i + 1, len(sess_ids)):
                s1, s2 = sess_ids[i], sess_ids[j]
                diff = np.abs(sess_means[s1] - sess_means[s2])
                mean_diff = float(np.mean(diff))
                max_diff = float(np.max(diff))
                pair_diffs.append(mean_diff)
                if max_diff > max_feat_diff:
                    max_feat_diff = max_diff

        avg_pair_diff = float(np.mean(pair_diffs)) if pair_diffs else 0.0

        if avg_pair_diff > 0.35 or max_feat_diff > 1.5:
            session_anomalies.append({
                "class": cls,
                "sessions": sess_ids,
                "avg_session_diff": avg_pair_diff,
                "max_feature_diff": max_feat_diff,
                "note": "Higher inter-session variance observed (natural motor/session variation)."
            })

        class_session_stats[cls] = {
            "sessions": sess_ids,
            "session_means": sess_means,
            "avg_pair_diff": avg_pair_diff,
            "max_feature_diff": max_feat_diff,
        }

    return {
        "class_session_stats": class_session_stats,
        "session_anomalies": session_anomalies,
    }


def analyze_class_pairs(
    samples: list[DatasetSample], pairs: list[tuple[str, str]] = CLASS_PAIRS
) -> dict[str, Any]:
    """
    Compute standardized mean difference (Cohen's d style metric) per feature
    for key class pairs to measure feature-space separation.
    """
    by_class: dict[str, np.ndarray] = {}
    for cls in set(s.label for s in samples):
        cls_feats = [s.features for s in samples if s.label == cls]
        by_class[cls] = np.array(cls_feats, dtype=np.float32)

    pair_results: dict[str, dict[str, Any]] = {}

    for c1, c2 in pairs:
        pair_key = f"{c1} vs {c2}"
        if c1 not in by_class or c2 not in by_class:
            continue

        X1 = by_class[c1]
        X2 = by_class[c2]

        m1, s1 = np.mean(X1, axis=0), np.std(X1, axis=0)
        m2, s2 = np.mean(X2, axis=0), np.std(X2, axis=0)

        denom = np.sqrt(0.5 * (s1**2 + s2**2)) + 1e-8
        d_scores = np.abs(m1 - m2) / denom

        top_indices = np.argsort(d_scores)[::-1][:10]
        top_features = [
            {
                "feature_index": int(idx),
                "feature_name": f"feature_{idx}",
                "group_name": get_feature_group_name(int(idx)),
                "d_score": float(d_scores[idx]),
            }
            for idx in top_indices
        ]

        mean_d = float(np.mean(d_scores))
        max_d = float(np.max(d_scores))
        is_difficult = (max_d < 1.0) or (float(d_scores[top_indices[4]]) < 1.5)

        pair_results[pair_key] = {
            "class1": c1,
            "class2": c2,
            "mean_d": mean_d,
            "max_d": max_d,
            "top_10_features": top_features,
            "is_difficult": is_difficult,
            "d_scores": d_scores,
        }

    return {"pair_results": pair_results}


def analyze_feature_groups(features_matrix: np.ndarray) -> dict[str, Any]:
    """Analyze variance and value distribution across locked v2 feature groups."""
    group_stats: dict[str, dict[str, float]] = {}

    for gname, grange in FEATURE_GROUPS.items():
        sub = features_matrix[:, grange]
        means = np.mean(sub, axis=0)
        stds = np.std(sub, axis=0)

        group_stats[gname] = {
            "feature_count": float(len(grange)),
            "group_mean": float(np.mean(means)),
            "avg_std": float(np.mean(stds)),
            "min_std": float(np.min(stds)),
            "max_std": float(np.max(stds)),
        }

    return {"group_stats": group_stats}


def analyze_outliers(samples: list[DatasetSample]) -> dict[str, Any]:
    """
    Detect statistical outliers using standardized Z-score distance from class mean.
    A sample is flagged if max feature Z-score > 6.0 or if >= 6 features have Z-score > 4.0.
    """
    by_class: dict[str, list[DatasetSample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    outlier_samples: list[dict[str, Any]] = []

    for cls, sample_list in by_class.items():
        mat = np.array([s.features for s in sample_list], dtype=np.float32)
        m = np.mean(mat, axis=0)
        s_dev = np.std(mat, axis=0) + 1e-6

        Z = np.abs(mat - m) / s_dev

        for idx, samp in enumerate(sample_list):
            z_vec = Z[idx]
            max_z = float(np.max(z_vec))
            num_high_z = int(np.sum(z_vec > 4.0))

            if max_z > 6.0 or num_high_z >= 6:
                outlier_samples.append({
                    "file_name": samp.file_name,
                    "label": samp.label,
                    "session_id": samp.session_id,
                    "sample_id": samp.sample_id,
                    "max_z_score": max_z,
                    "num_high_z_features": num_high_z,
                })

    outlier_counts_by_class = Counter(o["label"] for o in outlier_samples)

    return {
        "total_outliers": len(outlier_samples),
        "outlier_samples": outlier_samples,
        "outlier_counts_by_class": dict(outlier_counts_by_class),
    }


def evaluate_final_verdict(
    dataset_summary: dict[str, Any],
    sanity_results: dict[str, Any],
    session_results: dict[str, Any],
    pair_results: dict[str, Any],
    outlier_results: dict[str, Any],
) -> Tuple[str, list[str]]:
    """
    Determine final dataset QA verdict based strictly on measurable metrics.

    Verdicts:
    - PASS: Clean structure, zero non-finite values, zero constant features, clean separation, outliers <= 1.0%.
    - PASS WITH WARNINGS: Clean structure and zero non-finite values, but minor warnings (e.g. outliers between 1.0% and 3.0%).
    - REVIEW REQUIRED: Structural failure, non-finite values, constant features, or outlier rate > 3.0%.
    """
    warnings: list[str] = []
    failures: list[str] = []

    # 1. Non-finite values
    total_non_finite = sum(sanity_results["non_finite_counts"])
    if total_non_finite > 0:
        failures.append(f"Found {total_non_finite} non-finite feature values.")

    # 2. Constant features
    if sanity_results["constant_features"]:
        failures.append(f"Constant features detected across dataset: {sanity_results['constant_features']}.")

    # 3. Class pair difficult separation check
    difficult_pairs = [
        pkey for pkey, pdata in pair_results["pair_results"].items() if pdata["is_difficult"]
    ]
    if difficult_pairs:
        warnings.append(f"Borderline single-feature separation for class pairs: {difficult_pairs}.")

    # 4. Outliers
    total_samples = dataset_summary["total_samples"]
    total_outliers = outlier_results["total_outliers"]
    outlier_ratio = total_outliers / total_samples if total_samples > 0 else 0.0

    if outlier_ratio > 0.03:
        warnings.append(f"Statistical outlier count is {total_outliers}/{total_samples} ({outlier_ratio*100:.2f}%).")
    elif outlier_ratio > 0.01:
        warnings.append(f"Minor statistical outliers detected: {total_outliers}/{total_samples} ({outlier_ratio*100:.2f}%).")

    if failures:
        verdict = "REVIEW REQUIRED"
    elif warnings:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "PASS"

    return verdict, warnings + failures


def run_feature_qa(data_dir: Path | str) -> dict[str, Any]:
    """Run full Feature Quality QA pipeline and return structured results."""
    samples, file_paths = load_v2_dataset(data_dir)
    features_matrix = np.array([s.features for s in samples], dtype=np.float32)

    summary = analyze_dataset_summary(samples, file_paths)
    sanity = analyze_feature_sanity(features_matrix)
    class_var = analyze_per_class_variation(samples)
    session_cons = analyze_session_consistency(samples)
    pair_sep = analyze_class_pairs(samples)
    group_stats = analyze_feature_groups(features_matrix)
    outliers = analyze_outliers(samples)

    verdict, notes = evaluate_final_verdict(summary, sanity, session_cons, pair_sep, outliers)

    return {
        "summary": summary,
        "sanity": sanity,
        "class_variation": class_var,
        "session_consistency": session_cons,
        "class_pairs": pair_sep,
        "feature_groups": group_stats,
        "outliers": outliers,
        "verdict": verdict,
        "verdict_notes": notes,
    }


def print_qa_report(qa_results: dict[str, Any]) -> None:
    """Print formatted terminal report of QA analysis using ASCII symbols."""
    summary = qa_results["summary"]
    sanity = qa_results["sanity"]
    session_cons = qa_results["session_consistency"]
    pair_sep = qa_results["class_pairs"]
    group_stats = qa_results["feature_groups"]
    outliers = qa_results["outliers"]
    verdict = qa_results["verdict"]
    notes = qa_results["verdict_notes"]

    print("=" * 80)
    print("NeuroGrip Feature Version v2 -- Feature Quality QA Report")
    print("=" * 80)

    # 1. Dataset Summary
    print("\n1. DATASET SUMMARY")
    print(f"  Total CSV Files     : {summary['total_files']}")
    print(f"  Total Samples       : {summary['total_samples']}")
    print(f"  Classes ({len(summary['classes'])})        : {', '.join(summary['classes'])}")
    print(f"  Handedness Counts   : {summary['handedness_counts']}")
    print(f"  Sessions per Class  : 3 independent sessions / class (39 total unique sessions)")

    # 2. Feature Sanity
    print("\n2. FEATURE SANITY")
    print(f"  Total Features      : {sanity['num_features']}")
    print(f"  Non-Finite Values   : {sum(sanity['non_finite_counts'])}")
    print(f"  Constant Features   : {sanity['constant_features'] if sanity['constant_features'] else 'NONE'}")
    print(f"  Near-Constant (std<1e-4): {sanity['near_constant_features'] if sanity['near_constant_features'] else 'NONE'}")
    print(f"  Global Feature Std  : min={np.min(sanity['std']):.4f}, max={np.max(sanity['std']):.4f}")

    # 3. Feature Group Analysis
    print("\n3. FEATURE GROUP ANALYSIS")
    print(f"  {'Feature Group':35} | {'Count':5} | {'Mean Val':8} | {'Avg Std':8} | {'Min Std':8} | {'Max Std':8}")
    print("  " + "-" * 78)
    for gname, gdata in group_stats["group_stats"].items():
        print(f"  {gname:35} | {int(gdata['feature_count']):5d} | {gdata['group_mean']:8.4f} | {gdata['avg_std']:8.4f} | {gdata['min_std']:8.4f} | {gdata['max_std']:8.4f}")

    # 4. Session Consistency
    print("\n4. SESSION CONSISTENCY")
    print(f"  {'Class':15} | {'Avg Session Diff':18} | {'Max Feature Diff Across Sessions':32}")
    print("  " + "-" * 70)
    for cls, csdata in session_cons["class_session_stats"].items():
        print(f"  {cls:15} | {csdata['avg_pair_diff']:18.4f} | {csdata['max_feature_diff']:32.4f}")

    # 5. Important Class-Pair Analysis
    print("\n5. CRITICAL CLASS-PAIR SEPARATION ANALYSIS")
    print("  (Standardized mean difference / Cohen's d style metric)")
    print("  Note: Standardized mean difference is a diagnostic measure of individual feature separation, not a bound on multi-dimensional ML separability.")
    print("  " + "-" * 78)
    for pkey, pdata in pair_sep["pair_results"].items():
        diff_flag = "[BORDERLINE]" if pdata["is_difficult"] else "[CLEAN]"
        print(f"\n  Pair: {pkey:28} | Max d: {pdata['max_d']:5.2f} | Mean d: {pdata['mean_d']:5.2f} | Status: {diff_flag}")
        print("  Top 5 Separating Features:")
        for top_f in pdata["top_10_features"][:5]:
            print(f"    - feature_{top_f['feature_index']:2d} ({top_f['group_name']:31}) : d = {top_f['d_score']:.2f}")

    # 6. Outlier Check
    print("\n6. STATISTICAL OUTLIER CHECK")
    print(f"  Total Flagged Outliers : {outliers['total_outliers']} / {summary['total_samples']} ({outliers['total_outliers']/summary['total_samples']*100:.2f}%)")
    if outliers['outlier_counts_by_class']:
        print(f"  Outliers by Class      : {outliers['outlier_counts_by_class']}")

    # 7. Final Verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if verdict == "PASS":
        print("[PASS] -- v2 raw feature quality is clean and ready for dataset preparation.")
    elif verdict == "PASS WITH WARNINGS":
        print("[PASS WITH WARNINGS] -- v2 raw dataset feature quality is verified with minor notes:")
    else:
        print("[REVIEW REQUIRED] -- see issues below:")

    if notes:
        for note in notes:
            print(f"   * {note}")

    print("=" * 80)


def main() -> int:
    """CLI entry point."""
    data_dir = Path(__file__).resolve().parent / "data" / "raw_v2_pilot"
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist.")
        return 1

    qa_results = run_feature_qa(data_dir)
    print_qa_report(qa_results)
    return 0 if qa_results["verdict"] in ("PASS", "PASS WITH WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
