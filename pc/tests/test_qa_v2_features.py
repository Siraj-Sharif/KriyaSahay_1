"""
pc/tests/test_qa_v2_features.py
────────────────────────────────
Unit tests for Feature Quality QA script (qa_v2_features.py).
Tests dataset loading, feature sanity, class pair separation,
outlier detection, and verdict evaluation without requiring pandas.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from qa_v2_features import (
    DatasetSample,
    FEATURE_DIM,
    analyze_class_pairs,
    analyze_dataset_summary,
    analyze_feature_groups,
    analyze_feature_sanity,
    analyze_outliers,
    analyze_per_class_variation,
    analyze_session_consistency,
    evaluate_final_verdict,
    get_feature_group_name,
    run_feature_qa,
)


def test_get_feature_group_name():
    assert get_feature_group_name(0) == "2D Coordinates [0..39]"
    assert get_feature_group_name(39) == "2D Coordinates [0..39]"
    assert get_feature_group_name(40) == "Fingertip Relative Z [40..44]"
    assert get_feature_group_name(45) == "3D Joint-Angle Cosines [45..54]"
    assert get_feature_group_name(55) == "Finger Straightness [55..59]"
    assert get_feature_group_name(60) == "Contrast Features [60..64]"
    assert get_feature_group_name(65) == "Thumb-Index Direction Cosine [65]"
    assert get_feature_group_name(66) == "Thumb-to-Palm Distance [66]"
    assert get_feature_group_name(67) == "Minimum Straightness [67]"


def test_analyze_feature_sanity_clean():
    mat = np.random.randn(100, FEATURE_DIM).astype(np.float32)
    # Ensure no constant features
    mat[:, 0] += np.linspace(-1, 1, 100)

    res = analyze_feature_sanity(mat)
    assert res["num_features"] == 68
    assert res["num_samples"] == 100
    assert sum(res["non_finite_counts"]) == 0
    assert len(res["constant_features"]) == 0


def test_analyze_feature_sanity_constant_and_non_finite():
    mat = np.random.randn(50, FEATURE_DIM).astype(np.float32)
    mat[:, 5] = 42.0  # Constant feature
    mat[10, 2] = np.nan  # Non-finite value

    res = analyze_feature_sanity(mat)
    assert 5 in res["constant_features"]
    assert res["non_finite_counts"][2] == 1


def test_analyze_class_pairs():
    # Construct distinct synthetic feature arrays for 2 classes
    np.random.seed(42)
    s1 = []
    s2 = []
    for _ in range(50):
        f1 = np.zeros(FEATURE_DIM, dtype=np.float32)
        f2 = np.ones(FEATURE_DIM, dtype=np.float32) * 5.0
        # Give feature 0 a huge separation
        f1[0] = 0.0
        f2[0] = 10.0
        s1.append(DatasetSample("f1.csv", "INDEX", "RIGHT", "sess1", 1, f1))
        s2.append(DatasetSample("f2.csv", "MIDDLE", "RIGHT", "sess2", 1, f2))

    res = analyze_class_pairs(s1 + s2, [("INDEX", "MIDDLE")])
    pair_data = res["pair_results"]["INDEX vs MIDDLE"]

    assert pair_data["max_d"] > 5.0
    assert pair_data["top_10_features"][0]["feature_index"] == 0


def test_analyze_outliers():
    np.random.seed(42)
    samples = []
    # 50 normal samples
    for i in range(50):
        feats = np.random.randn(FEATURE_DIM).astype(np.float32) * 0.1
        samples.append(DatasetSample("norm.csv", "INDEX", "RIGHT", "sess1", i, feats))

    # 1 outlier sample with huge Z-scores across features
    outlier_feats = np.ones(FEATURE_DIM, dtype=np.float32) * 10.0
    samples.append(DatasetSample("outlier.csv", "INDEX", "RIGHT", "sess1", 99, outlier_feats))

    res = analyze_outliers(samples)
    assert res["total_outliers"] >= 1
    assert res["outlier_samples"][0]["sample_id"] == 99


def test_evaluate_final_verdict():
    summary = {"total_samples": 3900}
    sanity = {"non_finite_counts": [0] * 68, "constant_features": []}
    session = {}
    pair_clean = {"pair_results": {"A vs B": {"is_difficult": False}}}
    pair_diff = {"pair_results": {"A vs B": {"is_difficult": True}}}

    outliers_clean = {"total_outliers": 10}  # < 1%
    outliers_warn = {"total_outliers": 80}   # ~2%
    outliers_fail = {"total_outliers": 200}  # > 3%

    v1, _ = evaluate_final_verdict(summary, sanity, session, pair_clean, outliers_clean)
    assert v1 == "PASS"

    v2, _ = evaluate_final_verdict(summary, sanity, session, pair_diff, outliers_warn)
    assert v2 == "PASS WITH WARNINGS"

    v3, _ = evaluate_final_verdict(summary, sanity, session, pair_clean, outliers_fail)
    assert v3 == "PASS WITH WARNINGS"

    sanity_bad = {"non_finite_counts": [1] + [0]*67, "constant_features": []}
    v4, _ = evaluate_final_verdict(summary, sanity_bad, session, pair_clean, outliers_clean)
    assert v4 == "REVIEW REQUIRED"


def test_run_feature_qa_on_raw_v2_pilot_dataset():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw_v2_pilot"
    if not data_dir.exists():
        data_dir = Path(__file__).resolve().parent.parent.parent / "archive" / "data" / "raw_v2_pilot"
    assert data_dir.exists()

    res = run_feature_qa(data_dir)
    assert res["summary"]["total_samples"] == 3900
    assert res["summary"]["total_files"] == 39
    assert res["sanity"]["num_features"] == 68
    assert res["verdict"] in ("PASS", "PASS WITH WARNINGS")
