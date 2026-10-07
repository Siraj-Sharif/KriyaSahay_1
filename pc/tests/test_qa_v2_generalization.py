"""
pc/tests/test_qa_v2_generalization.py
───────────────────────────────────────
Unit tests for the NeuroGrip v2 Generalization & Diagnostic QA script.
"""

from pathlib import Path
import pytest
import numpy as np

from qa_v2_generalization import (
    SampleRecord,
    analyze_distribution_shifts,
    analyze_session_generalization,
    analyze_nearest_neighbor_cross_session,
    analyze_class_centroids,
    analyze_critical_pairs,
    analyze_simple_classifiers,
    run_generalization_qa,
)


def _make_dummy_records(num_samples_per_class: int = 10, prefix: str = "tr") -> list[SampleRecord]:
    classes = ["INDEX", "MIDDLE", "STOP", "FOUR_FINGERS"]
    records = []
    for cls_idx, cls in enumerate(classes):
        for s in range(num_samples_per_class):
            # Create separable dummy feature vectors
            feats = np.zeros(68, dtype=np.float32)
            feats[cls_idx] = 1.0 + s * 0.01 + (0.5 if prefix == "te" else 0.0)
            records.append(
                SampleRecord(
                    label=cls,
                    handedness="RIGHT",
                    session_id=f"{prefix}_sess_{cls_idx}",
                    sample_id=s + 1,
                    features=feats,
                )
            )
    return records


def test_analyze_distribution_shifts():
    tr = _make_dummy_records(10, "tr")
    val = _make_dummy_records(10, "val")
    te = _make_dummy_records(10, "te")

    res = analyze_distribution_shifts(tr, val, te)
    assert "global_top_20" in res
    assert len(res["global_top_20"]) <= 20
    assert "per_class_top_5" in res


def test_analyze_session_generalization():
    tr = _make_dummy_records(10, "tr")
    val = _make_dummy_records(10, "val")
    te = _make_dummy_records(10, "te")

    stab = analyze_session_generalization(tr, val, te)
    assert len(stab) == 4
    for item in stab:
        assert "class" in item
        assert "standardized_tr_te" in item


def test_analyze_nearest_neighbor_and_centroids():
    tr = _make_dummy_records(10, "tr")
    te = _make_dummy_records(10, "te")

    nn_res = analyze_nearest_neighbor_cross_session(tr, te)
    assert 0.0 <= nn_res["overall_acc"] <= 1.0
    assert len(nn_res["per_class_acc"]) == 4

    cent_res = analyze_class_centroids(tr, te)
    assert 0.0 <= cent_res["overall_acc"] <= 1.0
    assert len(cent_res["centroids"]) == 4


def test_run_generalization_qa_full(tmp_path: Path):
    processed_dir = Path("pc/data/processed_v2")
    model_path = Path("pc/models/neurogrip_rf_v2.pkl")

    if processed_dir.exists() and (processed_dir / "train.csv").exists():
        results = run_generalization_qa(processed_dir, model_path)
        assert "distribution_shifts" in results
        assert "session_stability" in results
        assert "nearest_neighbor" in results
        assert "class_centroids" in results
        assert "critical_pairs" in results
        assert "classifier_comparison" in results
        assert "session_confusion" in results
        assert "outliers" in results
        assert "verdict" in results
