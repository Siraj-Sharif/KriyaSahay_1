"""
pc/tests/test_qa_v2_gesture_forensics.py
──────────────────────────────────────────
Unit tests for the NeuroGrip v2 Targeted Gesture Geometry Forensics module.
"""

from pathlib import Path
import pytest
import numpy as np

from qa_v2_gesture_forensics import (
    SampleRecord,
    analyze_session_feature_statistics,
    analyze_session_geometry,
    analyze_cross_class_nearest_neighbors,
    analyze_feature_group_distributions,
    analyze_critical_comparison_pairs,
    investigate_middle_class_failure,
    analyze_handedness_effects,
    run_gesture_forensics,
    export_reports,
)


def _make_dummy_records(num_samples_per_class: int = 10, prefix: str = "tr") -> list[SampleRecord]:
    classes = ["MIDDLE", "TWO_FINGER", "THREE_FINGER", "FOUR_FINGERS", "STOP", "CLOSE", "INDEX_PINKY", "RING"]
    records = []
    for cls_idx, cls in enumerate(classes):
        for s in range(num_samples_per_class):
            feats = np.zeros(68, dtype=np.float32)
            # Give feature 47 a near-zero variation in train for MIDDLE
            if cls == "MIDDLE" and prefix == "tr":
                feats[47] = 0.5 + (0.0001 * s)
            else:
                feats[cls_idx] = 1.0 + s * 0.01 + (0.3 if prefix == "te" else 0.0)

            handedness = "RIGHT" if (s % 2 == 0) else "LEFT"
            records.append(
                SampleRecord(
                    label=cls,
                    handedness=handedness,
                    session_id=f"{prefix}_sess_{cls_idx}",
                    sample_id=s + 1,
                    features=feats,
                )
            )
    return records


def test_analyze_session_feature_statistics():
    tr = _make_dummy_records(10, "tr")
    val = _make_dummy_records(10, "val")
    te = _make_dummy_records(10, "te")

    res = analyze_session_feature_statistics(tr, val, te, target_classes=["MIDDLE", "TWO_FINGER"])
    assert "MIDDLE" in res
    assert "top_10_tr_te_shift" in res["MIDDLE"]
    assert len(res["MIDDLE"]["top_10_tr_te_shift"]) == 10


def test_analyze_session_geometry():
    tr = _make_dummy_records(10, "tr")
    val = _make_dummy_records(10, "val")
    te = _make_dummy_records(10, "te")

    geom = analyze_session_geometry(tr, val, te, target_classes=["MIDDLE"])
    assert "MIDDLE" in geom
    assert "centroid_dist_tr_te" in geom["MIDDLE"]
    assert "between_over_within_tr_te" in geom["MIDDLE"]


def test_investigate_middle_class_failure():
    tr = _make_dummy_records(10, "tr")
    val = _make_dummy_records(10, "val")
    te = _make_dummy_records(10, "te")

    middle_res = investigate_middle_class_failure(tr, val, te)
    assert "zero_std_features_train" in middle_res
    assert "unstandardized_dist_tr_te" in middle_res
    assert "top_spike_features_test" in middle_res


def test_run_gesture_forensics_full(tmp_path: Path):
    processed_dir = Path("pc/data/processed_v2")
    if processed_dir.exists() and (processed_dir / "train.csv").exists():
        json_out = tmp_path / "forensics.json"
        csv_out = tmp_path / "summary.csv"

        results = run_gesture_forensics(processed_dir)
        assert "middle_investigation" in results
        assert "session_geometry" in results
        assert "critical_pairs" in results
        assert "root_cause_summary" in results

        export_reports(results, json_out, csv_out)
        assert json_out.exists()
        assert csv_out.exists()
