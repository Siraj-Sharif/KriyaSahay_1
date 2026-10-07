"""
pc/tests/test_qa_v2_phase10h.py
────────────────────────────────
Unit tests for Phase 10H Targeted Data Collection QA module.
"""

import csv
from pathlib import Path
import pytest
import numpy as np

from qa_v2_phase10h import (
    validate_csv_file,
    run_phase10h_qa,
    PHASE10H_TARGET_CLASSES,
    FEATURE_DIM,
)


def _create_mock_phase10h_csv(csv_path: Path, label: str, session_id: str, num_samples: int = 10, handedness: str = "RIGHT"):
    header = ["label", "handedness", "session_id", "sample_id"] + [f"feature_{i}" for i in range(FEATURE_DIM)]
    rows = [header]
    for i in range(1, num_samples + 1):
        feats = [str(round(0.1 * (i + j), 4)) for j in range(FEATURE_DIM)]
        rows.append([label, handedness, session_id, str(i)] + feats)

    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_validate_csv_file_valid(tmp_path: Path):
    csv_p = tmp_path / "dataset_v2_MIDDLE_abc12345.csv"
    _create_mock_phase10h_csv(csv_p, "MIDDLE", "abc12345", num_samples=200)

    summary, records = validate_csv_file(csv_p)
    assert summary.qa_passed is True
    assert summary.label == "MIDDLE"
    assert summary.session_id == "abc12345"
    assert summary.sample_count == 200
    assert summary.is_feature_finite is True
    assert summary.malformed_row_count == 0
    assert len(records) == 200


def test_validate_csv_file_malformed(tmp_path: Path):
    csv_p = tmp_path / "bad.csv"
    with open(csv_p, mode="w", encoding="utf-8", newline="") as f:
        f.write("label,handedness,session_id,sample_id,feature_0\n")
        f.write("MIDDLE,RIGHT,s1,1,0.5\n")  # Only 5 columns

    summary, records = validate_csv_file(csv_p)
    assert summary.qa_passed is False
    assert summary.malformed_row_count > 0


def test_run_phase10h_qa_pipeline(tmp_path: Path):
    raw_dir = tmp_path / "raw_v2_phase10h"
    raw_dir.mkdir()

    for idx, cls in enumerate(PHASE10H_TARGET_CLASSES):
        csv_p = raw_dir / f"dataset_v2_{cls}_sess00{idx}.csv"
        _create_mock_phase10h_csv(csv_p, cls, f"sess00{idx}", num_samples=200)

    report = run_phase10h_qa(raw_dir)
    assert report["total_files"] == 6
    assert report["total_samples"] == 1200
    assert report["total_unique_sessions"] == 6
    assert report["overall_qa_passed"] is True
    assert all(report["target_classes_fulfilled"].values())
