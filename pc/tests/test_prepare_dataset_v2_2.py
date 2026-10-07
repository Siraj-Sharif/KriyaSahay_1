"""
tests/test_prepare_dataset_v2_2.py
───────────────────────────────
Unit tests for NeuroGrip v2.2 dataset assembly and preparation logic.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import pytest

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_dataset_v2_2 import (
    create_parser_v2_2,
    combine_raw_datasets_v2_2,
    run_pipeline_v2_2,
    EXPECTED_V2_2_CLASS_COUNTS,
)


def _create_synthetic_raw_csv(
    csv_path: Path,
    label: str,
    session_id: str,
    num_samples: int = 10,
    handedness: str = "RIGHT",
    feature_dim: int = 68,
    file_offset: int = 0,
) -> None:
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(feature_dim)
    ]
    rows = [header]
    for i in range(1, num_samples + 1):
        feats = [str(float(file_offset * 1000000 + i * 1000 + j)) for j in range(feature_dim)]
        rows.append([label, handedness, session_id, str(i)] + feats)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_prepare_dataset_v2_2_parser_defaults():
    parser = create_parser_v2_2()
    args = parser.parse_args([])
    assert args.seed == 42
    assert "raw_v2_1" in str(args.v2_1_raw_dir)
    assert "raw_v2_phase10h_middle" in str(args.middle_raw_dir)
    assert "raw_v2_2" in str(args.output_raw_dir)
    assert "processed_v2_2" in str(args.output_processed_dir)


def test_combine_raw_datasets_v2_2(tmp_path: Path):
    v2_1_dir = tmp_path / "raw_v2_1"
    middle_dir = tmp_path / "raw_v2_phase10h_middle"
    out_raw_dir = tmp_path / "raw_v2_2"

    _create_synthetic_raw_csv(v2_1_dir / "file1.csv", "INDEX", "s1", num_samples=10, file_offset=1)
    _create_synthetic_raw_csv(middle_dir / "file2.csv", "MIDDLE", "s2", num_samples=10, file_offset=2)

    num_files, _ = combine_raw_datasets_v2_2(v2_1_dir, middle_dir, out_raw_dir)
    assert num_files == 2
    assert (out_raw_dir / "file1.csv").exists()
    assert (out_raw_dir / "file2.csv").exists()


def test_run_pipeline_v2_2_synthetic(tmp_path: Path):
    v2_1_dir = tmp_path / "raw_v2_1"
    middle_dir = tmp_path / "raw_v2_phase10h_middle"
    out_raw_dir = tmp_path / "raw_v2_2"
    out_proc_dir = tmp_path / "processed_v2_2"

    # Create synthetic raw files for v2.1 (5,700 samples: MIDDLE has 500)
    v2_1_counts = dict(EXPECTED_V2_2_CLASS_COUNTS)
    v2_1_counts["MIDDLE"] = 500

    sess_counter = 1
    for cls, exp_cnt in v2_1_counts.items():
        num_sessions = exp_cnt // 100
        for _ in range(num_sessions):
            sess_id = f"s_{sess_counter:04d}"
            csv_path = v2_1_dir / f"dataset_v2_{cls.lower()}_{sess_id}.csv"
            _create_synthetic_raw_csv(csv_path, cls, sess_id, num_samples=100, file_offset=sess_counter)
            sess_counter += 1

    # Put 2 extra MIDDLE sessions (200 samples each) into middle_dir (400 samples total)
    _create_synthetic_raw_csv(middle_dir / "m1.csv", "MIDDLE", "m1_sess", num_samples=200, file_offset=sess_counter)
    sess_counter += 1
    _create_synthetic_raw_csv(middle_dir / "m2.csv", "MIDDLE", "m2_sess", num_samples=200, file_offset=sess_counter)

    ret = run_pipeline_v2_2(
        v2_1_raw_dir=v2_1_dir,
        middle_raw_dir=middle_dir,
        output_raw_dir=out_raw_dir,
        output_processed_dir=out_proc_dir,
        seed=42,
    )
    assert ret == 0

    assert (out_proc_dir / "train.csv").exists()
    assert (out_proc_dir / "val.csv").exists()
    assert (out_proc_dir / "test.csv").exists()
    assert (out_proc_dir / "v2_2_summary.json").exists()

    with open(out_proc_dir / "v2_2_summary.json", mode="r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["dataset_version"] == "v2.2"
    assert summary["total_samples"] == 6100
    assert summary["total_sessions"] == summary["total_files"]
    assert summary["class_counts"]["MIDDLE"] == 900
