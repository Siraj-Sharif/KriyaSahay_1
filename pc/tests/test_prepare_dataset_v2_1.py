"""
tests/test_prepare_dataset_v2_1.py
───────────────────────────────────
Unit tests for v2.1 dataset preparation pipeline (prepare_dataset_v2_1.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.prepare_dataset_v2_1 import create_parser, run_pipeline_v2_1


def test_prepare_dataset_v2_1_parser_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert "raw_v2_pilot" in str(args.pilot_dir)
    assert "raw_v2_phase10h" in str(args.phase10h_dir)
    assert "raw_v2_1" in str(args.output_raw_dir)
    assert "processed_v2_1" in str(args.output_processed_dir)
    assert args.seed == 42


def test_prepare_dataset_v2_1_full_pipeline(tmp_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    pilot_dir = project_root / "data" / "raw_v2_pilot"
    if not pilot_dir.exists():
        pilot_dir = project_root.parent / "archive" / "data" / "raw_v2_pilot"
    phase10h_dir = project_root / "data" / "raw_v2_phase10h"
    if not phase10h_dir.exists():
        phase10h_dir = project_root.parent / "archive" / "data" / "raw_v2_phase10h"

    out_raw = tmp_path / "raw_v2_1"
    out_proc = tmp_path / "processed_v2_1"

    ret = run_pipeline_v2_1(
        pilot_dir=pilot_dir,
        phase10h_dir=phase10h_dir,
        output_raw_dir=out_raw,
        output_processed_dir=out_proc,
        seed=42,
    )
    assert ret == 0

    # Verify raw_v2_1 files copied
    copied_csvs = list(out_raw.glob("*.csv"))
    assert len(copied_csvs) == 48

    # Verify processed_v2_1 files generated
    assert (out_proc / "train.csv").exists()
    assert (out_proc / "val.csv").exists()
    assert (out_proc / "test.csv").exists()
    assert (out_proc / "dataset_report.json").exists()
    assert (out_proc / "v2_1_summary.json").exists()

    # Read summary report
    with open(out_proc / "v2_1_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["dataset_version"] == "v2.1"
    assert summary["status"] == "PASS"
    assert summary["total_files"] == 48
    assert summary["total_samples"] == 5700
    assert summary["total_sessions"] == 48
    assert summary["feature_dim"] == 68
    assert len(summary["class_counts"]) == 13
