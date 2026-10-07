"""
tests/test_dataset_prepare.py
──────────────────────────────
Unit tests for Phase 10B Dataset QA engine, session-based splitter, and prepare_dataset CLI.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_dataset import build_prepare_parser, run_prepare_dataset

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.dataset.qa import DatasetQA, RowRecord
from neurogrip.dataset.splitter import SessionSplitter, export_split_csv
from neurogrip.features.extractor import FeatureExtractor


def create_sample_csv_row(
    label: str = "INDEX",
    handedness: str = "RIGHT",
    session_id: str = "sess_01",
    sample_id: int = 1,
    feature_val: float = 0.5,
) -> list[str]:
    """Helper to build a valid 71-column CSV row string list."""
    feats = [f"{feature_val:.6f}" for _ in range(FeatureExtractor.FEATURE_DIM)]
    return [label, handedness, session_id, str(sample_id)] + feats


def test_qa_valid_dataset_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "valid_dataset.csv"

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(create_sample_csv_row("INDEX", "RIGHT", "sess_01", 1, 0.1))
        writer.writerow(create_sample_csv_row("STOP", "LEFT", "sess_02", 2, 0.2))

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 2
    assert report.valid_rows == 2
    assert report.invalid_rows == 0
    assert len(valid_records) == 2
    assert report.class_counts["INDEX"] == 1
    assert report.class_counts["STOP"] == 1
    assert report.sessions_per_class["INDEX"] == 1
    assert report.sessions_per_class["STOP"] == 1


def test_qa_missing_feature_column_detected(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "bad_schema.csv"

    # Only 50 feature columns instead of 67
    header = ["label", "handedness", "session_id", "sample_id"] + [f"feature_{i}" for i in range(50)]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = ["INDEX", "RIGHT", "sess_01", "1"] + ["0.5"] * 50
        writer.writerow(row)

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 1
    assert report.valid_rows == 0
    assert report.invalid_rows == 1
    assert any("COL_COUNT_MISMATCH" in item["reason"] for item in report.invalid_row_details)


def test_qa_non_finite_feature_detected(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "nan_feature.csv"

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = create_sample_csv_row("INDEX", "RIGHT", "sess_01", 1, 0.5)
        row[10] = "NaN"  # Non-finite feature
        writer.writerow(row)

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 1
    assert report.valid_rows == 0
    assert report.invalid_rows == 1
    assert any("NON_FINITE_FEATURE" in item["reason"] for item in report.invalid_row_details)


def test_qa_invalid_label_detected(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "bad_label.csv"

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = create_sample_csv_row("INVALID_GESTURE_123", "RIGHT", "sess_01", 1, 0.5)
        writer.writerow(row)

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 1
    assert report.valid_rows == 0
    assert report.invalid_rows == 1
    assert any("INVALID_LABEL" in item["reason"] for item in report.invalid_row_details)


def test_qa_empty_session_id_detected(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "empty_sess.csv"

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = create_sample_csv_row("INDEX", "RIGHT", "", 1, 0.5)
        writer.writerow(row)

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 1
    assert report.valid_rows == 0
    assert report.invalid_rows == 1
    assert any("EMPTY_SESSION_ID" in item["reason"] for item in report.invalid_row_details)


def test_qa_duplicate_rows_detected(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_file = raw_dir / "duplicates.csv"

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    row_data = create_sample_csv_row("INDEX", "RIGHT", "sess_01", 1, 0.777)

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row_data)
        writer.writerow(row_data)  # Exact duplicate feature row

    qa = DatasetQA()
    report, valid_records = qa.scan_directory(raw_dir)

    assert report.total_rows == 2
    assert report.valid_rows == 2
    assert report.duplicate_rows == 1


def test_session_splitter_prevents_session_leakage():
    """CRITICAL TEST: Verify the exact same session_id NEVER appears in more than one split."""
    records = []
    # Create 10 sessions with 50 samples each
    for s_idx in range(10):
        session_id = f"sess_{s_idx:02d}"
        for sample_id in range(50):
            records.append(
                RowRecord(
                    file_path=Path("dummy.csv"),
                    line_number=sample_id,
                    label="INDEX" if s_idx % 2 == 0 else "STOP",
                    handedness="RIGHT",
                    session_id=session_id,
                    sample_id=str(sample_id),
                    features=np.full(67, 0.1 * (s_idx + 1), dtype=np.float32),
                )
            )

    splitter = SessionSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    split_res = splitter.split(records)

    train_sessions = set(split_res.split_sessions["train"])
    val_sessions = set(split_res.split_sessions["val"])
    test_sessions = set(split_res.split_sessions["test"])

    # Strict isolation assertions
    assert train_sessions.isdisjoint(val_sessions), "Session leakage between train and val!"
    assert train_sessions.isdisjoint(test_sessions), "Session leakage between train and test!"
    assert val_sessions.isdisjoint(test_sessions), "Session leakage between val and test!"

    total_sessions = train_sessions | val_sessions | test_sessions
    assert len(total_sessions) == 10


def test_session_splitter_ratios_and_determinism():
    records = []
    for s_idx in range(20):
        session_id = f"sess_{s_idx:02d}"
        for sample_id in range(10):
            records.append(
                RowRecord(
                    file_path=Path("dummy.csv"),
                    line_number=sample_id,
                    label="INDEX",
                    handedness="RIGHT",
                    session_id=session_id,
                    sample_id=str(sample_id),
                    features=np.zeros(67, dtype=np.float32),
                )
            )

    splitter_a = SessionSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    res_a = splitter_a.split(records)

    splitter_b = SessionSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    res_b = splitter_b.split(records)

    # Check seed determinism
    assert res_a.split_sessions == res_b.split_sessions
    assert res_a.split_counts == res_b.split_counts

    # Check that all 20 sessions were assigned without session overlap
    assert len(res_a.split_sessions["train"]) == 18
    assert len(res_a.split_sessions["val"]) == 1
    assert len(res_a.split_sessions["test"]) == 1
    assert len(res_a.split_sessions["train"]) + len(res_a.split_sessions["val"]) + len(res_a.split_sessions["test"]) == 20


def test_session_splitter_multi_class():
    records = []
    classes = ["INDEX", "MIDDLE", "RING", "PINKY", "STOP"]

    for c_idx, cls_name in enumerate(classes):
        for s_idx in range(4):
            session_id = f"sess_{cls_name}_{s_idx}"
            for sample_id in range(5):
                records.append(
                    RowRecord(
                        file_path=Path("dummy.csv"),
                        line_number=sample_id,
                        label=cls_name,
                        handedness="RIGHT",
                        session_id=session_id,
                        sample_id=str(sample_id),
                        features=np.zeros(67, dtype=np.float32),
                    )
                )

    splitter = SessionSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    res = splitter.split(records)

    # Check multi-class representation in train, val, test
    train_labels = {r.label for r in res.train_records}
    val_labels = {r.label for r in res.val_records}
    test_labels = {r.label for r in res.test_records}

    assert len(train_labels) == 5
    assert len(val_labels) == 5
    assert len(test_labels) == 5


def test_prepare_dataset_cli_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()

    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(FeatureExtractor.FEATURE_DIM)
    ]

    with open(raw_dir / "ds_1.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(10):
            writer.writerow(create_sample_csv_row("INDEX", "RIGHT", "sess_01", i, 0.1))
        for i in range(10):
            writer.writerow(create_sample_csv_row("STOP", "RIGHT", "sess_02", i, 0.2))

    ret = run_prepare_dataset(
        input_dir=raw_dir,
        output_dir=proc_dir,
        seed=42,
    )
    assert ret == 0

    assert (proc_dir / "train.csv").exists()
    assert (proc_dir / "val.csv").exists()
    assert (proc_dir / "test.csv").exists()
    assert (proc_dir / "dataset_report.json").exists()

    with open(proc_dir / "dataset_report.json", mode="r", encoding="utf-8") as f:
        report_data = json.load(f)
        assert report_data["total_rows"] == 20
        assert report_data["valid_rows"] == 20
        assert report_data["invalid_rows"] == 0
        assert "split_counts" in report_data
        assert "split_sessions" in report_data


def test_v2_pilot_dataset_preparation_leakage_and_schema(tmp_path):
    """
    Strong leakage and schema verification tests on v2 pilot dataset split:
    1. Train/val session IDs are disjoint.
    2. Train/test session IDs are disjoint.
    3. Val/test session IDs are disjoint.
    4. Every class has exactly one session in each split.
    5. Each split has exactly 1,300 samples.
    6. Each class has exactly 100 samples in each split.
    7. Feature dimension is exactly 68 (72 CSV columns total).
    8. No raw files are modified during preparation.
    """
    raw_v2_dir = Path(__file__).resolve().parent.parent / "data" / "raw_v2_pilot"
    if not raw_v2_dir.exists():
        raw_v2_dir = Path(__file__).resolve().parent.parent.parent / "archive" / "data" / "raw_v2_pilot"
    assert raw_v2_dir.exists(), "raw_v2_pilot directory missing!"

    # Record initial mtimes of raw v2 CSV files
    raw_files = list(raw_v2_dir.glob("dataset_v2_*.csv"))
    assert len(raw_files) == 39
    initial_mtimes = {f: f.stat().st_mtime for f in raw_files}

    proc_v2_dir = tmp_path / "processed_v2"

    ret = run_prepare_dataset(
        input_dir=raw_v2_dir,
        output_dir=proc_v2_dir,
        seed=42,
    )
    assert ret == 0

    # Test 8: Assert no raw files were modified
    for f in raw_files:
        assert f.stat().st_mtime == initial_mtimes[f], f"Raw file {f.name} was modified!"

    train_csv = proc_v2_dir / "train.csv"
    val_csv = proc_v2_dir / "val.csv"
    test_csv = proc_v2_dir / "test.csv"
    report_json = proc_v2_dir / "dataset_report.json"

    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()
    assert report_json.exists()

    with open(report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Test 7: Feature dimension is exactly 68
    assert report["feature_dim"] == 68
    assert report["feature_version"] == "v2"

    train_sess = set(report["split_sessions"]["train"])
    val_sess = set(report["split_sessions"]["val"])
    test_sess = set(report["split_sessions"]["test"])

    # Test 1, 2, 3: Session disjointness (zero leakage)
    assert train_sess.isdisjoint(val_sess), "Train and val session IDs overlap!"
    assert train_sess.isdisjoint(test_sess), "Train and test session IDs overlap!"
    assert val_sess.isdisjoint(test_sess), "Val and test session IDs overlap!"
    assert len(train_sess | val_sess | test_sess) == 39

    # Test 5: Each split has exactly 1,300 samples
    assert report["split_counts"]["train"] == 1300
    assert report["split_counts"]["val"] == 1300
    assert report["split_counts"]["test"] == 1300

    # Parse exported CSV files and check rows and column count
    from collections import Counter

    def check_split_csv(csv_path: Path):
        with open(csv_path, "r", encoding="utf-8") as fp:
            reader = list(csv.reader(fp))
            header = reader[0]
            # 72 columns: 4 metadata + 68 features
            assert len(header) == 72
            rows = reader[1:]
            assert len(rows) == 1300
            classes = Counter(r[0] for r in rows)
            sessions = Counter(r[2] for r in rows)
            return classes, sessions

    train_cls, train_s = check_split_csv(train_csv)
    val_cls, val_s = check_split_csv(val_csv)
    test_cls, test_s = check_split_csv(test_csv)

    # Test 4: Every class has exactly one session in each split
    assert len(train_s) == 13
    assert len(val_s) == 13
    assert len(test_s) == 13

    # Test 6: Each class has exactly 100 samples in each split
    for cmd in NeuroGripCommand:
        assert train_cls[cmd.value] == 100
        assert val_cls[cmd.value] == 100
        assert test_cls[cmd.value] == 100
