"""
pc/qa_v2_phase10h.py
────────────────────
NeuroGrip Phase 10H — Targeted Generalization Data Collection QA & Verification Script.

Performs quality assurance and structural validation on newly collected Phase 10H raw v2 CSV files.
Target classes (6): FOUR_FINGERS, STOP, TWO_FINGER, THREE_FINGER, MIDDLE, INDEX_PINKY.

Verifies:
  - Total samples collected per class (target: 200 valid samples / session)
  - Handedness distribution (RIGHT / LEFT)
  - Feature finiteness (zero NaN / Inf)
  - Duplicate feature vectors
  - session_id uniqueness across files
  - label consistency with ground-truth CSV header/file name
  - feature dimension (68-D v2)
  - malformed rows / column counts (expected 72 columns)

Outputs:
  - Per-session summary JSON files under output directory
  - Combined QA report JSON
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

FEATURE_DIM = 68
EXPECTED_COLUMNS = 72  # label, handedness, session_id, sample_id + 68 features

PHASE10H_TARGET_CLASSES: list[str] = [
    "FOUR_FINGERS",
    "STOP",
    "TWO_FINGER",
    "THREE_FINGER",
    "MIDDLE",
    "INDEX_PINKY",
]

TARGET_SAMPLES_PER_SESSION = 200


@dataclass
class Phase10HSampleRecord:
    file_name: str
    line_number: int
    label: str
    handedness: str
    session_id: str
    sample_id: int
    features: np.ndarray  # Shape (68,) float32


@dataclass
class SessionQASummary:
    file_name: str
    label: str
    session_id: str
    sample_count: int
    handedness_counts: dict[str, int]
    is_feature_finite: bool
    non_finite_count: int
    duplicate_count: int
    expected_dim: int
    malformed_row_count: int
    sample_id_start: int
    sample_id_end: int
    is_sample_id_sequential: bool
    qa_passed: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_csv_file(file_path: Path) -> tuple[SessionQASummary, list[Phase10HSampleRecord]]:
    """Validate a single Phase 10H raw dataset CSV file."""
    fname = file_path.name
    records: list[Phase10HSampleRecord] = []
    warnings: list[str] = []

    sample_count = 0
    malformed_rows = 0
    non_finite_count = 0
    handedness_counts: dict[str, int] = defaultdict(int)
    seen_feature_bytes: set[bytes] = set()
    duplicate_count = 0

    session_id = ""
    label = ""
    sample_ids: list[int] = []

    with open(file_path, mode="r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)

        if header is None:
            warnings.append(f"File '{fname}' is empty.")
            summary = SessionQASummary(
                file_name=fname,
                label="UNKNOWN",
                session_id="UNKNOWN",
                sample_count=0,
                handedness_counts={},
                is_feature_finite=False,
                non_finite_count=0,
                duplicate_count=0,
                expected_dim=FEATURE_DIM,
                malformed_row_count=1,
                sample_id_start=0,
                sample_id_end=0,
                is_sample_id_sequential=False,
                qa_passed=False,
                warnings=warnings,
            )
            return summary, records

        expected_header = ["label", "handedness", "session_id", "sample_id"] + [f"feature_{i}" for i in range(FEATURE_DIM)]
        clean_header = [c.strip() for c in header]
        if clean_header != expected_header:
            warnings.append(f"File '{fname}' header mismatch. Column count: {len(clean_header)} (expected {EXPECTED_COLUMNS}).")

        for line_idx, row in enumerate(reader, start=2):
            if len(row) != EXPECTED_COLUMNS:
                malformed_rows += 1
                warnings.append(f"Line {line_idx} in '{fname}' has {len(row)} columns (expected {EXPECTED_COLUMNS}).")
                continue

            r_label = row[0].strip()
            r_hand = row[1].strip().upper()
            r_sess = row[2].strip()

            try:
                r_samp_id = int(row[3].strip())
            except ValueError:
                malformed_rows += 1
                warnings.append(f"Line {line_idx} in '{fname}' has non-integer sample_id '{row[3]}'.")
                continue

            try:
                feats = np.array([float(x) for x in row[4:]], dtype=np.float32)
            except (ValueError, TypeError):
                malformed_rows += 1
                warnings.append(f"Line {line_idx} in '{fname}' has non-numeric features.")
                continue

            if len(feats) != FEATURE_DIM:
                malformed_rows += 1
                warnings.append(f"Line {line_idx} in '{fname}' has feature dimension {len(feats)} (expected {FEATURE_DIM}).")
                continue

            if not np.all(np.isfinite(feats)):
                non_finite_count += 1

            feat_bytes = feats.tobytes()
            if feat_bytes in seen_feature_bytes:
                duplicate_count += 1
            else:
                seen_feature_bytes.add(feat_bytes)

            if not session_id:
                session_id = r_sess
            elif r_sess != session_id:
                warnings.append(f"Line {line_idx} in '{fname}' session_id '{r_sess}' differs from initial session_id '{session_id}'.")

            if not label:
                label = r_label
            elif r_label != label:
                warnings.append(f"Line {line_idx} in '{fname}' label '{r_label}' differs from initial label '{label}'.")

            sample_count += 1
            handedness_counts[r_hand] += 1
            sample_ids.append(r_samp_id)

            records.append(
                Phase10HSampleRecord(
                    file_name=fname,
                    line_number=line_idx,
                    label=r_label,
                    handedness=r_hand,
                    session_id=r_sess,
                    sample_id=r_samp_id,
                    features=feats,
                )
            )

    samp_start = sample_ids[0] if sample_ids else 0
    samp_end = sample_ids[-1] if sample_ids else 0
    expected_sequence = list(range(1, sample_count + 1))
    is_sequential = (sample_ids == expected_sequence)

    if not is_sequential and sample_count > 0:
        warnings.append(f"File '{fname}' sample_id sequence is non-standard (range: {samp_start}..{samp_end}).")

    if sample_count < TARGET_SAMPLES_PER_SESSION:
        warnings.append(f"File '{fname}' sample count is {sample_count}/{TARGET_SAMPLES_PER_SESSION}.")

    qa_passed = (
        malformed_rows == 0
        and non_finite_count == 0
        and len(warnings) == 0
    )

    summary = SessionQASummary(
        file_name=fname,
        label=label or "UNKNOWN",
        session_id=session_id or "UNKNOWN",
        sample_count=sample_count,
        handedness_counts=dict(handedness_counts),
        is_feature_finite=(non_finite_count == 0),
        non_finite_count=non_finite_count,
        duplicate_count=duplicate_count,
        expected_dim=FEATURE_DIM,
        malformed_row_count=malformed_rows,
        sample_id_start=samp_start,
        sample_id_end=samp_end,
        is_sample_id_sequential=is_sequential,
        qa_passed=qa_passed,
        warnings=warnings,
    )

    return summary, records


def run_phase10h_qa(data_dir: Path | str) -> dict[str, Any]:
    """
    Run full QA analysis across all raw CSV files in Phase 10H directory.
    Outputs individual session JSON summaries and overall QA report.
    """
    raw_path = Path(data_dir)
    csv_files = sorted(list(raw_path.glob("*.csv")))

    session_summaries: list[SessionQASummary] = []
    all_records: list[Phase10HSampleRecord] = []
    session_ids: list[str] = []
    class_counts: dict[str, int] = defaultdict(int)

    for fpath in csv_files:
        summary, recs = validate_csv_file(fpath)
        session_summaries.append(summary)
        all_records.extend(recs)
        if summary.session_id and summary.session_id != "UNKNOWN":
            session_ids.append(summary.session_id)
        if summary.label and summary.label != "UNKNOWN":
            class_counts[summary.label] += summary.sample_count

        # Write per-session summary JSON next to CSV file or in summary folder
        json_fpath = fpath.with_name(f"session_{summary.session_id}_qa.json") if summary.session_id != "UNKNOWN" else fpath.with_suffix(".json")
        with open(json_fpath, mode="w", encoding="utf-8") as jfp:
            json.dump(summary.to_dict(), jfp, indent=2)

    # Global QA checks across directory
    unique_sessions = set(session_ids)
    duplicate_session_ids = [s for s, c in Counter(session_ids).items() if c > 1]
    total_samples = sum(s.sample_count for s in session_summaries)

    overall_passed = (
        len(csv_files) > 0
        and len(duplicate_session_ids) == 0
        and all(s.qa_passed for s in session_summaries)
    )

    global_report = {
        "raw_dir": str(raw_path),
        "total_files": len(csv_files),
        "total_samples": total_samples,
        "total_unique_sessions": len(unique_sessions),
        "duplicate_session_ids": duplicate_session_ids,
        "class_sample_counts": dict(class_counts),
        "target_classes_fulfilled": {
            cls: (class_counts.get(cls, 0) >= TARGET_SAMPLES_PER_SESSION)
            for cls in PHASE10H_TARGET_CLASSES
        },
        "overall_qa_passed": overall_passed,
        "session_summaries": [s.to_dict() for s in session_summaries],
    }

    report_out_path = raw_path / "qa_v2_phase10h_report.json"
    if raw_path.exists():
        with open(report_out_path, mode="w", encoding="utf-8") as rfp:
            json.dump(global_report, rfp, indent=2)

    return global_report


def print_phase10h_qa_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print("NeuroGrip Phase 10H — Targeted Generalization Data Collection QA Report")
    print("=" * 80)

    print(f"\nTarget Directory        : {report['raw_dir']}")
    print(f"Total CSV Files         : {report['total_files']}")
    print(f"Total Samples Collected : {report['total_samples']}")
    print(f"Unique Session IDs      : {report['total_unique_sessions']}")
    print(f"Duplicate Session IDs   : {report['duplicate_session_ids'] if report['duplicate_session_ids'] else 'NONE'}")
    print(f"Overall QA Status       : {'PASS' if report['overall_qa_passed'] else 'ATTENTION REQUIRED / INCOMPLETE'}")

    print("\nTARGET CLASSES FULFILLMENT STATUS (Target: 200 samples/class):")
    print(f"  {'Class Label':20} | {'Collected':10} | {'Status'}")
    print("  " + "-" * 45)
    for cls in PHASE10H_TARGET_CLASSES:
        cnt = report['class_sample_counts'].get(cls, 0)
        status = "COMPLETE" if cnt >= TARGET_SAMPLES_PER_SESSION else f"PENDING ({cnt}/{TARGET_SAMPLES_PER_SESSION})"
        print(f"  {cls:20} | {cnt:10d} | {status}")

    if report["session_summaries"]:
        print("\nSESSION SUMMARIES:")
        print(f"  {'File Name':35} | {'Class':15} | {'Samples':8} | {'Handedness':15} | {'QA Status'}")
        print("  " + "-" * 85)
        for s in report["session_summaries"]:
            h_str = ", ".join([f"{k}:{v}" for k, v in s['handedness_counts'].items()])
            status = "PASS" if s['qa_passed'] else f"WARNINGS ({len(s['warnings'])})"
            print(f"  {s['file_name']:35} | {s['label']:15} | {s['sample_count']:8d} | {h_str:15} | {status}")
    else:
        print("\nNo CSV files found in target directory. Collection pending user manual execution.")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroGrip Phase 10H Targeted Generalization Collection QA")
    parser.add_argument("--data-dir", type=str, default="pc/data/raw_v2_phase10h", help="Path to Phase 10H raw data directory")
    args = parser.parse_args()

    report = run_phase10h_qa(args.data_dir)
    print_phase10h_qa_report(report)


if __name__ == "__main__":
    main()
