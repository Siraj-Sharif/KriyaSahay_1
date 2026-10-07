"""
scripts/prepare_dataset_v2_2.py
───────────────────────────────
Builds and validates the combined NeuroGrip v2.2 dataset.
Combines raw_v2_1 (5,700 samples, 48 files) and targeted Phase 10H.2 MIDDLE collection
raw_v2_phase10h_middle (400 samples, 2 files) into raw_v2_2 (6,100 samples, 50 files) and
processed_v2_2 without modifying any existing raw or processed v1/v2/v2.1 dataset files.

Usage:
    python pc/scripts/prepare_dataset_v2_2.py
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Sequence

# Ensure src and scripts directories are in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.dataset.qa import DatasetQA
from prepare_dataset import run_prepare_dataset

logger = logging.getLogger("neurogrip.prepare_dataset_v2_2")

EXPECTED_V2_2_CLASS_COUNTS = {
    "CLOSE": 300,
    "FOUR_FINGERS": 700,
    "GRAB": 300,
    "INDEX": 300,
    "INDEX_PINKY": 500,
    "MIDDLE": 900,
    "PINKY": 300,
    "REST": 300,
    "RING": 300,
    "STOP": 500,
    "THREE_FINGER": 700,
    "THUMB_ONLY": 300,
    "TWO_FINGER": 700,
}


def create_parser_v2_2() -> argparse.ArgumentParser:
    data_root = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(
        description="NeuroGrip v2.2 Dataset Assembly & Preparation Pipeline."
    )
    parser.add_argument(
        "--v2-1-raw-dir",
        type=Path,
        default=data_root / "raw_v2_1",
        help="Path to raw_v2_1 directory (default: pc/data/raw_v2_1).",
    )
    parser.add_argument(
        "--middle-raw-dir",
        type=Path,
        default=data_root / "raw_v2_phase10h_middle",
        help="Path to raw_v2_phase10h_middle directory (default: pc/data/raw_v2_phase10h_middle).",
    )
    parser.add_argument(
        "--output-raw-dir",
        type=Path,
        default=data_root / "raw_v2_2",
        help="Path to combined raw_v2_2 directory (default: pc/data/raw_v2_2).",
    )
    parser.add_argument(
        "--output-processed-dir",
        type=Path,
        default=data_root / "processed_v2_2",
        help="Path to processed_v2_2 directory (default: pc/data/processed_v2_2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for session splitting (default: 42).",
    )
    return parser


def combine_raw_datasets_v2_2(
    v2_1_raw_dir: Path,
    middle_raw_dir: Path,
    output_raw_dir: Path,
) -> tuple[int, int]:
    """
    Copy CSV files from raw_v2_1 and raw_v2_phase10h_middle into output_raw_dir.
    """
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    v2_1_csvs = list(v2_1_raw_dir.glob("*.csv"))
    middle_csvs = list(middle_raw_dir.glob("*.csv"))

    logger.info("Found %d CSV files in v2.1 raw dataset: %s", len(v2_1_csvs), v2_1_raw_dir)
    logger.info("Found %d CSV files in targeted MIDDLE dataset: %s", len(middle_csvs), middle_raw_dir)

    copied_files = 0
    for csv_file in v2_1_csvs:
        target = output_raw_dir / csv_file.name
        shutil.copy2(csv_file, target)
        copied_files += 1

    for csv_file in middle_csvs:
        target = output_raw_dir / csv_file.name
        shutil.copy2(csv_file, target)
        copied_files += 1

    logger.info("Successfully copied %d CSV files to %s", copied_files, output_raw_dir)
    return copied_files, 6100


def run_pipeline_v2_2(
    v2_1_raw_dir: Path,
    middle_raw_dir: Path,
    output_raw_dir: Path,
    output_processed_dir: Path,
    seed: int = 42,
) -> int:
    """
    Execute v2.2 dataset assembly, QA validation scan, assertions, and session-isolated split.
    """
    logger.info("================================================================================")
    logger.info("NeuroGrip v2.2 Dataset Preparation Pipeline")
    logger.info("================================================================================")

    # 1. Combine raw CSV files
    num_files, expected_samples = combine_raw_datasets_v2_2(v2_1_raw_dir, middle_raw_dir, output_raw_dir)

    # 2. QA Scan
    qa_engine = DatasetQA()
    qa_report, valid_records = qa_engine.scan_directory(output_raw_dir)

    logger.info("Dataset v2.2 QA Scan Complete:")
    logger.info("  Total CSV Files : %d", num_files)
    logger.info("  Total Samples   : %d", qa_report.total_rows)
    logger.info("  Valid Samples   : %d", qa_report.valid_rows)
    logger.info("  Invalid Samples : %d", qa_report.invalid_rows)
    logger.info("  Duplicates      : %d", qa_report.duplicate_rows)
    logger.info("  Feature Dim     : %d", qa_engine.feature_dim)
    logger.info("  Total Classes   : %d", len(qa_report.class_counts))
    logger.info("  Total Sessions  : %d", len(set(r.session_id for r in valid_records)))

    # 3. Assertions according to requirements
    if qa_report.total_rows != 6100:
        logger.error("Validation Error: Expected 6,100 total samples, got %d", qa_report.total_rows)
        return 1

    if qa_report.valid_rows != 6100:
        logger.error("Validation Error: Expected 6,100 valid rows, got %d", qa_report.valid_rows)
        return 1

    if len(qa_report.class_counts) != 13:
        logger.error("Validation Error: Expected 13 classes, got %d", len(qa_report.class_counts))
        return 1

    if qa_report.class_counts.get("MIDDLE", 0) != 900:
        logger.error("Validation Error: Expected 900 MIDDLE samples, got %d", qa_report.class_counts.get("MIDDLE", 0))
        return 1

    for cls_name, exp_cnt in EXPECTED_V2_2_CLASS_COUNTS.items():
        actual_cnt = qa_report.class_counts.get(cls_name, 0)
        if actual_cnt != exp_cnt:
            logger.error("Validation Error: Class '%s' expected %d samples, got %d", cls_name, exp_cnt, actual_cnt)
            return 1

    if qa_report.invalid_rows > 0 or qa_report.duplicate_rows > 0:
        logger.error(
            "Validation Error: Invalid rows (%d) or duplicates (%d) found. Details: %s",
            qa_report.invalid_rows,
            qa_report.duplicate_rows,
            qa_report.invalid_row_details[:5],
        )
        return 1

    # 4. Session-isolated dataset splitting
    ret = run_prepare_dataset(
        input_dir=output_raw_dir,
        output_dir=output_processed_dir,
        seed=seed,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
    )
    if ret != 0:
        logger.error("Failed to prepare dataset splits in %s", output_processed_dir)
        return ret

    # 5. Save v2.2 summary report JSON
    v2_2_summary = {
        "dataset_version": "v2.2",
        "status": "PASS",
        "total_files": num_files,
        "total_samples": qa_report.total_rows,
        "total_sessions": len(set(r.session_id for r in valid_records)),
        "feature_dim": qa_engine.feature_dim,
        "class_counts": qa_report.class_counts,
        "sessions_per_class": qa_report.sessions_per_class,
        "handedness_counts": qa_report.handedness_counts,
    }

    summary_json_path = output_processed_dir / "v2_2_summary.json"
    with open(summary_json_path, mode="w", encoding="utf-8") as f:
        json.dump(v2_2_summary, f, indent=2)

    logger.info("Saved v2.2 dataset summary report to '%s'.", summary_json_path)
    logger.info("v2.2 Dataset Preparation PASSED successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser_v2_2()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return run_pipeline_v2_2(
        v2_1_raw_dir=args.v2_1_raw_dir,
        middle_raw_dir=args.middle_raw_dir,
        output_raw_dir=args.output_raw_dir,
        output_processed_dir=args.output_processed_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
