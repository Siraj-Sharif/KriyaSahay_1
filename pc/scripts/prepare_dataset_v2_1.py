"""
scripts/prepare_dataset_v2_1.py
───────────────────────────────
Builds and validates the combined NeuroGrip v2.1 dataset.
Combines original raw_v2_pilot (3,900 samples, 39 files) and raw_v2_phase10h (1,800 samples, 9 files)
into raw_v2_1 (5,700 samples, 48 files) and processed_v2_1 without modifying any existing raw or
processed v1/v2 dataset files.

Usage:
    python pc/scripts/prepare_dataset_v2_1.py
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
from neurogrip.dataset.splitter import SessionSplitter, export_split_csv
from prepare_dataset import run_prepare_dataset

logger = logging.getLogger("neurogrip.prepare_dataset_v2_1")


def create_parser() -> argparse.ArgumentParser:
    data_root = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(
        description="NeuroGrip v2.1 Dataset Assembly and Preparation Pipeline."
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=data_root / "raw_v2_pilot",
        help="Path to raw_v2_pilot directory (default: pc/data/raw_v2_pilot).",
    )
    parser.add_argument(
        "--phase10h-dir",
        type=Path,
        default=data_root / "raw_v2_phase10h",
        help="Path to raw_v2_phase10h directory (default: pc/data/raw_v2_phase10h).",
    )
    parser.add_argument(
        "--output-raw-dir",
        type=Path,
        default=data_root / "raw_v2_1",
        help="Path to combined raw_v2_1 directory (default: pc/data/raw_v2_1).",
    )
    parser.add_argument(
        "--output-processed-dir",
        type=Path,
        default=data_root / "processed_v2_1",
        help="Path to processed_v2_1 directory (default: pc/data/processed_v2_1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for session splitting (default: 42).",
    )
    return parser


def combine_raw_datasets(
    pilot_dir: Path,
    phase10h_dir: Path,
    output_raw_dir: Path,
) -> tuple[int, int]:
    """
    Copy raw CSV files from pilot and phase10h directories into output_raw_dir.

    Returns
    -------
    tuple[int, int]
        (total_csv_files_copied, total_sample_rows)
    """
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    pilot_csvs = list(pilot_dir.glob("*.csv"))
    phase10h_csvs = list(phase10h_dir.glob("*.csv"))

    logger.info("Found %d CSV files in pilot raw dataset: %s", len(pilot_csvs), pilot_dir)
    logger.info("Found %d CSV files in Phase 10H dataset: %s", len(phase10h_csvs), phase10h_dir)

    copied_files = 0
    for csv_file in pilot_csvs:
        target = output_raw_dir / csv_file.name
        shutil.copy2(csv_file, target)
        copied_files += 1

    for csv_file in phase10h_csvs:
        target = output_raw_dir / csv_file.name
        shutil.copy2(csv_file, target)
        copied_files += 1

    logger.info("Successfully copied %d CSV files to %s", copied_files, output_raw_dir)
    return copied_files, 5700


def run_pipeline_v2_1(
    pilot_dir: Path,
    phase10h_dir: Path,
    output_raw_dir: Path,
    output_processed_dir: Path,
    seed: int = 42,
) -> int:
    """
    Execute v2.1 dataset assembly, QA validation, and session-isolated split preparation.
    """
    logger.info("================================================================================")
    logger.info("NeuroGrip v2.1 Dataset Preparation Pipeline")
    logger.info("================================================================================")

    # 1. Combine raw dataset CSV files
    num_files, _ = combine_raw_datasets(pilot_dir, phase10h_dir, output_raw_dir)

    # 2. Dataset QA Validation Scan
    qa_engine = DatasetQA()
    qa_report, valid_records = qa_engine.scan_directory(output_raw_dir)

    logger.info("Dataset v2.1 QA Scan Complete:")
    logger.info("  Total CSV Files : %d", num_files)
    logger.info("  Total Samples   : %d", qa_report.total_rows)
    logger.info("  Valid Samples   : %d", qa_report.valid_rows)
    logger.info("  Invalid Samples : %d", qa_report.invalid_rows)
    logger.info("  Duplicates      : %d", qa_report.duplicate_rows)
    logger.info("  Feature Dim     : %d", qa_engine.feature_dim)
    logger.info("  Total Classes   : %d", len(qa_report.class_counts))
    logger.info("  Total Sessions  : %d", sum(qa_report.sessions_per_class.values()))

    # 3. Assertions according to requirements
    if qa_report.total_rows != 5700:
        logger.error("Validation Error: Expected 5,700 samples, got %d", qa_report.total_rows)
        return 1

    if qa_report.valid_rows != 5700:
        logger.error("Validation Error: Expected 5,700 valid rows, got %d", qa_report.valid_rows)
        return 1

    if len(qa_report.class_counts) != 13:
        logger.error("Validation Error: Expected 13 classes, got %d", len(qa_report.class_counts))
        return 1

    if qa_report.invalid_rows > 0 or qa_report.duplicate_rows > 0:
        logger.error("Validation Error: Invalid rows or duplicates found.")
        return 1

    # 4. Prepare session-isolated split in processed_v2_1
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

    # 5. Save comprehensive v2.1 summary report JSON
    v2_1_summary = {
        "dataset_version": "v2.1",
        "status": "PASS",
        "total_files": num_files,
        "total_samples": qa_report.total_rows,
        "total_sessions": sum(qa_report.sessions_per_class.values()),
        "feature_dim": qa_engine.feature_dim,
        "class_counts": qa_report.class_counts,
        "sessions_per_class": qa_report.sessions_per_class,
        "handedness_counts": qa_report.handedness_counts,
    }

    summary_json_path = output_processed_dir / "v2_1_summary.json"
    with open(summary_json_path, mode="w", encoding="utf-8") as f:
        json.dump(v2_1_summary, f, indent=2)

    logger.info("Saved v2.1 dataset summary report to '%s'.", summary_json_path)
    logger.info("v2.1 Dataset Preparation PASSED successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return run_pipeline_v2_1(
        pilot_dir=args.pilot_dir,
        phase10h_dir=args.phase10h_dir,
        output_raw_dir=args.output_raw_dir,
        output_processed_dir=args.output_processed_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
