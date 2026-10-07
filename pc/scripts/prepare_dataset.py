"""
scripts/prepare_dataset.py
───────────────────────────
Phase 10B: Dataset QA and Session-Based Splitter Script.
Scans raw dataset CSV files under pc/data/raw/, validates data quality, detects duplicates,
groups samples strictly by session_id to guarantee zero session leakage, and exports
train.csv, val.csv, test.csv, and dataset_report.json to pc/data/processed/.

Usage:
    python pc/scripts/prepare_dataset.py
    python pc/scripts/prepare_dataset.py --input-dir pc/data/raw --output-dir pc/data/processed --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

# Ensure src directory is in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.dataset.qa import DatasetQA
from neurogrip.dataset.splitter import SessionSplitter, export_split_csv

logger = logging.getLogger("neurogrip.prepare_dataset")


def build_prepare_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for prepare_dataset.py."""
    data_root = Path(__file__).resolve().parent.parent / "data"
    default_input = data_root / "raw_v2_pilot" if (data_root / "raw_v2_pilot").exists() else data_root / "raw"
    default_output = data_root / "processed_v2" if "v2" in str(default_input) else data_root / "processed"

    parser = argparse.ArgumentParser(
        prog="prepare_dataset",
        description="NeuroGrip Dataset QA & Session-Based Splitter.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing raw dataset CSV files (default: {default_input}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output directory for processed CSV splits and JSON report (default: {default_output}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic session-based splitting (default: 42).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Target proportion for train split (default: 0.80).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Target proportion for val split (default: 0.10).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Target proportion for test split (default: 0.10).",
    )
    return parser


def run_prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> int:
    """
    Execute dataset QA and session-based dataset splitting workflow.
    """
    logger.info("Starting Dataset QA and Session-Based Splitter...")
    logger.info("  Input Directory  : %s", input_dir)
    logger.info("  Output Directory : %s", output_dir)
    logger.info("  Ratios (T/V/T)   : %.2f / %.2f / %.2f", train_ratio, val_ratio, test_ratio)
    logger.info("  Random Seed      : %d", seed)

    # 1. Dataset QA Scan
    qa_engine = DatasetQA()
    qa_report, valid_records = qa_engine.scan_directory(input_dir)

    feat_dim = valid_records[0].features.shape[0] if valid_records else qa_engine.feature_dim
    feat_ver = "v2" if feat_dim == 68 else ("v1" if feat_dim == 67 else f"v_{feat_dim}")

    logger.info(
        "QA Scan Summary: Total: %d, Valid: %d, Invalid: %d, Duplicates: %d, Feature Dim: %d (%s)",
        qa_report.total_rows,
        qa_report.valid_rows,
        qa_report.invalid_rows,
        qa_report.duplicate_rows,
        feat_dim,
        feat_ver,
    )

    # 2. Session-Based Dataset Split (Zero Leakage)
    splitter = SessionSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_res = splitter.split(valid_records)

    # 3. Create Output Directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 4. Export CSV Splits
    train_csv = out_path / "train.csv"
    val_csv = out_path / "val.csv"
    test_csv = out_path / "test.csv"

    export_split_csv(split_res.train_records, train_csv, feature_dim=feat_dim)
    export_split_csv(split_res.val_records, val_csv, feature_dim=feat_dim)
    export_split_csv(split_res.test_records, test_csv, feature_dim=feat_dim)

    # 5. Handedness counts per split
    from collections import Counter
    handedness_per_split = {
        "train": dict(Counter(r.handedness for r in split_res.train_records)),
        "val": dict(Counter(r.handedness for r in split_res.val_records)),
        "test": dict(Counter(r.handedness for r in split_res.test_records)),
    }

    # 6. Compile Final JSON Dataset Report
    all_warnings = list(qa_report.warnings) + list(split_res.warnings)
    unique_warnings = []
    for w in all_warnings:
        if w not in unique_warnings:
            unique_warnings.append(w)

    final_report = {
        "total_samples": qa_report.total_rows,
        "total_rows": qa_report.total_rows,
        "valid_rows": qa_report.valid_rows,
        "invalid_rows": qa_report.invalid_rows,
        "duplicate_rows": qa_report.duplicate_rows,
        "feature_dim": feat_dim,
        "feature_version": feat_ver,
        "class_counts": qa_report.class_counts,
        "sessions_per_class": qa_report.sessions_per_class,
        "handedness_counts": qa_report.handedness_counts,
        "handedness_per_split": handedness_per_split,
        "split_counts": split_res.split_counts,
        "split_sessions": split_res.split_sessions,
        "warnings": unique_warnings,
    }

    report_json_path = out_path / "dataset_report.json"
    with open(report_json_path, mode="w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Exported train split to '%s' (%d rows).", train_csv, len(split_res.train_records))
    logger.info("Exported val split to '%s' (%d rows).", val_csv, len(split_res.val_records))
    logger.info("Exported test split to '%s' (%d rows).", test_csv, len(split_res.test_records))
    logger.info("Exported dataset report to '%s'.", report_json_path)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_prepare_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    return run_prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    sys.exit(main())
