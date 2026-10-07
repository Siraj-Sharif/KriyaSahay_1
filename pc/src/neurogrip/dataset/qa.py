"""
neurogrip/dataset/qa.py
────────────────────────
Dataset Quality Assurance (QA) Module.
Validates dataset schema, 67-D feature values, 13 canonical command labels, duplicate vectors,
and summarizes data quality metrics.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.features.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RowRecord:
    """Represents a validated single sample row from a raw CSV file."""
    file_path: Path
    line_number: int
    label: str
    handedness: str
    session_id: str
    sample_id: str
    features: np.ndarray


@dataclass
class RowValidationResult:
    """Validation status of an individual dataset row."""
    is_valid: bool
    reason: str
    record: Optional[RowRecord] = None


@dataclass
class QAReport:
    """Quality Assurance Summary Report structure."""
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    sessions_per_class: dict[str, int] = field(default_factory=dict)
    handedness_counts: dict[str, int] = field(default_factory=dict)
    invalid_row_details: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert QA metrics to JSON-serializable dictionary."""
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "duplicate_rows": self.duplicate_rows,
            "class_counts": self.class_counts,
            "sessions_per_class": self.sessions_per_class,
            "handedness_counts": self.handedness_counts,
            "invalid_row_details": self.invalid_row_details,
            "warnings": self.warnings,
        }


class DatasetQA:
    """
    Dataset Quality Assurance engine.
    Scans raw CSV files, enforces schema and value constraints, and detects exact duplicate features.
    """

    def __init__(self, feature_dim: Optional[int] = None) -> None:
        self._auto_detect = (feature_dim is None)
        self.feature_dim = feature_dim if feature_dim is not None else FeatureExtractor.FEATURE_DIM
        self.EXPECTED_HEADER = [
            "label",
            "handedness",
            "session_id",
            "sample_id",
        ] + [f"feature_{i}" for i in range(self.feature_dim)]

    def validate_row(
        self,
        row: list[str],
        file_path: Path,
        line_number: int,
    ) -> RowValidationResult:
        """
        Validate a raw CSV row.
        """
        # 1. Column count check
        expected_len = 4 + self.feature_dim
        if len(row) != expected_len:
            reason = f"COL_COUNT_MISMATCH: Expected {expected_len} columns, got {len(row)}."
            return RowValidationResult(is_valid=False, reason=reason)

        # 2. Label check (must be non-empty and one of 13 locked NeuroGrip commands)
        raw_label = row[0].strip()
        if not raw_label:
            return RowValidationResult(is_valid=False, reason="EMPTY_LABEL: Label is empty.")

        cmd_enum = NeuroGripCommand.from_string(raw_label)
        if cmd_enum is None:
            reason = f"INVALID_LABEL: '{raw_label}' is not one of the 13 locked NeuroGrip commands."
            return RowValidationResult(is_valid=False, reason=reason)

        canonical_label = cmd_enum.value

        # 3. Handedness check
        handedness = row[1].strip().upper()
        if not handedness:
            handedness = "UNKNOWN"

        # 4. Session ID check
        session_id = row[2].strip()
        if not session_id:
            return RowValidationResult(is_valid=False, reason="EMPTY_SESSION_ID: session_id is empty.")

        # 5. Sample ID check
        sample_id = row[3].strip()
        if not sample_id:
            return RowValidationResult(is_valid=False, reason="EMPTY_SAMPLE_ID: sample_id is empty.")

        # 6. Feature vector check (must be numeric, finite, feature_dim-D)
        raw_features = row[4:]
        try:
            feats = np.array([float(x) for x in raw_features], dtype=np.float32)
        except (ValueError, TypeError) as e:
            return RowValidationResult(is_valid=False, reason=f"NON_NUMERIC_FEATURE: {e}")

        if len(feats) != self.feature_dim:
            reason = f"INVALID_FEATURE_DIM: Expected {self.feature_dim}, got {len(feats)}."
            return RowValidationResult(is_valid=False, reason=reason)

        if not np.all(np.isfinite(feats)):
            return RowValidationResult(is_valid=False, reason="NON_FINITE_FEATURE: Feature contains NaN or Inf.")

        record = RowRecord(
            file_path=file_path,
            line_number=line_number,
            label=canonical_label,
            handedness=handedness,
            session_id=session_id,
            sample_id=sample_id,
            features=feats,
        )
        return RowValidationResult(is_valid=True, reason="VALID", record=record)

    def scan_directory(self, input_dir: Path | str) -> tuple[QAReport, list[RowRecord]]:
        """
        Scan all raw CSV files in input_dir, perform QA checks, detect exact duplicates, and return (report, valid_records).
        """
        input_path = Path(input_dir)
        report = QAReport()
        valid_records: list[RowRecord] = []

        # Zero out initial class counts for all 14 locked commands
        for cmd in NeuroGripCommand:
            report.class_counts[cmd.value] = 0
            report.sessions_per_class[cmd.value] = 0

        if not input_path.exists() or not input_path.is_dir():
            warn_msg = f"Input directory '{input_path}' does not exist or is not a directory."
            logger.warning(warn_msg)
            report.warnings.append(warn_msg)
            return report, valid_records

        csv_files = sorted(list(input_path.glob("*.csv")))
        if not csv_files:
            warn_msg = f"No CSV files found under input directory '{input_path}'."
            logger.warning(warn_msg)
            report.warnings.append(warn_msg)
            return report, valid_records

        # Map to track unique sessions per class
        class_sessions_map: dict[str, set[str]] = {cmd.value: set() for cmd in NeuroGripCommand}

        for file_p in csv_files:
            logger.info("QA Scanning CSV file: %s", file_p.name)
            try:
                with open(file_p, mode="r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)

                    if header is None:
                        warn = f"File '{file_p.name}' is empty."
                        report.warnings.append(warn)
                        continue

                    clean_header = [c.strip() for c in header]
                    if self._auto_detect and len(clean_header) >= 4 and clean_header[:4] == ["label", "handedness", "session_id", "sample_id"]:
                        detected_dim = len(clean_header) - 4
                        if detected_dim in (67, 68):
                            self.feature_dim = detected_dim
                            self.EXPECTED_HEADER = ["label", "handedness", "session_id", "sample_id"] + [f"feature_{i}" for i in range(self.feature_dim)]
                            self._auto_detect = False  # Lock after first valid header auto-detection

                    if clean_header != self.EXPECTED_HEADER:
                        warn = f"File '{file_p.name}' header mismatch. Expected {len(self.EXPECTED_HEADER)} columns."
                        logger.warning(warn)
                        report.warnings.append(warn)

                    for line_idx, row in enumerate(reader, start=2):
                        report.total_rows += 1
                        val_res = self.validate_row(row, file_p, line_idx)

                        if val_res.is_valid and val_res.record is not None:
                            rec = val_res.record
                            valid_records.append(rec)
                            report.valid_rows += 1

                            # Update distributions
                            report.class_counts[rec.label] = report.class_counts.get(rec.label, 0) + 1
                            class_sessions_map[rec.label].add(rec.session_id)
                            report.handedness_counts[rec.handedness] = (
                                report.handedness_counts.get(rec.handedness, 0) + 1
                            )
                        else:
                            report.invalid_rows += 1
                            report.invalid_row_details.append(
                                {
                                    "file": file_p.name,
                                    "line": str(line_idx),
                                    "reason": val_res.reason,
                                }
                            )
            except Exception as e:
                err_msg = f"Error reading file '{file_p.name}': {e}"
                logger.error(err_msg)
                report.warnings.append(err_msg)

        # Update sessions_per_class
        for cmd_str, sess_set in class_sessions_map.items():
            report.sessions_per_class[cmd_str] = len(sess_set)

        # Detect exact duplicate feature rows
        seen_features: set[bytes] = set()
        duplicate_count = 0
        for rec in valid_records:
            feat_bytes = rec.features.tobytes()
            if feat_bytes in seen_features:
                duplicate_count += 1
            else:
                seen_features.add(feat_bytes)

        report.duplicate_rows = duplicate_count
        if duplicate_count > 0:
            dup_warn = f"Detected {duplicate_count} exact duplicate feature rows across raw dataset."
            logger.info(dup_warn)
            report.warnings.append(dup_warn)

        # Check for classes with few sessions
        for cmd_str, n_sess in report.sessions_per_class.items():
            if n_sess < 3:
                s_warn = f"Class '{cmd_str}' has only {n_sess} session(s); 80/10/10 split will have limited test/val representation."
                logger.info(s_warn)
                report.warnings.append(s_warn)

        return report, valid_records
