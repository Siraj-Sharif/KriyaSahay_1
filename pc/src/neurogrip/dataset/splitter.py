"""
neurogrip/dataset/splitter.py
──────────────────────────────
Session-Based Dataset Splitter Module.
Splits dataset into train/val/test sets grouped strictly by session_id to guarantee zero session leakage.
"""
from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from neurogrip.dataset.qa import RowRecord

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Structure holding assigned records and split metadata."""
    train_records: list[RowRecord] = field(default_factory=list)
    val_records: list[RowRecord] = field(default_factory=list)
    test_records: list[RowRecord] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    split_sessions: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict() -> dict:
        return {
            "split_counts": self.split_counts,
            "split_sessions": self.split_sessions,
            "warnings": self.warnings,
        }


class SessionSplitter:
    """
    Session-based dataset splitter.
    Guarantees zero session leakage across train, val, and test splits.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, valid_records: Sequence[RowRecord]) -> SplitResult:
        """
        Group records by session_id and assign sessions deterministically to train/val/test splits.
        """
        rng = random.Random(self.seed)
        result = SplitResult()

        if not valid_records:
            warn = "No valid records provided for dataset splitting."
            logger.warning(warn)
            result.warnings.append(warn)
            result.split_counts = {"train": 0, "val": 0, "test": 0}
            result.split_sessions = {"train": [], "val": [], "test": []}
            return result

        # 1. Group records by session_id
        session_map: dict[str, list[RowRecord]] = {}
        for rec in valid_records:
            session_map.setdefault(rec.session_id, []).append(rec)

        unique_sessions = sorted(list(session_map.keys()))

        # 2. Map class labels to sessions containing them
        class_to_sessions: dict[str, list[str]] = {}
        for sess_id, recs in session_map.items():
            classes_in_sess = {r.label for r in recs}
            for cls in classes_in_sess:
                class_to_sessions.setdefault(cls, []).append(sess_id)

        # 3. Class-aware Session Assignment
        assigned_sessions: dict[str, str] = {}  # session_id -> "train" | "val" | "test"

        # Sort classes alphabetically for determinism
        sorted_classes = sorted(list(class_to_sessions.keys()))

        for cls in sorted_classes:
            sess_list = sorted(class_to_sessions[cls])
            n_sess = len(sess_list)

            # Check existing assignments for this class
            val_has = any(assigned_sessions.get(s) == "val" for s in sess_list)
            test_has = any(assigned_sessions.get(s) == "test" for s in sess_list)

            # Find unassigned sessions for this class
            unassigned = [s for s in sess_list if s not in assigned_sessions]
            rng.shuffle(unassigned)

            if n_sess < 3:
                warn = f"Class '{cls}' has only {n_sess} session(s). Session isolation preserved; train/val/test may lack full coverage for '{cls}'."
                result.warnings.append(warn)

            # Assign to val if needed and possible
            if not val_has and unassigned and n_sess >= 2:
                s_val = unassigned.pop(0)
                assigned_sessions[s_val] = "val"

            # Assign to test if needed and possible
            if not test_has and unassigned and n_sess >= 3:
                s_test = unassigned.pop(0)
                assigned_sessions[s_test] = "test"

            # Remaining unassigned for this class go to train by default (or ratio)
            for s_rem in unassigned:
                assigned_sessions[s_rem] = "train"

        # 4. Assign any leftover sessions not covered above
        unassigned_leftovers = [s for s in unique_sessions if s not in assigned_sessions]
        if unassigned_leftovers:
            rng.shuffle(unassigned_leftovers)
            n_left = len(unassigned_leftovers)
            n_val = max(1, int(round(n_left * self.val_ratio))) if self.val_ratio > 0 else 0
            n_test = max(1, int(round(n_left * self.test_ratio))) if self.test_ratio > 0 else 0

            val_left = unassigned_leftovers[:n_val]
            test_left = unassigned_leftovers[n_val : n_val + n_test]
            train_left = unassigned_leftovers[n_val + n_test :]

            for s in val_left:
                assigned_sessions[s] = "val"
            for s in test_left:
                assigned_sessions[s] = "test"
            for s in train_left:
                assigned_sessions[s] = "train"

        # Group session IDs by split
        split_sessions = {"train": [], "val": [], "test": []}
        for s_id in unique_sessions:
            sp = assigned_sessions.get(s_id, "train")
            split_sessions[sp].append(s_id)

        # 5. STRICT SANITY CHECK (Session Leakage Prevention)
        set_train = set(split_sessions["train"])
        set_val = set(split_sessions["val"])
        set_test = set(split_sessions["test"])

        assert set_train.isdisjoint(set_val), "SESSION LEAKAGE: Session overlap between train and val!"
        assert set_train.isdisjoint(set_test), "SESSION LEAKAGE: Session overlap between train and test!"
        assert set_val.isdisjoint(set_test), "SESSION LEAKAGE: Session overlap between val and test!"

        # 6. Populate record lists
        train_recs = [r for s in split_sessions["train"] for r in session_map[s]]
        val_recs = [r for s in split_sessions["val"] for r in session_map[s]]
        test_recs = [r for s in split_sessions["test"] for r in session_map[s]]

        result.train_records = train_recs
        result.val_records = val_recs
        result.test_records = test_recs

        result.split_counts = {
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
        }
        result.split_sessions = split_sessions

        logger.info(
            "Session split complete. Train: %d recs (%d sessions), Val: %d recs (%d sessions), Test: %d recs (%d sessions).",
            len(train_recs),
            len(split_sessions["train"]),
            len(val_recs),
            len(split_sessions["val"]),
            len(test_recs),
            len(split_sessions["test"]),
        )
        return result


def export_split_csv(records: Sequence[RowRecord], output_path: Path, feature_dim: int | None = None) -> None:
    """Export a list of RowRecords to a clean CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        dim = feature_dim if feature_dim is not None else 68
    else:
        dim = feature_dim if feature_dim is not None else len(records[0].features)

    header = [
        "label",
        "handedness",
        "session_id",
        "sample_id",
    ] + [f"feature_{i}" for i in range(dim)]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in records:
            row = [rec.label, rec.handedness, rec.session_id, rec.sample_id] + [
                f"{float(val):.6f}" for val in rec.features
            ]
            writer.writerow(row)
