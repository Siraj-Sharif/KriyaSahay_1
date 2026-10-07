"""
tests/test_train_model_v2_2.py
───────────────────────────────
Unit tests for Phase 10I.2 v2.2 model training, candidate selection, and evaluation pipeline.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from neurogrip.config.settings import AppConfig
from neurogrip.recognition.ml_recognizer import MLRecognizer
from train_model_v2_2 import (
    build_candidate_models_v2_2,
    create_v2_2_train_parser,
    run_phase10i_2_training,
    evaluate_unseen_middle_session,
)


def _create_synthetic_split_v2_2_csv(
    csv_path: Path,
    num_samples_per_class: int = 10,
    feature_dim: int = 68,
    session_prefix: str = "sess",
) -> None:
    header = ["label", "handedness", "session_id", "sample_id"] + [
        f"feature_{i}" for i in range(feature_dim)
    ]
    classes = [
        "CLOSE",
        "FOUR_FINGERS",
        "GRAB",
        "INDEX",
        "INDEX_PINKY",
        "MIDDLE",
        "PINKY",
        "REST",
        "RING",
        "STOP",
        "THREE_FINGER",
        "THUMB_ONLY",
        "TWO_FINGER",
    ]

    rows: list[list[str]] = [header]

    for cls in classes:
        session_id = f"{session_prefix}_{cls.lower()}"
        for i in range(num_samples_per_class):
            sample_id = f"s_{i:04d}"
            feat_val = float(classes.index(cls) + 1.0)
            feats = [str(feat_val + (j * 0.01)) for j in range(feature_dim)]
            row = [cls, "RIGHT", session_id, sample_id] + feats
            rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_create_v2_2_train_parser():
    parser = create_v2_2_train_parser()
    args = parser.parse_args([])
    assert args.seed == 42
    assert args.n_jobs == -1
    assert "processed_v2_2" in str(args.processed_dir)


def test_build_candidate_models_v2_2():
    candidates = build_candidate_models_v2_2(seed=42, n_jobs=1)
    assert len(candidates) == 4
    assert set(candidates.keys()) == {"extra_trees", "random_forest", "knn_1", "rbf_svm"}
    assert candidates["extra_trees"]["requires_scaling"] is False
    assert candidates["knn_1"]["requires_scaling"] is True


def test_run_phase10i_2_training_pipeline(tmp_path: Path):
    proc_dir = tmp_path / "processed_v2_2"
    model_dir = tmp_path / "models"

    _create_synthetic_split_v2_2_csv(proc_dir / "train.csv", num_samples_per_class=10, session_prefix="tr")
    _create_synthetic_split_v2_2_csv(proc_dir / "val.csv", num_samples_per_class=5, session_prefix="val")
    _create_synthetic_split_v2_2_csv(proc_dir / "test.csv", num_samples_per_class=5, session_prefix="te")

    ret = run_phase10i_2_training(
        processed_dir=proc_dir,
        model_dir=model_dir,
        seed=42,
        n_jobs=1,
    )
    assert ret == 0

    report_json = model_dir / "neurogrip_v2_2_report.json"
    assert report_json.exists()

    with open(report_json, mode="r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["dataset"] == "v2.2"
    assert rep["feature_dim"] == 68
    assert rep["training_samples"] == 130
    assert rep["validation_samples"] == 65
    assert rep["test_samples"] == 65
    assert "candidates_validation_benchmark" in rep
    assert "v2_1_vs_v2_2_comparison" in rep

    model_pkl = model_dir / rep["saved_model_file"]
    assert model_pkl.exists()

    cfg = AppConfig()
    cfg.recognition.model_path = str(model_pkl)
    recognizer = MLRecognizer(config=cfg)
    assert recognizer.is_ready is True
