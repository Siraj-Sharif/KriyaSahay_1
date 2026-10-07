"""
pc/tests/test_grouped_cv.py
─────────────────────────────
Unit tests for Phase 10G session-aware GroupKFold cross-validation script.
"""

from pathlib import Path
import pytest
import numpy as np

from scripts.evaluate_grouped_cv import (
    TrainRecord,
    create_candidate_model,
    load_train_csv,
    run_session_aware_cv,
    export_cv_reports,
)


def test_create_candidate_model():
    candidates = [
        "Random Forest — full 68D",
        "Extra Trees — full 68D",
        "RBF SVM — full 68D",
        "Random Forest — geometry 28D [40..67]",
        "RBF SVM — geometry 28D [40..67]",
        "1-NN — full 68D",
    ]

    for cand_name in candidates:
        model = create_candidate_model(cand_name)
        assert model is not None


def test_load_train_csv_schema():
    train_csv = Path("pc/data/processed_v2/train.csv")
    if train_csv.exists():
        X, y, groups, records = load_train_csv(train_csv)
        assert len(X) == 1300
        assert X.shape[1] == 68
        assert len(y) == 1300
        assert len(groups) == 1300
        assert len(records) == 1300
        # Verify 13 unique session_ids (1 per class in train.csv)
        assert len(set(groups)) == 13


def test_run_session_aware_cv_full(tmp_path: Path):
    train_csv = Path("pc/data/processed_v2/train.csv")
    if train_csv.exists():
        json_out = tmp_path / "cv_report.json"
        csv_out = tmp_path / "cv_summary.csv"

        cv_output = run_session_aware_cv(train_csv, n_splits=5)
        assert "ranked_candidates" in cv_output
        assert len(cv_output["ranked_candidates"]) == 6
        assert cv_output["total_train_samples"] == 1300

        export_cv_reports(cv_output, json_out, csv_out)
        assert json_out.exists()
        assert csv_out.exists()
