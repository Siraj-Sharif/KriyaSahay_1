"""
pc/tests/test_benchmark_recognition.py
────────────────────────────────────────
Unit tests for the NeuroGrip Recognition Optimization Benchmark script.
"""

from pathlib import Path
import pytest
import numpy as np

from scripts.benchmark_recognition import (
    BenchmarkSample,
    reconstruct_hand_landmarks,
    evaluate_model_performance,
    run_benchmark,
    export_benchmark_reports,
)


def test_reconstruct_hand_landmarks():
    row = {"handedness": "RIGHT"}
    feats = np.zeros(68, dtype=np.float32)
    lms = reconstruct_hand_landmarks(row, feats)
    assert len(lms.landmarks) == 21
    assert lms.landmarks[0].x == 0.0
    assert lms.landmarks[0].y == 0.0


def test_evaluate_model_performance():
    y_true = np.array(["INDEX", "INDEX", "MIDDLE", "STOP"])
    y_pred = np.array(["INDEX", "MIDDLE", "MIDDLE", "STOP"])
    metrics = evaluate_model_performance(y_true, y_pred, classes=["INDEX", "MIDDLE", "STOP"])

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "INDEX" in metrics["per_class_metrics"]
    assert metrics["per_class_metrics"]["INDEX"]["precision"] == 1.0
    assert metrics["per_class_metrics"]["INDEX"]["recall"] == 0.5


def test_run_benchmark_full(tmp_path: Path):
    processed_dir = Path("pc/data/processed_v2")
    if processed_dir.exists() and (processed_dir / "train.csv").exists():
        json_out = tmp_path / "report.json"
        csv_out = tmp_path / "summary.csv"

        results = run_benchmark(processed_dir)
        assert "benchmark_results" in results
        assert len(results["benchmark_results"]) == 8
        assert "best_val_model" in results
        assert "best_test_model" in results

        export_benchmark_reports(results, json_out, csv_out)
        assert json_out.exists()
        assert csv_out.exists()
