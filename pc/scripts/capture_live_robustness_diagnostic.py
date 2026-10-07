"""
pc/scripts/capture_live_robustness_diagnostic.py
────────────────────────────────────────────────────
Phase 10K.1 — Live CV Pipeline Robustness Diagnostic Capture Tool.
Isolated read-only diagnostic script. Does NOT alter any production code or behavior.

Captures 100-200 consecutive valid hand frames while a user holds a static gesture.
Computes:
1. Per-feature mean, standard deviation, and coefficient of variation (CV).
2. Group-level feature stability across the 8 v2 feature vector groups.
3. Frame-to-frame L2 feature displacement.
4. Raw prediction switching rate (top-1 class instability).
5. Raw confidence & margin statistics (mean, min, max, std).
6. Handedness stability and dropped frame tracking.
7. Explicit breakdown distinguishing: Landmark Jitter vs Feature Sensitivity vs Classifier Instability.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Ensure src directory is in sys.path
PC_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PC_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.detector import HandDetector
from neurogrip.hand_tracking.landmarks import DetectionResult
from neurogrip.recognition.ml_recognizer import MLRecognizer

logger = logging.getLogger("neurogrip.robustness_diagnostic")

# Feature Group Boundaries for 68-D v2 Feature Vector
FEATURE_GROUPS: dict[str, tuple[int, int]] = {
    "1. Norm 2D Coords [0..39]": (0, 40),
    "2. Relative Z Depths [40..44]": (40, 45),
    "3. 3D Joint Angles [45..54]": (45, 55),
    "4. Straightness Indices [55..59]": (55, 60),
    "5. Inter-Finger Contrast [60..64]": (60, 65),
    "6. Thumb-Index Direction [65]": (65, 66),
    "7. Thumb-to-Palm Distance [66]": (66, 67),
    "8. Min Straightness [67]": (67, 68),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 10K.1 Live Robustness Diagnostic Capture Tool."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--gesture",
        type=str,
        default="PINKY",
        help="Target held gesture name for metadata tracking (e.g. PINKY, THUMB_ONLY, GRAB, MIDDLE).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Number of valid hand frames to collect (default: 150).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without OpenCV visualization window.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for raw frame recordings.",
    )
    return parser


def run_diagnostic_capture(
    camera_index: int = 0,
    target_gesture: str = "PINKY",
    max_valid_frames: int = 150,
    no_gui: bool = False,
    output_json: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Execute robust live frame capture and statistical diagnostic analysis.
    """
    cfg = AppConfig.default()
    cfg.camera.index = camera_index
    cfg.camera.backend = "opencv"
    cfg.visualization.enabled = not no_gui

    logger.info("Initializing Robustness Diagnostic Capture...")
    logger.info("  Camera Index     : %d", camera_index)
    logger.info("  Target Gesture   : %s", target_gesture)
    logger.info("  Target Frames    : %d", max_valid_frames)
    logger.info("  GUI Window       : %s", not no_gui)

    pipeline = NeuroGripPipeline(config=cfg)
    if not pipeline.initialize():
        logger.error("Pipeline initialization failed.")
        return {}

    valid_frame_records: list[dict[str, Any]] = []
    dropped_frames_count = 0
    total_camera_frames = 0
    last_handedness: Optional[str] = None
    handedness_flips = 0
    handedness_counts: dict[str, int] = {}

    print(f"\n==========================================================================")
    print(f"NEUROGRIP PHASE 10K.1 — LIVE ROBUSTNESS DIAGNOSTIC CAPTURE")
    print(f"==========================================================================")
    print(f"Please hold gesture '{target_gesture}' STEADY in front of the camera...")
    print(f"Collecting {max_valid_frames} valid hand frames...")
    print(f"==========================================================================\n")

    try:
        while len(valid_frame_records) < max_valid_frames:
            total_camera_frames += 1
            frame_container, rendered, viz_state = pipeline.process_frame()

            if not frame_container.is_valid or frame_container.frame is None:
                dropped_frames_count += 1
                time.sleep(0.005)
                continue

            det_res = pipeline.last_detection_result
            if det_res is None or det_res.is_empty or det_res.is_ambiguous or det_res.primary_hand is None:
                dropped_frames_count += 1
                continue

            primary_hand = det_res.primary_hand
            current_handedness = primary_hand.handedness.value
            handedness_counts[current_handedness] = handedness_counts.get(current_handedness, 0) + 1

            if last_handedness is not None and current_handedness != last_handedness:
                handedness_flips += 1
            last_handedness = current_handedness

            features = pipeline.last_features
            rec_res = pipeline.last_rec_result

            if features is None or rec_res is None or len(features) != 68:
                dropped_frames_count += 1
                continue

            # Extract raw probabilities and top 2 predictions
            all_scores = rec_res.all_scores
            sorted_scores = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)

            top1_cls, top1_prob = sorted_scores[0] if len(sorted_scores) > 0 else ("UNKNOWN", 0.0)
            top2_cls, top2_prob = sorted_scores[1] if len(sorted_scores) > 1 else ("UNKNOWN", 0.0)
            margin = top1_prob - top2_prob

            frame_rec = {
                "frame_index": total_camera_frames,
                "valid_sample_index": len(valid_frame_records) + 1,
                "handedness": current_handedness,
                "handedness_score": float(primary_hand.score),
                "features": features.tolist(),
                "top1_class": top1_cls,
                "top1_prob": float(top1_prob),
                "top2_class": top2_cls,
                "top2_prob": float(top2_prob),
                "margin": float(margin),
                "hybrid_label": rec_res.label,
                "hybrid_conf": float(rec_res.confidence),
            }

            valid_frame_records.append(frame_rec)

            if len(valid_frame_records) % 25 == 0:
                logger.info(
                    "Captured %d/%d frames | Latest: Top1=%s (%.2f), Top2=%s (%.2f), Margin=%.2f",
                    len(valid_frame_records),
                    max_valid_frames,
                    top1_cls,
                    top1_prob,
                    top2_cls,
                    top2_prob,
                    margin,
                )

    finally:
        pipeline.shutdown()

    # ─────────────────────────────────────────────────────────
    # STATISTICAL ANALYSIS OF CAPTURED FRAMES
    # ─────────────────────────────────────────────────────────
    N = len(valid_frame_records)
    if N == 0:
        logger.error("No valid hand frames were captured.")
        return {}

    feature_matrix = np.array([r["features"] for r in valid_frame_records], dtype=np.float64)  # Shape (N, 68)

    # 1. Per-feature mean, std, CV
    feat_means = np.mean(feature_matrix, axis=0)
    feat_stds = np.std(feature_matrix, axis=0)
    feat_cvs = feat_stds / (np.abs(feat_means) + 1e-6)

    # 2. Group-level metrics
    group_stats: dict[str, dict[str, float]] = {}
    for g_name, (start_i, end_i) in FEATURE_GROUPS.items():
        g_stds = feat_stds[start_i:end_i]
        g_cvs = feat_cvs[start_i:end_i]
        g_means = feat_means[start_i:end_i]

        # Calculate frame-to-frame displacement for group
        g_matrix = feature_matrix[:, start_i:end_i]
        if N > 1:
            g_disp = np.linalg.norm(np.diff(g_matrix, axis=0), axis=1)
            avg_g_disp = float(np.mean(g_disp))
        else:
            avg_g_disp = 0.0

        group_stats[g_name] = {
            "mean_std": float(np.mean(g_stds)),
            "max_std": float(np.max(g_stds)),
            "mean_cv": float(np.mean(g_cvs)),
            "avg_disp": avg_g_disp,
        }

    # 3. Overall Frame-to-Frame Feature Vector L2 Displacement
    if N > 1:
        f_diffs = np.diff(feature_matrix, axis=0)  # Shape (N-1, 68)
        frame_displacements = np.linalg.norm(f_diffs, axis=1)
        disp_mean = float(np.mean(frame_displacements))
        disp_std = float(np.std(frame_displacements))
        disp_max = float(np.max(frame_displacements))
        disp_min = float(np.min(frame_displacements))
    else:
        disp_mean = disp_std = disp_max = disp_min = 0.0

    # 4. Raw Prediction Switching Rate
    top1_classes = [r["top1_class"] for r in valid_frame_records]
    prediction_switches = 0
    for i in range(1, N):
        if top1_classes[i] != top1_classes[i - 1]:
            prediction_switches += 1

    switching_rate = float(prediction_switches / (N - 1)) if N > 1 else 0.0

    # 5. Top-1 Confidence & Margin Stats
    top1_probs = np.array([r["top1_prob"] for r in valid_frame_records], dtype=np.float64)
    margins = np.array([r["margin"] for r in valid_frame_records], dtype=np.float64)

    conf_stats = {
        "mean": float(np.mean(top1_probs)),
        "min": float(np.min(top1_probs)),
        "max": float(np.max(top1_probs)),
        "std": float(np.std(top1_probs)),
    }

    margin_stats = {
        "mean": float(np.mean(margins)),
        "min": float(np.min(margins)),
        "max": float(np.max(margins)),
        "std": float(np.std(margins)),
    }

    # 6. Gated Pass Count (>= 0.60 confidence)
    above_60_count = int(np.sum(top1_probs >= 0.60))
    above_60_pct = float(above_60_count / N * 100.0)

    # Compile Summary Object
    analysis_results: dict[str, Any] = {
        "target_gesture": target_gesture,
        "total_valid_frames": N,
        "total_camera_frames": total_camera_frames,
        "dropped_frames": dropped_frames_count,
        "handedness_counts": handedness_counts,
        "handedness_flips": handedness_flips,
        "prediction_switches": prediction_switches,
        "prediction_switching_rate": switching_rate,
        "frames_above_60_conf": above_60_count,
        "pct_above_60_conf": above_60_pct,
        "confidence_stats": conf_stats,
        "margin_stats": margin_stats,
        "frame_displacement_stats": {
            "mean": disp_mean,
            "std": disp_std,
            "min": disp_min,
            "max": disp_max,
        },
        "group_feature_stats": group_stats,
        "per_feature_means": feat_means.tolist(),
        "per_feature_stds": feat_stds.tolist(),
        "per_feature_cvs": feat_cvs.tolist(),
    }

    # Print Summary Report to Console
    print("\n==========================================================================")
    print(f"LIVE ROBUSTNESS STATISTICAL SUMMARY — GESTURE: {target_gesture}")
    print("==========================================================================")
    print(f"Total Valid Frames Collected : {N}")
    print(f"Dropped / Invalid Frames    : {dropped_frames_count}")
    print(f"Handedness Distribution      : {handedness_counts} (Flips: {handedness_flips})")
    print(f"Raw Prediction Switches      : {prediction_switches} ({switching_rate*100:.1f}% switching rate)")
    print(f"Frames >= 0.60 Confidence    : {above_60_count}/{N} ({above_60_pct:.1f}%)")
    print("\nConfidence (Top-1 Probability):")
    print(f"  Mean: {conf_stats['mean']:.4f} | Min: {conf_stats['min']:.4f} | Max: {conf_stats['max']:.4f} | Std: {conf_stats['std']:.4f}")
    print("Margin (Top-1 minus Top-2):")
    print(f"  Mean: {margin_stats['mean']:.4f} | Min: {margin_stats['min']:.4f} | Max: {margin_stats['max']:.4f} | Std: {margin_stats['std']:.4f}")
    print("Frame-to-Frame L2 Vector Displacement:")
    print(f"  Mean: {disp_mean:.4f} | Std: {disp_std:.4f} | Min: {disp_min:.4f} | Max: {disp_max:.4f}")

    print("\nFeature Group Breakdown (Stability Ranking):")
    print(f"{'Feature Group':<35} | {'Mean Std':<10} | {'Max Std':<10} | {'Mean CV':<10} | {'Avg Disp':<10}")
    print("-" * 85)
    for g_name, g_info in group_stats.items():
        print(f"{g_name:<35} | {g_info['mean_std']:10.4f} | {g_info['max_std']:10.4f} | {g_info['mean_cv']:10.4f} | {g_info['avg_disp']:10.4f}")

    print("==========================================================================\n")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": analysis_results, "frames": valid_frame_records}, f, indent=2)
        logger.info("Saved raw diagnostic session recording to '%s'.", output_json)

    return analysis_results


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(AppConfig.default().logging)
    run_diagnostic_capture(
        camera_index=args.camera,
        target_gesture=args.gesture,
        max_valid_frames=args.max_frames,
        no_gui=args.no_gui,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
