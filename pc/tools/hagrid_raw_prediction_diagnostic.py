"""Isolated Raw HaGRID Prediction Diagnostic Tool for Phase 2.3.

This tool runs full-frame inference using the official HaGRID ResNet18 model and
captures un-gated, raw top-5 prediction distributions for the 3 target gestures:
1. INDEX_FINGER (Expected raw: point)
2. TWO_FINGERS  (Expected raw: two_up, two_up_inverted)
3. PINKY        (Expected raw: little_finger)

CRITICAL DIAGNOSTIC RULES:
- Reuses existing neurogrip.hagrid.model.HaGRIDClassifier.
- Captures RAW model outputs WITHOUT applying confidence threshold gates.
- Preserves full floating-point precision in JSON logging.
- Does NOT alter production configuration, thresholds, or taxonomy.

Controls:
  [1]     : Select INDEX_FINGER (Expected: point)
  [2]     : Select TWO_FINGERS  (Expected: two_up, two_up_inverted)
  [3]     : Select PINKY        (Expected: little_finger)
  [SPACE] : Start 5-second diagnostic trial recording
  [ESC]   : Exit diagnostic
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure pc/src is on sys.path
pc_src = Path(__file__).resolve().parent.parent / "src"
if str(pc_src) not in sys.path:
    sys.path.insert(0, str(pc_src))

from neurogrip.hagrid import (
    HAGRID_CLASSES,
    HAGRID_INDEX_TO_CLASS,
    HaGRIDClassifier,
    HaGRIDConfig,
    map_hagrid_to_neurogrip,
)

# Target Gesture Specs for Phase 2.3 — Locked 14 NeuroGrip Canonical Gestures
TARGET_GESTURES: Dict[str, Dict[str, Any]] = {
    "CALL": {
        "key": "1",
        "alt_keys": ["1", "!"],
        "canonical": "CALL",
        "expected_raw": ["call"],
    },
    "CLOSED_FIST": {
        "key": "2",
        "alt_keys": ["2", "@"],
        "canonical": "CLOSED_FIST",
        "expected_raw": ["fist"],
    },
    "FOUR_FINGERS": {
        "key": "3",
        "alt_keys": ["3", "#"],
        "canonical": "FOUR_FINGERS",
        "expected_raw": ["four"],
    },
    "GRABBING": {
        "key": "4",
        "alt_keys": ["4", "$"],
        "canonical": "GRABBING",
        "expected_raw": ["grabbing"],
    },
    "GRIP": {
        "key": "5",
        "alt_keys": ["5", "%"],
        "canonical": "GRIP",
        "expected_raw": ["grip"],
    },
    "THUMBS_UP": {
        "key": "6",
        "alt_keys": ["6", "^"],
        "canonical": "THUMBS_UP",
        "expected_raw": ["like"],
    },
    "PINKY": {
        "key": "7",
        "alt_keys": ["7", "&"],
        "canonical": "PINKY",
        "expected_raw": ["little_finger"],
    },
    "MIDDLE_FINGER": {
        "key": "8",
        "alt_keys": ["8", "*"],
        "canonical": "MIDDLE_FINGER",
        "expected_raw": ["middle_finger"],
    },
    "OK": {
        "key": "9",
        "alt_keys": ["9", "("],
        "canonical": "OK",
        "expected_raw": ["ok"],
    },
    "INDEX_FINGER": {
        "key": "0",
        "alt_keys": ["0", ")"],
        "canonical": "INDEX_FINGER",
        "expected_raw": ["point"],
    },
    "INDEX_PINKY": {
        "key": "-",
        "alt_keys": ["-", "_"],
        "canonical": "INDEX_PINKY",
        "expected_raw": ["rock"],
    },
    "STOP": {
        "key": "=",
        "alt_keys": ["=", "+"],
        "canonical": "STOP",
        "expected_raw": ["stop", "stop_inverted"],
    },
    "TWO_FINGERS": {
        "key": "[",
        "alt_keys": ["[", "{"],
        "canonical": "TWO_FINGERS",
        "expected_raw": ["two_up", "two_up_inverted"],
    },
    "THREE_FINGERS": {
        "key": "]",
        "alt_keys": ["]", "}"],
        "canonical": "THREE_FINGERS",
        "expected_raw": ["three", "three2", "three3", "three_gun"],
    },
}

KEY_TO_GESTURE_KEY: Dict[str, str] = {}
for g_name, g_info in TARGET_GESTURES.items():
    for k in g_info.get("alt_keys", [g_info["key"]]):
        KEY_TO_GESTURE_KEY[k] = g_name


@dataclass
class DiagnosticFrameRecord:
    """Individual frame record preserving un-gated raw model probabilities."""

    frame_index: int
    timestamp: str
    raw_top1_class: str
    raw_top1_probability: float
    top5: List[Dict[str, Any]]  # List of {"class": str, "probability": float}
    canonical_command: str


def compute_raw_predictions_unfiltered(
    classifier: HaGRIDClassifier, image: np.ndarray, is_bgr: bool = True, top_k: int = 5
) -> Dict[str, Any]:
    """Run ResNet18 model inference and return un-gated raw probabilities and top-5 predictions.

    Args:
        classifier: Initialized HaGRIDClassifier instance.
        image: NumPy BGR frame.
        is_bgr: True if frame is in BGR color order.
        top_k: Number of top raw predictions to return.

    Returns:
        Dict with raw_top1_class, raw_top1_probability, top5 list, and mapped canonical command.
    """
    input_tensor = classifier.preprocess(image, is_bgr=is_bgr)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = classifier.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    probs_np = probs.cpu().numpy()
    top_indices = np.argsort(probs_np)[::-1][:top_k]

    top5_list = []
    for idx in top_indices:
        cls_name = HAGRID_INDEX_TO_CLASS[int(idx)]
        prob_val = float(probs_np[idx])  # Preserve full float precision
        top5_list.append({"class": cls_name, "probability": prob_val})

    top1_class = top5_list[0]["class"]
    top1_prob = top5_list[0]["probability"]
    canonical_cmd = map_hagrid_to_neurogrip(top1_class)

    return {
        "raw_top1_class": top1_class,
        "raw_top1_probability": top1_prob,
        "top5": top5_list,
        "canonical_command": canonical_cmd,
        "latency_ms": latency_ms,
    }


def compute_trial_summary_stats(
    target_name: str, expected_raw_classes: List[str], frames: List[DiagnosticFrameRecord]
) -> Dict[str, Any]:
    """Compute trial summary statistics for terminal output and report logging."""
    if not frames:
        return {}

    total_frames = len(frames)
    raw_top1_classes = [f.raw_top1_class for f in frames]
    raw_probs = [f.raw_top1_probability for f in frames]

    # Calculate hit rate: RAW TOP-1 CLASS in expected_raw_classes
    matching_hits = sum(1 for c in raw_top1_classes if c in expected_raw_classes)
    hit_rate_pct = (matching_hits / total_frames) * 100.0

    # Calculate distribution
    class_counts = Counter(raw_top1_classes)
    distribution_pct = {cls: round((cnt / total_frames) * 100.0, 2) for cls, cnt in class_counts.most_common()}

    # Calculate confidence stats
    mean_conf_pct = float(np.mean(raw_probs)) * 100.0
    median_conf_pct = float(np.median(raw_probs)) * 100.0
    p95_conf_pct = float(np.percentile(raw_probs, 95)) * 100.0

    # Non-target / unsupported top-1 %
    unsupported_cnt = sum(1 for c in raw_top1_classes if c not in expected_raw_classes)
    unsupported_pct = (unsupported_cnt / total_frames) * 100.0

    # Raw switch count
    switch_count = 0
    for i in range(1, total_frames):
        if raw_top1_classes[i] != raw_top1_classes[i - 1]:
            switch_count += 1

    # Top confusions (excluding expected raw classes)
    confusions = {cls: pct for cls, pct in distribution_pct.items() if cls not in expected_raw_classes}
    top_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "target_gesture": target_name,
        "expected_raw_classes": expected_raw_classes,
        "total_frames": total_frames,
        "hit_rate_pct": round(hit_rate_pct, 2),
        "mean_confidence_pct": round(mean_conf_pct, 2),
        "median_confidence_pct": round(median_conf_pct, 2),
        "p95_confidence_pct": round(p95_conf_pct, 2),
        "unsupported_pct": round(unsupported_pct, 2),
        "switch_count": switch_count,
        "distribution_pct": distribution_pct,
        "top_confusions": top_confusions,
    }


def print_trial_terminal_summary(stats: Dict[str, Any]):
    """Print concise, clear trial summary in the terminal."""
    print("\n" + "=" * 70)
    print(f" TARGET: {stats['target_gesture']}")
    print(f" EXPECTED RAW: {', '.join(stats['expected_raw_classes'])}")
    print("=" * 70)
    print(f"FRAMES: {stats['total_frames']}")
    print("\nRAW PREDICTION DISTRIBUTION:")
    for cls, pct in stats["distribution_pct"].items():
        tag = " [EXPECTED]" if cls in stats["expected_raw_classes"] else ""
        print(f"  {cls:<20}: {pct:6.2f}%{tag}")

    print(f"\nEXPECTED RAW CLASS HIT RATE: {stats['hit_rate_pct']:.2f}%")
    print(f"MEAN RAW CONFIDENCE:         {stats['mean_confidence_pct']:.2f}%")
    print(f"MEDIAN RAW CONFIDENCE:       {stats['median_confidence_pct']:.2f}%")
    print(f"P95 RAW CONFIDENCE:          {stats['p95_confidence_pct']:.2f}%")
    print(f"UNSUPPORTED / NON-TARGET:    {stats['unsupported_pct']:.2f}%")
    print(f"RAW CLASS SWITCH COUNT:      {stats['switch_count']}")

    if stats["top_confusions"]:
        print("\nTOP CONFUSIONS:")
        for rank, (cls, pct) in enumerate(stats["top_confusions"], start=1):
            print(f"  {rank}. {cls:<20} -> {pct:.2f}%")
    print("=" * 70 + "\n")


def run_raw_prediction_diagnostic(camera_index: int = 0, trial_duration: float = 5.0):
    """Run real-time USB webcam raw prediction diagnostic GUI."""
    print("=" * 70)
    print(" NeuroGrip Phase 2.3 — Raw HaGRID Prediction Diagnostic")
    print("=" * 70)
    print("Keyboard Target Selection Controls:")
    print("  [1] : CALL           (Expected: call)")
    print("  [2] : CLOSED_FIST    (Expected: fist)")
    print("  [3] : FOUR_FINGERS   (Expected: four)")
    print("  [4] : GRABBING       (Expected: grabbing)")
    print("  [5] : GRIP           (Expected: grip)")
    print("  [6] : THUMBS_UP      (Expected: like)")
    print("  [7] : PINKY          (Expected: little_finger)")
    print("  [8] : MIDDLE_FINGER  (Expected: middle_finger)")
    print("  [9] : OK             (Expected: ok)")
    print("  [0] : INDEX_FINGER   (Expected: point)")
    print("  [-] : INDEX_PINKY    (Expected: rock)")
    print("  [=] : STOP           (Expected: stop, stop_inverted)")
    print("  [[] : TWO_FINGERS    (Expected: two_up, two_up_inverted)")
    print("  []] : THREE_FINGERS  (Expected: three, three2, three3, three_gun)")
    print("  [SPACE] : Start 5-second raw diagnostic trial")
    print("  [ESC]   : Exit diagnostic")
    print("=" * 70)

    config = HaGRIDConfig(model_name="ResNet18")
    classifier = HaGRIDClassifier(config=config, auto_load=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open USB webcam at index {camera_index}.")
        sys.exit(1)

    window_name = "NeuroGrip Phase 2.3 — Raw HaGRID Prediction Diagnostic"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    current_target_key = "INDEX_FINGER"
    target_info = TARGET_GESTURES[current_target_key]

    recording = False
    rec_start_time = 0.0
    rec_frames: List[DiagnosticFrameRecord] = []

    fps_count = 0
    t_start = time.perf_counter()
    fps = 0.0

    json_file_path = Path(__file__).resolve().parent.parent / "reports" / "hagrid_raw_prediction_diagnostic.json"
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    all_trials_json: List[Dict[str, Any]] = []
    if json_file_path.exists():
        try:
            with open(json_file_path, "r") as f:
                all_trials_json = json.load(f)
        except Exception:
            all_trials_json = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error reading webcam frame. Exiting.")
                break

            h, w, _ = frame.shape

            # Compute un-gated raw model predictions
            pred = compute_raw_predictions_unfiltered(classifier, frame, is_bgr=True, top_k=5)

            fps_count += 1
            if fps_count % 10 == 0:
                t_now = time.perf_counter()
                fps = 10.0 / (t_now - t_start)
                t_start = t_now

            # Recording loop
            rec_elapsed = 0.0
            if recording:
                rec_elapsed = time.perf_counter() - rec_start_time
                rec_frames.append(
                    DiagnosticFrameRecord(
                        frame_index=len(rec_frames),
                        timestamp=datetime.now().isoformat(),
                        raw_top1_class=pred["raw_top1_class"],
                        raw_top1_probability=pred["raw_top1_probability"],
                        top5=pred["top5"],
                        canonical_command=pred["canonical_command"],
                    )
                )

                if rec_elapsed >= trial_duration:
                    recording = False
                    # Compute trial statistics and print terminal summary
                    stats = compute_trial_summary_stats(
                        target_name=current_target_key,
                        expected_raw_classes=target_info["expected_raw"],
                        frames=rec_frames,
                    )
                    print_trial_terminal_summary(stats)

                    # Append trial JSON
                    trial_json_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "target_gesture": current_target_key,
                        "expected_raw_classes": target_info["expected_raw"],
                        "duration_s": trial_duration,
                        "total_frames": len(rec_frames),
                        "summary_stats": stats,
                        "frames": [asdict(f) for f in rec_frames],
                    }
                    all_trials_json.append(trial_json_entry)

                    with open(json_file_path, "w") as f:
                        json.dump(all_trials_json, f, indent=2)
                    print(f"[*] Trial recording saved to: {json_file_path}")

            # Render Overlay Window HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 240), (15, 15, 15), -1)

            # Controls HUD Banner
            cv2.putText(
                overlay,
                "CONTROLS: 1-9,0,-,=,[,] Select Gesture | SPACE Start Trial | ESC Exit",
                (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

            # TARGET Banner
            exp_str = " / ".join(target_info["expected_raw"])
            cv2.putText(
                overlay,
                f"TARGET: {current_target_key} (Expected: {exp_str})",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # STATUS Banner
            status_str = f"STATUS: RECORDING ({rec_elapsed:.1f}s / {trial_duration:.1f}s)" if recording else "STATUS: READY"
            status_color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(
                overlay,
                status_str,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            # RAW Prediction & Un-Gated Confidence
            is_expected_hit = pred["raw_top1_class"] in target_info["expected_raw"]
            raw_color = (0, 255, 0) if is_expected_hit else (0, 165, 255)

            cv2.putText(
                overlay,
                f"RAW TOP-1: {pred['raw_top1_class']}",
                (15, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                raw_color,
                2,
            )
            cv2.putText(
                overlay,
                f"CONF: {pred['raw_top1_probability']*100:.2f}%",
                (350, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # Render TOP-5 Raw Predictions
            cv2.putText(overlay, "TOP 5 RAW PREDICTIONS:", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_pos = 138
            for rank, p in enumerate(pred["top5"][:5], start=1):
                p_tag = " [EXPECTED]" if p["class"] in target_info["expected_raw"] else ""
                cv2.putText(
                    overlay,
                    f"  {rank}. {p['class']:<16} {p['probability']*100:6.2f}%{p_tag}",
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (200, 200, 200),
                    1,
                )
                y_pos += 16

            # Mapped Canonical Command & FPS
            cv2.putText(
                overlay,
                f"CANONICAL MAPPED: {pred['canonical_command']}",
                (15, 228),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                f"FPS: {fps:4.1f} | Latency: {pred['latency_ms']:4.1f}ms",
                (420, 228),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 180, 180),
                1,
            )

            # Recording Status Banner at Bottom
            rec_banner_color = (0, 0, 180) if recording else (40, 40, 40)
            cv2.rectangle(overlay, (0, h - 45), (w, h), rec_banner_color, -1)
            rec_status = (
                f"RECORDING RAW TRIAL ({rec_elapsed:.1f}s / {trial_duration:.1f}s)..."
                if recording
                else f"READY | TARGET: {current_target_key} | Press SPACE to record 5s trial | ESC to exit"
            )
            cv2.putText(overlay, rec_status, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            cv2.imshow(window_name, overlay)

            key_raw = cv2.waitKey(1)
            if key_raw != -1:
                key = key_raw & 0xFF
                if key == 27:  # ESC
                    print("[*] ESC pressed. Exiting diagnostic.")
                    break
                elif key == 32:  # SPACE
                    if not recording:
                        recording = True
                        rec_start_time = time.perf_counter()
                        rec_frames = []
                        print(f"[*] Started 5s raw diagnostic recording for target: {current_target_key}")
                    else:
                        recording = False
                        print("[*] Recording cancelled.")
                elif key != 255 and key != 0:
                    char_key = chr(key)
                    if char_key in KEY_TO_GESTURE_KEY:
                        current_target_key = KEY_TO_GESTURE_KEY[char_key]
                        target_info = TARGET_GESTURES[current_target_key]
                        print(
                            f"[*] Selected target gesture: [{current_target_key}] "
                            f"(Key: '{char_key}', Expected raw: {target_info['expected_raw']})"
                        )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Webcam released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="HaGRID Raw Prediction Diagnostic CLI Tool")
    parser.add_argument("-c", "--camera", type=int, default=0, help="USB webcam index (default: 0)")
    parser.add_argument("-t", "--duration", type=float, default=5.0, help="Trial recording duration in seconds")
    args = parser.parse_args()

    run_raw_prediction_diagnostic(camera_index=args.camera, trial_duration=args.duration)


if __name__ == "__main__":
    main()
