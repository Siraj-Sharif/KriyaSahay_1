"""Real-World Webcam Evaluation & Trial Logger for HaGRID Architecture Comparison.

Side-by-side live webcam diagnostic tool comparing Architecture A (Full-Frame Baseline)
and Architecture B (Hand-Crop Pipeline) with interactive 5-second trial recording,
metric calculation, and safety rule enforcement.

Key Controls:
- Keys 1-9, 0, '-', '=', '[', ']': Select expected NeuroGrip gesture (14 gestures).
- SPACE: Start / stop 5-second trial recording.
- 'm': Toggle side-by-side vs single frame mode.
- 'q' / ESC: Exit diagnostic window.

SAFETY RULES:
- Diagnostic tool only.
- NO serial communication, NO ESP32 hardware dispatch, NO motor control.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure pc/src is on sys.path
pc_src = Path(__file__).resolve().parent.parent / "src"
if str(pc_src) not in sys.path:
    sys.path.insert(0, str(pc_src))

from neurogrip.hagrid import (
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    HaGRIDClassifier,
    HaGRIDConfig,
    HaGRIDHandDetector,
    predict_architecture_a,
    predict_architecture_b,
)

# 14 Canonical Gesture Key Mapping
GESTURE_KEY_MAP: Dict[str, str] = {
    "1": "CALL",
    "2": "CLOSED_FIST",
    "3": "FOUR_FINGERS",
    "4": "GRABBING",
    "5": "GRIP",
    "6": "THUMBS_UP",
    "7": "PINKY",
    "8": "MIDDLE_FINGER",
    "9": "OK",
    "0": "INDEX_FINGER",
    "-": "INDEX_PINKY",
    "=": "STOP",
    "[": "TWO_FINGERS",
    "]": "THREE_FINGERS",
}


@dataclass
class TrialMetrics:
    """Class tracking single-architecture evaluation metrics over a trial period."""

    architecture: str
    expected_gesture: str
    total_observed_frames: int = 0
    matching_frames: int = 0
    wrong_command_frames: int = 0
    no_command_frames: int = 0
    switch_count: int = 0
    last_command: Optional[str] = None
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def manual_recognition_consistency(self) -> float:
        """Calculate manual recognition consistency percentage: matching / total * 100."""
        if self.total_observed_frames == 0:
            return 0.0
        return (self.matching_frames / self.total_observed_frames) * 100.0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def median_latency_ms(self) -> float:
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    def update(self, predicted_command: str, latency_ms: float):
        """Update metrics for a newly observed frame."""
        self.total_observed_frames += 1
        self.latencies_ms.append(latency_ms)

        if predicted_command == self.expected_gesture:
            self.matching_frames += 1
        elif predicted_command == NO_COMMAND:
            self.no_command_frames += 1
        else:
            self.wrong_command_frames += 1

        if self.last_command is not None and predicted_command != self.last_command:
            self.switch_count += 1
        self.last_command = predicted_command

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for structured JSON logging."""
        return {
            "architecture": self.architecture,
            "expected_gesture": self.expected_gesture,
            "total_observed_frames": self.total_observed_frames,
            "matching_frames": self.matching_frames,
            "wrong_command_frames": self.wrong_command_frames,
            "no_command_frames": self.no_command_frames,
            "manual_recognition_consistency_pct": round(self.manual_recognition_consistency, 2),
            "switch_count": self.switch_count,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


def calculate_agreement_rate(cmd_a_list: List[str], cmd_b_list: List[str]) -> float:
    """Calculate diagnostic agreement rate percentage between Architecture A and B."""
    if not cmd_a_list or len(cmd_a_list) != len(cmd_b_list):
        return 0.0
    agreed = sum(1 for a, b in zip(cmd_a_list, cmd_b_list) if a == b)
    return (agreed / len(cmd_a_list)) * 100.0


def run_realworld_evaluation(
    camera_index: int = 0,
    model_name: str = "ResNet18",
    detector_conf: float = 0.40,
    pad_ratio: float = 0.15,
    trial_duration: float = 5.0,
):
    """Run real-time side-by-side webcam evaluation diagnostic."""
    print("=" * 85)
    print(" NeuroGrip Phase 2.2 — Real-World Recognition Evaluation GUI")
    print("=" * 85)
    print("On-Screen Key Mapping for Expected Gesture Selection:")
    for k, v in GESTURE_KEY_MAP.items():
        print(f"  [{k}] -> {v:<15}", end="  " if k in ["5", "0", "]"] else "")
        if k in ["5", "0", "]"]:
            print()
    print("\nControls:")
    print("  [SPACE]  : Start/Stop 5-second evaluation trial")
    print("  [m]      : Toggle Side-by-Side / Single-Frame View")
    print("  [q/ESC]  : Exit Diagnostic GUI")
    print("=" * 85)

    config = HaGRIDConfig(
        model_name=model_name,
        detector_conf_threshold=detector_conf,
        crop_pad_ratio=pad_ratio,
    )

    classifier = HaGRIDClassifier(config=config, auto_load=True)
    detector = HaGRIDHandDetector(config=config, auto_load=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open USB webcam at index {camera_index}.")
        sys.exit(1)

    selected_expected_gesture = "CALL"  # Default expected gesture
    trial_active = False
    trial_start_time = 0.0

    metrics_a: Optional[TrialMetrics] = None
    metrics_b: Optional[TrialMetrics] = None
    cmds_a_trial: List[str] = []
    cmds_b_trial: List[str] = []

    frame_count = 0
    side_by_side = True

    fps_count = 0
    t_start = time.perf_counter()
    fps_a = 0.0
    fps_b = 0.0

    log_file_path = Path(__file__).resolve().parent.parent / "reports" / "realworld_trial_logs.json"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    saved_trials: List[Dict[str, Any]] = []

    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[!] Error reading webcam frame.")
                break

            frame_count += 1
            h, w, _ = frame.shape

            # Run Architecture A & Architecture B on identical frame
            res_a = predict_architecture_a(classifier, frame, is_bgr=True)
            res_b = predict_architecture_b(
                detector,
                classifier,
                frame,
                is_bgr=True,
                conf_threshold=detector_conf,
                pad_ratio=pad_ratio,
            )

            cmd_a = res_a["mapped_command"]
            cmd_b = res_b["mapped_command"]
            agreement = "AGREE" if cmd_a == cmd_b else "DISAGREE"

            # FPS calculation
            fps_count += 1
            if fps_count % 10 == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                fps_a = 10.0 / elapsed
                fps_b = 10.0 / elapsed
                t_start = t_now

            # Check trial status
            trial_elapsed = 0.0
            if trial_active:
                trial_elapsed = time.perf_counter() - trial_start_time
                if metrics_a and metrics_b:
                    metrics_a.update(cmd_a, res_a["latency_ms"])
                    metrics_b.update(cmd_b, res_b["latency_ms"])
                    cmds_a_trial.append(cmd_a)
                    cmds_b_trial.append(cmd_b)

                # Check 5-second completion
                if trial_elapsed >= trial_duration:
                    trial_active = False
                    agree_pct = calculate_agreement_rate(cmds_a_trial, cmds_b_trial)
                    print("\n" + "=" * 70)
                    print(f" TRACT COMPLETE ({trial_duration}s) FOR EXPECTED: {selected_expected_gesture}")
                    print("=" * 70)
                    print(f"[*] Arch A Manual Consistency: {metrics_a.manual_recognition_consistency:5.1f}% (Matches: {metrics_a.matching_frames}/{metrics_a.total_observed_frames})")
                    print(f"[*] Arch B Manual Consistency: {metrics_b.manual_recognition_consistency:5.1f}% (Matches: {metrics_b.matching_frames}/{metrics_b.total_observed_frames})")
                    print(f"[*] A/B Diagnostic Agreement:  {agree_pct:5.1f}%")
                    print(f"[*] Arch A Switches / Latency: {metrics_a.switch_count} switches | {metrics_a.mean_latency_ms:.1f} ms")
                    print(f"[*] Arch B Switches / Latency: {metrics_b.switch_count} switches | {metrics_b.mean_latency_ms:.1f} ms")
                    print("=" * 70)

                    # Append to log list
                    trial_record = {
                        "timestamp": datetime.now().isoformat(),
                        "expected_gesture": selected_expected_gesture,
                        "duration_s": trial_duration,
                        "agreement_pct": round(agree_pct, 2),
                        "arch_a": metrics_a.to_dict(),
                        "arch_b": metrics_b.to_dict(),
                    }
                    saved_trials.append(trial_record)
                    with open(log_file_path, "w") as f:
                        json.dump(saved_trials, f, indent=2)

            # Render Overlay Frames
            frame_a = frame.copy()
            frame_b = frame.copy()

            # --- HUD Arch A ---
            cv2.rectangle(frame_a, (0, 0), (w, 140), (20, 20, 20), -1)
            cv2.putText(frame_a, "[ARCH A: FULL-FRAME BASELINE]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_a, f"Raw Class: {res_a['raw_class']} ({res_a['confidence']*100:.1f}%)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cmd_color_a = (0, 255, 0) if cmd_a != NO_COMMAND else (0, 0, 255)
            cv2.putText(frame_a, f"Cmd: {cmd_a}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cmd_color_a, 2)
            cv2.putText(frame_a, f"Latency: {res_a['latency_ms']:4.1f} ms | FPS: {fps_a:4.1f}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # --- HUD Arch B ---
            cv2.rectangle(frame_b, (0, 0), (w, 140), (20, 20, 20), -1)
            cv2.putText(frame_b, "[ARCH B: HAND-CROP PIPELINE]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            hand_cnt = res_b["detected_hands_count"]
            boxes = res_b["detected_boxes"]

            if hand_cnt == 0:
                cv2.putText(frame_b, "Hands: 0 -> NO_COMMAND (NO_HAND)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)
            elif hand_cnt == 1:
                box = boxes[0]
                crop_box = res_b["selected_crop_box"]
                cv2.rectangle(frame_b, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                if crop_box:
                    cv2.rectangle(frame_b, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255, 255, 0), 1)

                cv2.putText(frame_b, f"Raw Class: {res_b['raw_class']} ({res_b['confidence']*100:.1f}%)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
                cmd_color_b = (0, 255, 0) if cmd_b != NO_COMMAND else (0, 0, 255)
                cv2.putText(frame_b, f"Cmd: {cmd_b}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, cmd_color_b, 2)
            else:
                for idx, b in enumerate(boxes):
                    cv2.rectangle(frame_b, (b[0], b[1]), (b[2], b[3]), (0, 140, 255), 2)
                    cv2.putText(frame_b, f"Hand {idx+1}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
                cv2.putText(frame_b, f"Hands: {hand_cnt} -> NO_COMMAND (MULTI_HAND_AMBIGUITY)", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame_b, "Cmd: NO_COMMAND (SAFETY REJECT)", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            bd = res_b["breakdown"]
            cv2.putText(frame_b, f"Det: {bd['detector_ms']:4.1f}ms | Clf: {bd['classifier_ms']:4.1f}ms | Tot: {bd['total_ms']:4.1f}ms | FPS: {fps_b:4.1f}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Global Trial & Agreement Banner at Bottom
            banner_bg = (0, 80, 0) if trial_active else (40, 40, 40)
            cv2.rectangle(frame_a, (0, h - 60), (w, h), banner_bg, -1)
            cv2.rectangle(frame_b, (0, h - 60), (w, h), banner_bg, -1)

            status_str = f"TRIAL RECORDING ({trial_elapsed:.1f}s / {trial_duration}s)" if trial_active else "IDLE (Press SPACE to start 5s trial)"
            agree_color = (0, 255, 0) if agreement == "AGREE" else (0, 165, 255)

            cv2.putText(frame_a, f"Expected: [{selected_expected_gesture}] | {status_str}", (15, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_a, f"Agreement: {agreement} | Frame: #{frame_count}", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, agree_color, 1)

            cv2.putText(frame_b, f"Expected: [{selected_expected_gesture}] | {status_str}", (15, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame_b, f"Agreement: {agreement} | Frame: #{frame_count}", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, agree_color, 1)

            if side_by_side:
                disp_w = 640
                disp_h = 480
                f_a_res = cv2.resize(frame_a, (disp_w, disp_h))
                f_b_res = cv2.resize(frame_b, (disp_w, disp_h))
                combined = np.hstack((f_a_res, f_b_res))
                cv2.imshow("NeuroGrip Phase 2.2 Real-World Evaluation [Left: Arch A Full-Frame | Right: Arch B Hand-Crop]", combined)
            else:
                cv2.imshow("NeuroGrip Phase 2.2 Real-World Evaluation [Arch B Hand-Crop View]", frame_b)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[*] Exit requested.")
                break
            elif key == ord("m"):
                side_by_side = not side_by_side
                cv2.destroyAllWindows()
            elif key == ord(" "):
                # Toggle trial recording
                if not trial_active:
                    trial_active = True
                    trial_start_time = time.perf_counter()
                    metrics_a = TrialMetrics(architecture="A_FULL_FRAME", expected_gesture=selected_expected_gesture)
                    metrics_b = TrialMetrics(architecture="B_HAND_CROP", expected_gesture=selected_expected_gesture)
                    cmds_a_trial = []
                    cmds_b_trial = []
                    print(f"[*] Started 5s evaluation trial for expected gesture: '{selected_expected_gesture}'")
                else:
                    trial_active = False
                    print("[*] Trial manually cancelled.")
            else:
                char_key = chr(key).lower() if 0 <= key < 128 else ""
                if char_key in GESTURE_KEY_MAP:
                    selected_expected_gesture = GESTURE_KEY_MAP[char_key]
                    print(f"[*] Selected expected gesture: [{selected_expected_gesture}]")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[*] Evaluation completed. Total trials recorded and saved to: {log_file_path}")


def main():
    parser = argparse.ArgumentParser(description="HaGRID Phase 2.2 Real-World Recognition Evaluation GUI")
    parser.add_argument("-c", "--camera", type=int, default=0, help="USB webcam index (default: 0)")
    parser.add_argument("-m", "--model", type=str, default="ResNet18", help="Classifier model name")
    parser.add_argument("-d", "--detector_conf", type=float, default=0.40, help="Detector threshold")
    parser.add_argument("-p", "--pad", type=float, default=0.15, help="Hand bounding box padding margin ratio")
    parser.add_argument("-t", "--duration", type=float, default=5.0, help="Evaluation trial duration in seconds")
    args = parser.parse_args()

    run_realworld_evaluation(
        camera_index=args.camera,
        model_name=args.model,
        detector_conf=args.detector_conf,
        pad_ratio=args.pad,
        trial_duration=args.duration,
    )


if __name__ == "__main__":
    main()
