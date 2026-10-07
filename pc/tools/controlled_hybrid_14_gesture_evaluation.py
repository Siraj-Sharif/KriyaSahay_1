"""Controlled 14-Gesture Real-World Evaluation Tool for NeuroGrip Hybrid Pipeline.

Evaluates the real-world performance, latency, gesture routing, and safety behavior of the
FINAL HYBRID PRODUCTION PIPELINE (HaGRID ResNet18 + Extra Trees 68-D + Rule-Based STOP Safety).

Features:
- Standard 14 Canonical Commands Ground-Truth Selector (Keys 1-9, 0, '-', '=', '[', ']')
- Negative & Safety Test Selectors ('N'=No Hand, 'M'=Two Hands, 'X'=Out of Frame, 'U'=Unsupported, 'S'=Toggle STOP)
- Interactive 5-second trial recorder ([SPACE] key to start trial)
- Structured metric collection (correct/wrong/NO_COMMAND frames, latencies, switch count, top predictions)
- Dedicated STOP safety logging (model prediction vs rule-based detector vs final emitted command)
- Output saved to pc/reports/hybrid_14_gesture_evaluation_logs.json

Controls:
  1-9, 0, -, =, [, ] : Select 14 canonical commands
  N                 : Select NO_HAND trial
  M                 : Select TWO_HANDS trial
  X                 : Select OUT_OF_FRAME trial
  U                 : Select UNSUPPORTED trial
  S                 : Toggle Software STOP Arming State
  SPACE             : Start / Stop 5-second trial recording
  Q / ESC           : Exit evaluation window
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Ensure pc/src is on sys.path
pc_src = Path(__file__).resolve().parent.parent / "src"
if str(pc_src) not in sys.path:
    sys.path.insert(0, str(pc_src))

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.communication.transport import DisplayConsoleTransport
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.recognition.base import RecognitionResult
from neurogrip.stabilization.temporal import StabilizerState

logger = logging.getLogger(__name__)

# Key mapping for canonical gestures & negative tests
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
    "n": "NO_HAND",
    "m": "TWO_HANDS",
    "x": "OUT_OF_FRAME",
    "u": "UNSUPPORTED",
}

# Expected model route mapping for 14 canonical commands
EXPECTED_MODEL_ROUTES: Dict[str, str] = {
    "INDEX_FINGER": "EXTRA_TREES",
    "TWO_FINGERS": "EXTRA_TREES",
    "PINKY": "EXTRA_TREES",
    "CALL": "HAGRID",
    "CLOSED_FIST": "HAGRID",
    "FOUR_FINGERS": "HAGRID",
    "GRABBING": "HAGRID",
    "GRIP": "HAGRID",
    "THUMBS_UP": "HAGRID",
    "MIDDLE_FINGER": "HAGRID",
    "OK": "HAGRID",
    "INDEX_PINKY": "HAGRID",
    "STOP": "HAGRID",
    "THREE_FINGERS": "HAGRID",
    "NO_HAND": "N/A",
    "TWO_HANDS": "N/A",
    "OUT_OF_FRAME": "N/A",
    "UNSUPPORTED": "N/A",
}


@dataclass
class FrameObservation:
    timestamp_ms: float
    ground_truth: str
    predicted_command: str
    model_used: str
    raw_prediction: str
    confidence: float
    latency_ms: float
    hand_count: str
    stabilizer_state: str
    emitted_command: Optional[str]
    is_stop_armed: bool
    is_stop_active: bool


@dataclass
class TrialResult:
    trial_id: str
    timestamp: str
    ground_truth: str
    expected_route: str
    duration_s: float
    total_frames: int
    correct_frames: int
    wrong_frames: int
    no_command_frames: int
    accuracy_percentage: float
    command_switch_count: int
    emitted_command_count: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    observed_fps: float
    model_route_distribution: Dict[str, int]
    top_prediction_distribution: Dict[str, int]
    raw_prediction_distribution: Dict[str, int]
    stop_safety_details: Dict[str, Any]
    frame_samples: List[Dict[str, Any]] = field(default_factory=list)


class HybridTrialTracker:
    """Tracks per-frame metrics for a single 5-second ground-truth trial."""

    def __init__(self, ground_truth: str, duration_s: float = 5.0) -> None:
        self.ground_truth = ground_truth.upper()
        self.duration_s = duration_s
        self.trial_id = f"TRIAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.ground_truth}"
        self.timestamp = datetime.now().isoformat()
        self.expected_route = EXPECTED_MODEL_ROUTES.get(self.ground_truth, "UNKNOWN")

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        self.observations: List[FrameObservation] = []
        self.last_predicted_command: Optional[str] = None
        self.command_switch_count: int = 0

    @property
    def is_complete(self) -> bool:
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) >= self.duration_s

    def add_frame(self, obs: FrameObservation) -> None:
        if self.start_time is None:
            self.start_time = time.time()

        if self.last_predicted_command is not None and obs.predicted_command != self.last_predicted_command:
            self.command_switch_count += 1
        self.last_predicted_command = obs.predicted_command

        self.observations.append(obs)
        self.end_time = time.time()

    def summarize(self) -> TrialResult:
        total_frames = len(self.observations)
        if total_frames == 0:
            return TrialResult(
                trial_id=self.trial_id,
                timestamp=self.timestamp,
                ground_truth=self.ground_truth,
                expected_route=self.expected_route,
                duration_s=0.0,
                total_frames=0,
                correct_frames=0,
                wrong_frames=0,
                no_command_frames=0,
                accuracy_percentage=0.0,
                command_switch_count=0,
                emitted_command_count=0,
                mean_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                mean_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                observed_fps=0.0,
                model_route_distribution={},
                top_prediction_distribution={},
                raw_prediction_distribution={},
                stop_safety_details={},
            )

        correct_frames = 0
        wrong_frames = 0
        no_command_frames = 0
        emitted_count = 0

        confidences: List[float] = []
        latencies: List[float] = []
        routes: Counter[str] = Counter()
        predictions: Counter[str] = Counter()
        raw_predictions: Counter[str] = Counter()

        stop_armed_count = 0
        stop_active_count = 0
        stop_emitted_count = 0
        non_stop_bypass_count = 0

        # Ground-truth evaluation rules
        target = self.ground_truth

        for obs in self.observations:
            latencies.append(obs.latency_ms)
            routes[obs.model_used] += 1
            predictions[obs.predicted_command] += 1
            raw_predictions[obs.raw_prediction] += 1

            if obs.confidence > 0.0:
                confidences.append(obs.confidence)

            if obs.emitted_command and obs.emitted_command != "NO_COMMAND":
                emitted_count += 1
                if obs.emitted_command == "STOP":
                    stop_emitted_count += 1

            if obs.is_stop_armed:
                stop_armed_count += 1
            if obs.is_stop_active:
                stop_active_count += 1

            # Safety check: track if non-STOP command was emitted while stop_active
            if obs.is_stop_active and obs.emitted_command and obs.emitted_command not in ("STOP", "NO_COMMAND"):
                non_stop_bypass_count += 1

            # Classification correctness logic
            if target in ("NO_HAND", "TWO_HANDS", "OUT_OF_FRAME", "UNSUPPORTED"):
                if obs.predicted_command == "NO_COMMAND":
                    correct_frames += 1
                else:
                    wrong_frames += 1
            else:
                if obs.predicted_command == target:
                    correct_frames += 1
                elif obs.predicted_command == "NO_COMMAND":
                    no_command_frames += 1
                else:
                    wrong_frames += 1

        acc_pct = (correct_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
        actual_duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else self.duration_s
        fps = (total_frames / actual_duration) if actual_duration > 0 else 0.0

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0
        max_conf = float(np.max(confidences)) if confidences else 0.0

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        median_lat = float(np.median(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        stop_details = {
            "stop_armed_frames": stop_armed_count,
            "stop_active_frames": stop_active_count,
            "stop_emitted_frames": stop_emitted_count,
            "non_stop_bypass_count": non_stop_bypass_count,
            "safety_path_integrity": (non_stop_bypass_count == 0),
        }

        # Keep 5 sample frame dicts for diagnostics
        sample_frames = [asdict(obs) for obs in self.observations[:5]]

        return TrialResult(
            trial_id=self.trial_id,
            timestamp=self.timestamp,
            ground_truth=self.ground_truth,
            expected_route=self.expected_route,
            duration_s=actual_duration,
            total_frames=total_frames,
            correct_frames=correct_frames,
            wrong_frames=wrong_frames,
            no_command_frames=no_command_frames,
            accuracy_percentage=acc_pct,
            command_switch_count=self.command_switch_count,
            emitted_command_count=emitted_count,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            mean_latency_ms=mean_lat,
            median_latency_ms=median_lat,
            p95_latency_ms=p95_lat,
            observed_fps=fps,
            model_route_distribution=dict(routes),
            top_prediction_distribution=dict(predictions),
            raw_prediction_distribution=dict(raw_predictions),
            stop_safety_details=stop_details,
            frame_samples=sample_frames,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled 14-Gesture Evaluation Tool for Final Hybrid Production Pipeline."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720).")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30).")
    parser.add_argument("--duration", type=float, default=5.0, help="Trial recording duration in seconds (default: 5.0).")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process before exiting.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pc/reports"),
        help="Directory to save JSON trial reports (default: pc/reports).",
    )
    return parser.parse_args(argv)


def run_evaluation_tool(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = AppConfig.default()
    cfg.camera.index = args.camera
    cfg.camera.width = args.width
    cfg.camera.height = args.height
    cfg.camera.fps = args.fps
    cfg.camera.backend = "opencv"
    cfg.serial.enabled = False
    cfg.serial.mock = True

    configure_logging(cfg.logging)

    print("==========================================================================")
    print(" NEUROGRIP — CONTROLLED 14-GESTURE HYBRID PIPELINE EVALUATION TOOL")
    print("==========================================================================")
    print(f"  Camera Device Index : {args.camera} ({args.width}x{args.height} @ {args.fps} FPS)")
    print(f"  Trial Duration      : {args.duration} seconds per recording")
    print(f"  Output Report Dir   : {args.output_dir.resolve()}")
    print("--------------------------------------------------------------------------")
    print("  KEYBOARD SELECTION CONTROLS:")
    print("    1: CALL          2: CLOSED_FIST   3: FOUR_FINGERS   4: GRABBING")
    print("    5: GRIP          6: THUMBS_UP     7: PINKY          8: MIDDLE_FINGER")
    print("    9: OK            0: INDEX_FINGER  -: INDEX_PINKY    =: STOP")
    print("    [: TWO_FINGERS   ]: THREE_FINGERS")
    print("  NEGATIVE / SAFETY TESTS:")
    print("    N: NO_HAND       M: TWO_HANDS     X: OUT_OF_FRAME   U: UNSUPPORTED")
    print("    S: Toggle STOP Arming State")
    print("  OPERATIONAL CONTROLS:")
    print("    SPACE: Start / Stop 5-second trial recording")
    print("    Q/ESC: Exit evaluation")
    print("==========================================================================")

    transport = DisplayConsoleTransport()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "hybrid_14_gesture_evaluation_logs.json"

    completed_trials: List[TrialResult] = []
    selected_ground_truth: str = "INDEX_FINGER"
    active_tracker: Optional[HybridTrialTracker] = None
    frames_processed: int = 0

    window_name = "NeuroGrip — Controlled 14-Gesture Hybrid Pipeline Evaluation"

    with NeuroGripPipeline(config=cfg, transport=transport) as pipeline:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

        try:
            while pipeline.is_running:
                frame_container, rendered, viz_state = pipeline.process_frame()

                if rendered is None or frame_container.frame is None:
                    continue

                # Build frame observation snapshot
                rec_res = pipeline.last_rec_result
                stab_res = pipeline.last_stab_result

                model_used_val = getattr(rec_res, "model_used", "N/A") if rec_res else "N/A"
                raw_pred_val = getattr(rec_res, "raw_prediction", "NONE") if rec_res else "NONE"
                pred_cmd = viz_state.command
                emitted_cmd = getattr(stab_res, "emitted_command", None) if stab_res else None

                obs = FrameObservation(
                    timestamp_ms=frame_container.timestamp_ms,
                    ground_truth=selected_ground_truth,
                    predicted_command=pred_cmd,
                    model_used=model_used_val,
                    raw_prediction=raw_pred_val,
                    confidence=float(viz_state.confidence),
                    latency_ms=float(viz_state.latency_ms),
                    hand_count=str(viz_state.hand_count),
                    stabilizer_state=str(viz_state.stabilizer_state),
                    emitted_command=emitted_cmd,
                    is_stop_armed=pipeline.is_stop_armed,
                    is_stop_active=viz_state.is_stop_active,
                )

                # Active trial recording loop
                if active_tracker is not None:
                    active_tracker.add_frame(obs)
                    if active_tracker.is_complete:
                        trial_summary = active_tracker.summarize()
                        completed_trials.append(trial_summary)

                        print(f"\n[TRIAL RECORDED] {trial_summary.trial_id}")
                        print(f"  Ground Truth: {trial_summary.ground_truth} | Route: {trial_summary.expected_route}")
                        print(f"  Frames: {trial_summary.total_frames} | Correct: {trial_summary.correct_frames} | Wrong: {trial_summary.wrong_frames} | NO_CMD: {trial_summary.no_command_frames}")
                        print(f"  Accuracy: {trial_summary.accuracy_percentage:.1f}% | Mean Latency: {trial_summary.mean_latency_ms:.2f} ms | FPS: {trial_summary.observed_fps:.1f}")
                        print(f"  Predictions: {trial_summary.top_prediction_distribution}")
                        print(f"  Routes Used: {trial_summary.model_route_distribution}")

                        # Save updated JSON file after every trial
                        with open(report_file, "w", encoding="utf-8") as f:
                            json.dump([asdict(t) for t in completed_trials], f, indent=2)
                        print(f"  Updated JSON log saved to {report_file}")

                        active_tracker = None

                # Render evaluation HUD overlay
                eval_rendered = rendered.copy()
                h, w, _ = eval_rendered.shape

                # Top evaluation banner
                banner_color = (0, 165, 255) if active_tracker else (50, 50, 50)
                cv2.rectangle(eval_rendered, (0, 0), (w, 40), banner_color, -1)

                status_txt = f"RECORDING TRIAL ({selected_ground_truth})..." if active_tracker else "READY FOR TRIAL"
                cv2.putText(
                    eval_rendered,
                    f"EVAL TARGET: [{selected_ground_truth}]  |  STATUS: {status_txt}  |  TRIALS COMPLETED: {len(completed_trials)}",
                    (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(window_name, eval_rendered)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("\nExiting evaluation tool cleanly.")
                    break

                key_char = chr(key).lower() if key < 128 else ""
                if key_char in GESTURE_KEY_MAP:
                    selected_ground_truth = GESTURE_KEY_MAP[key_char]
                    print(f"Selected ground truth: [{selected_ground_truth}] (Expected Route: {EXPECTED_MODEL_ROUTES.get(selected_ground_truth, 'UNKNOWN')})")
                elif key_char == "s":
                    armed = pipeline.toggle_stop_arm()
                    print(f"Software STOP state toggled: Armed={armed}")
                elif key == ord(" "):
                    if active_tracker is None:
                        active_tracker = HybridTrialTracker(selected_ground_truth, duration_s=args.duration)
                        print(f"\nStarted 5-second trial recording for [{selected_ground_truth}]...")
                    else:
                        print("Stopping active trial recording early...")
                        active_tracker.end_time = time.time()
                        trial_summary = active_tracker.summarize()
                        completed_trials.append(trial_summary)
                        with open(report_file, "w", encoding="utf-8") as f:
                            json.dump([asdict(t) for t in completed_trials], f, indent=2)
                        active_tracker = None

                frames_processed += 1
                if args.max_frames is not None and frames_processed >= args.max_frames:
                    print(f"\nReached max frames limit ({args.max_frames}). Exiting evaluation loop.")
                    break

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by KeyboardInterrupt.")
        finally:
            cv2.destroyAllWindows()

    print(f"\nEvaluation session complete. Total trials recorded: {len(completed_trials)}")
    if completed_trials:
        print(f"Final results logged to: {report_file.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation_tool())
