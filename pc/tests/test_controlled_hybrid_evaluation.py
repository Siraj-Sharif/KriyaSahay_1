"""Unit tests for Controlled 14-Gesture Hybrid Pipeline Evaluation Tool.

Verifies:
- Gesture key mapping for 14 canonical commands and 4 negative test cases.
- Route mapping (Extra Trees for INDEX_FINGER, TWO_FINGERS, PINKY; HaGRID for 11 others).
- Trial tracker metric calculations (accuracy, latencies, switch counts, STOP safety checks).
- JSON summary serialization.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure pc/tools is on sys.path
pc_tools = Path(__file__).resolve().parent.parent / "tools"
if str(pc_tools) not in sys.path:
    sys.path.insert(0, str(pc_tools))

import pytest

from controlled_hybrid_14_gesture_evaluation import (
    EXPECTED_MODEL_ROUTES,
    GESTURE_KEY_MAP,
    FrameObservation,
    HybridTrialTracker,
    TrialResult,
    parse_args,
)


def test_gesture_key_map_completeness():
    """Verify all 14 canonical commands and 4 negative test keys are mapped."""
    canonical_14 = {
        "CALL",
        "CLOSED_FIST",
        "FOUR_FINGERS",
        "GRABBING",
        "GRIP",
        "THUMBS_UP",
        "PINKY",
        "MIDDLE_FINGER",
        "OK",
        "INDEX_FINGER",
        "INDEX_PINKY",
        "STOP",
        "TWO_FINGERS",
        "THREE_FINGERS",
    }
    mapped_values = set(GESTURE_KEY_MAP.values())
    for cmd in canonical_14:
        assert cmd in mapped_values, f"Missing canonical command in key map: {cmd}"

    assert "NO_HAND" in mapped_values
    assert "TWO_HANDS" in mapped_values
    assert "OUT_OF_FRAME" in mapped_values
    assert "UNSUPPORTED" in mapped_values


def test_expected_model_routes():
    """Verify locked routing rules."""
    assert EXPECTED_MODEL_ROUTES["INDEX_FINGER"] == "EXTRA_TREES"
    assert EXPECTED_MODEL_ROUTES["TWO_FINGERS"] == "EXTRA_TREES"
    assert EXPECTED_MODEL_ROUTES["PINKY"] == "EXTRA_TREES"

    hagrid_commands = [
        "CALL",
        "CLOSED_FIST",
        "FOUR_FINGERS",
        "GRABBING",
        "GRIP",
        "THUMBS_UP",
        "MIDDLE_FINGER",
        "OK",
        "INDEX_PINKY",
        "STOP",
        "THREE_FINGERS",
    ]
    for cmd in hagrid_commands:
        assert EXPECTED_MODEL_ROUTES[cmd] == "HAGRID", f"Command {cmd} should route to HAGRID"


def test_trial_tracker_metrics_calculation():
    """Verify accuracy, switch count, latency statistics, and STOP safety metrics."""
    tracker = HybridTrialTracker(ground_truth="INDEX_FINGER", duration_s=10.0)

    # Add 10 frames: 8 correct, 1 wrong, 1 NO_COMMAND
    # Frame 1-5: INDEX_FINGER (Extra Trees, 25ms)
    for i in range(5):
        obs = FrameObservation(
            timestamp_ms=i * 33.3,
            ground_truth="INDEX_FINGER",
            predicted_command="INDEX_FINGER",
            model_used="EXTRA_TREES",
            raw_prediction="INDEX",
            confidence=0.90,
            latency_ms=25.0 + i,
            hand_count="1",
            stabilizer_state="STABLE",
            emitted_command="INDEX_FINGER",
            is_stop_armed=False,
            is_stop_active=False,
        )
        tracker.add_frame(obs)

    # Frame 6-8: INDEX_FINGER (Extra Trees, 30ms)
    for i in range(5, 8):
        obs = FrameObservation(
            timestamp_ms=i * 33.3,
            ground_truth="INDEX_FINGER",
            predicted_command="INDEX_FINGER",
            model_used="EXTRA_TREES",
            raw_prediction="INDEX",
            confidence=0.88,
            latency_ms=30.0,
            hand_count="1",
            stabilizer_state="STABLE",
            emitted_command="INDEX_FINGER",
            is_stop_armed=False,
            is_stop_active=False,
        )
        tracker.add_frame(obs)

    # Frame 9: NO_COMMAND
    tracker.add_frame(
        FrameObservation(
            timestamp_ms=8 * 33.3,
            ground_truth="INDEX_FINGER",
            predicted_command="NO_COMMAND",
            model_used="N/A",
            raw_prediction="NONE",
            confidence=0.0,
            latency_ms=10.0,
            hand_count="0",
            stabilizer_state="UNKNOWN_STATE",
            emitted_command=None,
            is_stop_armed=False,
            is_stop_active=False,
        )
    )

    # Frame 10: Wrong command (GRIP)
    tracker.add_frame(
        FrameObservation(
            timestamp_ms=9 * 33.3,
            ground_truth="INDEX_FINGER",
            predicted_command="GRIP",
            model_used="HAGRID",
            raw_prediction="grip",
            confidence=0.75,
            latency_ms=28.0,
            hand_count="1",
            stabilizer_state="STABLE",
            emitted_command="GRIP",
            is_stop_armed=False,
            is_stop_active=False,
        )
    )

    summary = tracker.summarize()

    assert summary.total_frames == 10
    assert summary.correct_frames == 8
    assert summary.no_command_frames == 1
    assert summary.wrong_frames == 1
    assert summary.accuracy_percentage == 80.0
    assert summary.command_switch_count == 2  # INDEX_FINGER -> NO_COMMAND -> GRIP
    assert summary.expected_route == "EXTRA_TREES"
    assert summary.model_route_distribution["EXTRA_TREES"] == 8
    assert summary.model_route_distribution["HAGRID"] == 1
    assert summary.stop_safety_details["non_stop_bypass_count"] == 0
    assert summary.stop_safety_details["safety_path_integrity"] is True


def test_negative_test_metrics_calculation():
    """Verify negative test ground truth (NO_HAND) expects NO_COMMAND."""
    tracker = HybridTrialTracker(ground_truth="NO_HAND", duration_s=5.0)

    # Add 5 frames of NO_COMMAND
    for i in range(5):
        tracker.add_frame(
            FrameObservation(
                timestamp_ms=i * 33.3,
                ground_truth="NO_HAND",
                predicted_command="NO_COMMAND",
                model_used="N/A",
                raw_prediction="NONE",
                confidence=0.0,
                latency_ms=8.0,
                hand_count="0",
                stabilizer_state="UNKNOWN_STATE",
                emitted_command=None,
                is_stop_armed=False,
                is_stop_active=False,
            )
        )

    summary = tracker.summarize()
    assert summary.total_frames == 5
    assert summary.correct_frames == 5
    assert summary.wrong_frames == 0
    assert summary.accuracy_percentage == 100.0


def test_stop_safety_details_tracking():
    """Verify STOP safety path tracking detects active STOP and non-STOP bypass violations if any."""
    tracker = HybridTrialTracker(ground_truth="STOP", duration_s=5.0)

    # Frame 1: STOP active, emitted STOP
    tracker.add_frame(
        FrameObservation(
            timestamp_ms=0.0,
            ground_truth="STOP",
            predicted_command="STOP",
            model_used="RULE_BASED",
            raw_prediction="STOP",
            confidence=1.0,
            latency_ms=5.0,
            hand_count="1",
            stabilizer_state="STOP",
            emitted_command="STOP",
            is_stop_armed=True,
            is_stop_active=True,
        )
    )

    summary = tracker.summarize()
    assert summary.stop_safety_details["stop_armed_frames"] == 1
    assert summary.stop_safety_details["stop_active_frames"] == 1
    assert summary.stop_safety_details["stop_emitted_frames"] == 1
    assert summary.stop_safety_details["non_stop_bypass_count"] == 0
    assert summary.stop_safety_details["safety_path_integrity"] is True


def test_json_summary_serialization():
    """Verify TrialResult dataclass serializes cleanly to JSON."""
    tracker = HybridTrialTracker(ground_truth="THUMBS_UP", duration_s=1.0)
    tracker.add_frame(
        FrameObservation(
            timestamp_ms=0.0,
            ground_truth="THUMBS_UP",
            predicted_command="THUMBS_UP",
            model_used="HAGRID",
            raw_prediction="like",
            confidence=0.95,
            latency_ms=27.0,
            hand_count="1",
            stabilizer_state="STABLE",
            emitted_command="THUMBS_UP",
            is_stop_armed=False,
            is_stop_active=False,
        )
    )

    summary = tracker.summarize()
    dumped = json.dumps(asdict(summary))
    loaded = json.loads(dumped)

    assert loaded["ground_truth"] == "THUMBS_UP"
    assert loaded["expected_route"] == "HAGRID"
    assert loaded["accuracy_percentage"] == 100.0


def test_cli_parse_args():
    """Verify CLI argument parser defaults and override flags."""
    args = parse_args(["--camera", "1", "--duration", "3.0", "--output-dir", "custom/reports"])
    assert args.camera == 1
    assert args.duration == 3.0
    assert args.output_dir == Path("custom/reports")
