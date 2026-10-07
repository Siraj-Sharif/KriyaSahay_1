"""
tests/test_stabilization.py
────────────────────────────
Unit tests for TemporalStabilizer.
Verifies sliding window majority vote, threshold calculations, UNKNOWN handling,
command transitions, duplicate emission suppression, and software-armed STOP safety bypass.
"""
import pytest

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig
from neurogrip.recognition.base import RecognitionResult
from neurogrip.stabilization.temporal import StabilizerResult, StabilizerState, TemporalStabilizer


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def make_result(label: str, confidence: float = 0.95) -> RecognitionResult:
    return RecognitionResult(
        label=label,
        confidence=confidence,
        all_scores={label: confidence},
        recognizer_type="mock",
    )


# ─────────────────────────────────────────────────────────
# Initial & Basic Tests
# ─────────────────────────────────────────────────────────

def test_stabilizer_initial_state():
    """Verify initial state of TemporalStabilizer."""
    stabilizer = TemporalStabilizer()
    assert stabilizer.current_state == StabilizerState.UNKNOWN_STATE
    assert stabilizer.last_emitted_command is None


def test_stabilizer_empty_and_none_input():
    """Verify stabilizer handles None or UNKNOWN input safely."""
    stabilizer = TemporalStabilizer()
    res = stabilizer.update(None)

    assert res.state == StabilizerState.UNKNOWN_STATE
    assert res.stable_command is None
    assert res.emitted_command is None
    assert res.confidence == 0.0
    assert res.window_fill == 1


# ─────────────────────────────────────────────────────────
# Stabilization & Window Logic Tests
# ─────────────────────────────────────────────────────────

def test_stabilizer_window_building_and_index_stable():
    """
    Test window building (window_size=15, threshold=0.70, min_frames=5):
    11 INDEX out of 15 frames = 73.3% >= 70% -> INDEX becomes stable.
    """
    stabilizer = TemporalStabilizer(window_size=15, vote_threshold=0.70, min_frames_required=5)
    rec_index = make_result("INDEX")
    rec_unk = make_result("UNKNOWN")
    rec_mid = make_result("MIDDLE")

    # Feed 4 INDEX frames -> N=4 < min_frames_required(5) -> STABILIZING
    for _ in range(4):
        res_building = stabilizer.update(rec_index)

    assert res_building.state == StabilizerState.STABILIZING
    assert res_building.emitted_command is None

    # 5th INDEX frame -> N=5 >= 5, 5/5=100% >= 70% -> STABLE, emitted ONCE
    res_stable = stabilizer.update(rec_index)
    assert res_stable.state == StabilizerState.STABLE
    assert res_stable.stable_command == "INDEX"
    assert res_stable.emitted_command == "INDEX"

    # 6th INDEX frame -> Remains STABLE, but emitted_command is None (suppressed)
    res_repeat = stabilizer.update(rec_index)
    assert res_repeat.state == StabilizerState.STABLE
    assert res_repeat.stable_command == "INDEX"
    assert res_repeat.emitted_command is None  # De-duplicated!

    # Feed 5 more INDEX (total 11 INDEX)
    for _ in range(5):
        stabilizer.update(rec_index)

    # Feed 2 UNKNOWN and 2 MIDDLE -> window has 11 INDEX, 2 UNKNOWN, 2 MIDDLE (N=15)
    # INDEX votes = 11/15 = 73.3% >= 70%
    stabilizer.update(rec_unk)
    stabilizer.update(rec_unk)
    stabilizer.update(rec_mid)
    res_final = stabilizer.update(rec_mid)

    assert res_final.state == StabilizerState.STABLE
    assert res_final.stable_command == "INDEX"
    assert res_final.winning_count == 11
    assert pytest.approx(res_final.confidence, 0.01) == 0.733


def test_stabilizer_below_threshold():
    """
    Test below-threshold behavior (window_size=15, threshold=0.70):
    10 INDEX out of 15 frames = 66.7% < 70% -> STABILIZING state, no stable command.
    """
    stabilizer = TemporalStabilizer(window_size=15, vote_threshold=0.70, min_frames_required=5)
    rec_index = make_result("INDEX")
    rec_mid = make_result("MIDDLE")

    # Feed 10 INDEX and 5 MIDDLE -> N=15, INDEX votes = 10/15 = 66.7% < 70%
    for _ in range(10):
        stabilizer.update(rec_index)
    for _ in range(4):
        stabilizer.update(rec_mid)

    res = stabilizer.update(rec_mid)

    assert res.state == StabilizerState.STABILIZING
    assert res.stable_command is None
    assert res.emitted_command is None
    assert pytest.approx(res.confidence, 0.01) == 0.667


def test_stabilizer_exact_threshold_boundary():
    """Test exact threshold boundary (e.g. 7 out of 10 = 70.0% >= 70.0%)."""
    stabilizer = TemporalStabilizer(window_size=10, vote_threshold=0.70, min_frames_required=5)
    rec_index = make_result("INDEX")
    rec_unk = make_result("UNKNOWN")

    for _ in range(7):
        stabilizer.update(rec_index)
    for _ in range(2):
        stabilizer.update(rec_unk)

    res = stabilizer.update(rec_unk)  # N=10, 7 INDEX, 3 UNKNOWN -> 7/10 = 70.0%

    assert res.state == StabilizerState.STABLE
    assert res.stable_command == "INDEX"
    assert res.winning_count == 7


def test_stabilizer_unknown_never_becomes_stable():
    """Verify UNKNOWN predictions count against threshold and NEVER become a stable command."""
    stabilizer = TemporalStabilizer(window_size=10, vote_threshold=0.70, min_frames_required=5)
    rec_unk = make_result("UNKNOWN")

    # Feed 10 UNKNOWN frames
    for _ in range(10):
        res = stabilizer.update(rec_unk)

    assert res.state == StabilizerState.UNKNOWN_STATE
    assert res.stable_command is None
    assert res.emitted_command is None
    assert res.winning_count == 0


def test_stabilizer_command_transition():
    """Test transitioning from stable INDEX to stable MIDDLE."""
    stabilizer = TemporalStabilizer(window_size=10, vote_threshold=0.70, min_frames_required=5)
    rec_index = make_result("INDEX")
    rec_mid = make_result("MIDDLE")

    # 1. Fill window with INDEX -> INDEX stable, emitted "INDEX"
    for _ in range(10):
        res_i = stabilizer.update(rec_index)

    assert res_i.state == StabilizerState.STABLE
    assert res_i.stable_command == "INDEX"

    # 2. Fill window with MIDDLE -> transitions to MIDDLE, emitted "MIDDLE"
    for _ in range(7):
        res_m = stabilizer.update(rec_mid)

    assert res_m.state == StabilizerState.STABLE
    assert res_m.stable_command == "MIDDLE"
    assert res_m.emitted_command == "MIDDLE"


def test_stabilizer_reset():
    """Verify reset() clears window and emission history."""
    stabilizer = TemporalStabilizer(window_size=5, vote_threshold=0.70, min_frames_required=3)
    rec_index = make_result("INDEX")

    for _ in range(5):
        stabilizer.update(rec_index)

    assert stabilizer.current_state == StabilizerState.STABLE

    stabilizer.reset()
    assert stabilizer.current_state == StabilizerState.UNKNOWN_STATE
    assert stabilizer.last_emitted_command is None

    # After reset, INDEX emitted again on first stability
    for _ in range(3):
        res = stabilizer.update(rec_index)
    assert res.emitted_command == "INDEX"


# ─────────────────────────────────────────────────────────
# STOP Safety Path Tests
# ─────────────────────────────────────────────────────────

def test_stabilizer_stop_armed_immediate_bypass():
    """Verify armed STOP immediately bypasses window and emits STOP."""
    stabilizer = TemporalStabilizer(window_size=15, vote_threshold=0.70)
    rec_stop = make_result("STOP")

    # Send armed STOP prediction
    res = stabilizer.update(rec_stop, is_stop_armed=True)

    assert res.state == StabilizerState.STOP
    assert res.stable_command == "STOP"
    assert res.emitted_command == "STOP"
    assert res.confidence == 1.0


def test_stabilizer_stop_unarmed_decoupled():
    """Verify STOP prediction is recognized, stabilized, and emitted as STOP (normal command)."""
    stabilizer = TemporalStabilizer(window_size=15, vote_threshold=0.70)
    rec_stop = make_result("STOP")

    # Send STOP prediction -> recognized as STOP and emitted as normal command
    res = stabilizer.update(rec_stop, is_stop_armed=False)

    assert res.state == StabilizerState.STOP
    assert res.stable_command == "STOP"
    assert res.emitted_command == "STOP"


def test_stabilizer_stop_interrupts_active_stable_command():
    """Verify armed STOP immediately interrupts an active stable gesture."""
    stabilizer = TemporalStabilizer(window_size=10, vote_threshold=0.70, min_frames_required=5)
    rec_index = make_result("INDEX")
    rec_stop = make_result("STOP")

    # 1. Establish stable INDEX
    for _ in range(10):
        stabilizer.update(rec_index)
    assert stabilizer.current_state == StabilizerState.STABLE

    # 2. Armed STOP arrives -> interrupts INDEX, enters STOP state
    res_stop = stabilizer.update(rec_stop, is_stop_armed=True)
    assert res_stop.state == StabilizerState.STOP
    assert res_stop.stable_command == "STOP"
    assert res_stop.emitted_command == "STOP"


def test_stabilizer_stop_followed_by_normal_gesture():
    """Verify system resumes normal window building after STOP bypass."""
    stabilizer = TemporalStabilizer(window_size=5, vote_threshold=0.70, min_frames_required=3)
    rec_stop = make_result("STOP")
    rec_index = make_result("INDEX")

    # 1. Trigger STOP bypass
    stabilizer.update(rec_stop, is_stop_armed=True)

    # 2. Next frame is normal gesture -> window starts building, not immediately stable
    res1 = stabilizer.update(rec_index, is_stop_armed=False)
    assert res1.state == StabilizerState.STABILIZING
    assert res1.stable_command is None

    # 3. Fill remaining required frames -> INDEX becomes stable
    stabilizer.update(rec_index)
    res_final = stabilizer.update(rec_index)
    assert res_final.state == StabilizerState.STABLE
    assert res_final.stable_command == "INDEX"
    assert res_final.emitted_command == "INDEX"


def test_stabilizer_custom_configuration():
    """Verify stabilizer respects custom window_size and vote_threshold settings."""
    cfg = AppConfig()
    cfg.stabilization.window_size = 5
    cfg.stabilization.vote_threshold = 0.80
    cfg.stabilization.min_frames_required = 4

    stabilizer = TemporalStabilizer(config=cfg)
    assert stabilizer.window_size == 5
    assert stabilizer.vote_threshold == 0.80

    rec_index = make_result("INDEX")
    rec_mid = make_result("MIDDLE")

    # 3 INDEX + 1 MIDDLE = 3/4 = 75% < 80% -> STABILIZING
    for _ in range(3):
        stabilizer.update(rec_index)
    res = stabilizer.update(rec_mid)

    assert res.state == StabilizerState.STABILIZING
    assert res.stable_command is None
