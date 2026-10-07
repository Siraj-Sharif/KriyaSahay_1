"""
tests/test_call_ok_hardware_command.py
───────────────────────────────────────
Regression suite for the CALL / OK hardware-command fix.

Background
----------
CALL and OK are part of the locked 13-gesture hardware taxonomy
(``neurogrip.hagrid.taxonomy.NEUROGRIP_TAXONOMY``) and were recognised and displayed
by the UI, but they never reached the ESP32: the ``CommandValidator`` resolved command
candidates exclusively through the legacy *dataset* vocabulary
(``NeuroGripCommand``), which does not contain CALL or OK, so validation failed and no
serial frame was ever emitted.

These tests lock down the complete hardware path for every taxonomy command:

    recognition -> temporal stabilization -> command validation ->
    NG1 protocol encoding -> serial transport write

They also guard the two vocabularies against accidental merging.
"""
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.commands.definitions import (
    NeuroGripCommand,
    command_name,
    resolve_command,
)
from neurogrip.commands.validator import CommandValidator, PipelineState
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.communication.protocol import ProtocolEncoder
from neurogrip.communication.serial_port import RealSerialInterface
from neurogrip.config.settings import AppConfig, SerialConfig
from neurogrip.hagrid.taxonomy import (
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    CanonicalCommand,
    canonical_from_string,
)
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)
from neurogrip.recognition.base import RecognitionResult
from neurogrip.stabilization.temporal import StabilizerState, TemporalStabilizer

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _make_landmarks() -> HandLandmarks:
    """Synthetic single hand (geometrically irrelevant: the recognizer is stubbed)."""
    return HandLandmarks(
        landmarks=[NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)],
        handedness=Handedness.RIGHT,
        score=0.99,
    )


class _StubRecognizer:
    """Minimal GestureRecognizer stand-in that always predicts one fixed label."""

    def __init__(self, label: str) -> None:
        self.label = label

    @property
    def is_ready(self) -> bool:
        return True

    def predict(self, features, hand_landmarks=None, is_stop_armed=False, raw_frame=None) -> RecognitionResult:
        return RecognitionResult(
            label=self.label,
            confidence=1.0,
            all_scores={self.label: 1.0},
            recognizer_type="stub",
            reason="TEST_STUB",
            model_used="RULE_BASED",
            raw_prediction=self.label,
        )


def _build_pipeline(recognizer, serial) -> NeuroGripPipeline:
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True
    pipeline = NeuroGripPipeline(config=config, serial=serial, recognizer=recognizer)
    assert pipeline.initialize() is True
    pipeline.detector.detect = MagicMock(  # type: ignore[method-assign]
        return_value=DetectionResult(hands=[_make_landmarks()], num_hands=1)
    )
    return pipeline


# ─────────────────────────────────────────────────────────
# A. Vocabulary integrity (locked taxonomy + dataset vocabulary)
# ─────────────────────────────────────────────────────────

def test_locked_taxonomy_is_unchanged_and_contains_call_and_ok():
    """The locked hardware taxonomy must remain exactly these 13 gestures."""
    assert NEUROGRIP_TAXONOMY == [
        "CALL",
        "CLOSED_FIST",
        "FOUR_FINGERS",
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
    ]
    assert len(NEUROGRIP_TAXONOMY) == 13
    assert "CALL" in NEUROGRIP_TAXONOMY
    assert "OK" in NEUROGRIP_TAXONOMY


def test_canonical_command_mirrors_taxonomy_exactly():
    """CanonicalCommand is derived from NEUROGRIP_TAXONOMY and can never drift from it."""
    assert [member.value for member in CanonicalCommand] == NEUROGRIP_TAXONOMY
    assert CanonicalCommand.CALL.value == "CALL"
    assert CanonicalCommand.OK.value == "OK"


def test_canonical_lookup_rejects_non_taxonomy_names():
    assert canonical_from_string(" call ") is CanonicalCommand.CALL
    assert canonical_from_string("ok") is CanonicalCommand.OK
    assert canonical_from_string("REST") is None
    assert canonical_from_string("NO_COMMAND") is None
    assert canonical_from_string("UNKNOWN") is None
    assert canonical_from_string("") is None
    assert canonical_from_string(None) is None


def test_legacy_dataset_vocabulary_is_not_modified():
    """NeuroGripCommand is the dataset/training vocabulary and must stay 13 legacy labels."""
    assert len(NeuroGripCommand) == 13
    assert {cmd.value for cmd in NeuroGripCommand} == {
        "INDEX",
        "MIDDLE",
        "RING",
        "PINKY",
        "THUMB_ONLY",
        "TWO_FINGER",
        "THREE_FINGER",
        "INDEX_PINKY",
        "FOUR_FINGERS",
        "CLOSE",
        "GRAB",
        "REST",
        "STOP",
    }
    # CALL/OK must NOT be injected into the dataset vocabulary.
    assert NeuroGripCommand.from_string("CALL") is None
    assert NeuroGripCommand.from_string("OK") is None


def test_frontend_gesture_list_matches_backend_taxonomy():
    """Guards the frontend/backend gesture contract (src/services/types.ts GESTURES)."""
    types_ts = REPO_ROOT / "src" / "services" / "types.ts"
    if not types_ts.exists():  # pragma: no cover - frontend not present
        pytest.skip("frontend types.ts not available")

    match = re.search(r"export const GESTURES = \[(.*?)\] as const;", types_ts.read_text(encoding="utf-8"), re.S)
    assert match is not None, "GESTURES array not found in src/services/types.ts"
    frontend = re.findall(r'"([A-Z_]+)"', match.group(1))
    assert frontend == NEUROGRIP_TAXONOMY


# ─────────────────────────────────────────────────────────
# B. Shared resolution helpers
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["CALL", "OK", "GRIP", "CLOSED_FIST", "INDEX_FINGER", "STOP"])
def test_command_name_extracts_wire_name_from_both_vocabularies(name):
    """
    ``command_name`` must return the vocabulary value (never 'ClassName.MEMBER').

    Legacy names resolve to their legacy value (GRIP -> GRAB etc.); the invariant that
    matters for hardware is that the name still encodes to a valid NG1 frame.
    """
    resolved = resolve_command(name)
    assert resolved is not None
    assert command_name(resolved) == resolved.value
    assert command_name(name) == name
    assert command_name(None) == ""
    assert ProtocolEncoder.encode_command(command_name(resolved)) is not None


@pytest.mark.parametrize(
    "name,expected_wire",
    [
        ("CALL", b"NG1|CALL\n"),
        ("OK", b"NG1|OK\n"),
        ("GRIP", b"NG1|GRIP\n"),
        ("CLOSED_FIST", b"NG1|CLOSED_FIST\n"),
        ("INDEX_FINGER", b"NG1|INDEX_FINGER\n"),
    ],
)
def test_legacy_aliases_still_encode_to_their_historical_wire_frames(name, expected_wire):
    """Regression guard: the legacy alias round-trip must keep producing the same wire bytes."""
    resolved = resolve_command(name)
    assert resolved is not None
    assert ProtocolEncoder.encode_command(resolved) == expected_wire
    assert ProtocolEncoder.encode_command(name) == expected_wire


# ─────────────────────────────────────────────────────────
# C. Validation accepts CALL / OK
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", NEUROGRIP_TAXONOMY)
def test_every_taxonomy_command_passes_validation(command):
    """Every recognised gesture must be a valid, transmittable hardware command."""
    validator = CommandValidator()
    result = validator.validate(command)

    assert result.is_valid is True, f"{command} was rejected by the CommandValidator"
    assert result.command is not None
    assert result.was_deduplicated is False
    # The resolved command must encode back to the gesture it was recognised as.
    # (Legacy aliases keep their historical wire frames, e.g. GRIP -> NG1|GRIP.)
    assert ProtocolEncoder.encode_command(command) == f"NG1|{command}\n".encode("ascii")
    assert ProtocolEncoder.encode_command(result.command) is not None


@pytest.mark.parametrize("command", ["CALL", "OK"])
def test_call_and_ok_resolve_to_canonical_vocabulary(command):
    """CALL/OK resolve through the canonical vocabulary, not the dataset vocabulary."""
    validator = CommandValidator()
    result = validator.validate(command)

    assert isinstance(result.command, CanonicalCommand)
    assert not isinstance(result.command, NeuroGripCommand)
    assert result.command.value == command


def test_call_and_ok_are_case_insensitive_and_trimmed():
    validator = CommandValidator()
    assert validator.validate(" call ").is_valid is True
    assert validator.validate("ok").is_valid is True


def test_call_and_ok_are_deduplicated_like_normal_commands():
    """CALL/OK follow the standard de-duplication rule (only STOP is exempt)."""
    validator = CommandValidator()

    first = validator.validate("CALL")
    assert first.is_valid is True
    second = validator.validate("CALL")
    assert second.is_valid is False
    assert second.was_deduplicated is True

    # A different command still passes, and re-emitting CALL after it works again.
    assert validator.validate("OK").is_valid is True
    assert validator.validate("CALL").is_valid is True


def test_invalid_candidates_are_still_rejected():
    validator = CommandValidator()
    for candidate in ("NO_COMMAND", "UNKNOWN", "INVALID", NO_COMMAND, "REST", "one", "peace", "", None):
        assert validator.validate(candidate).is_valid is False


def test_safety_states_still_block_call_and_ok():
    """NO_HAND / AMBIGUOUS / ERROR must still suppress CALL and OK."""
    for state in (PipelineState.NO_HAND, PipelineState.AMBIGUOUS, PipelineState.ERROR):
        validator = CommandValidator()
        result = validator.validate("CALL", pipeline_state=state)
        assert result.is_valid is False
        assert result.command is None


# ─────────────────────────────────────────────────────────
# D. Protocol encoding produces the exact NG1 frame
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", NEUROGRIP_TAXONOMY)
def test_every_taxonomy_command_encodes_to_exact_ng1_frame(command):
    expected = f"NG1|{command}\n".encode("ascii")
    assert ProtocolEncoder.encode_command(command) == expected
    assert ProtocolEncoder.encode_command_string(command) == f"NG1|{command}\n"
    assert len(expected) <= 32


@pytest.mark.parametrize(
    "command,expected",
    [
        ("CALL", b"NG1|CALL\n"),
        ("OK", b"NG1|OK\n"),
        (CanonicalCommand.CALL, b"NG1|CALL\n"),
        (CanonicalCommand.OK, b"NG1|OK\n"),
    ],
)
def test_call_and_ok_encode_to_expected_bytes(command, expected):
    assert ProtocolEncoder.encode_command(command) == expected
    assert len(expected) <= ProtocolEncoder.MAX_FRAME_LENGTH


# ─────────────────────────────────────────────────────────
# E. Serial transport actually writes the CALL / OK frames
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,expected", [("CALL", b"NG1|CALL\n"), ("OK", b"NG1|OK\n")])
def test_call_and_ok_are_written_to_serial_port(command, expected, monkeypatch):
    """RealSerialInterface must put the exact bytes on the wire (pyserial mocked)."""
    mock_port = MagicMock()
    mock_port.is_open = True
    monkeypatch.setattr("serial.Serial", MagicMock(return_value=mock_port))

    transport = RealSerialInterface(config=SerialConfig(port="COM_TEST", baud_rate=115200))
    assert transport.connect() is True
    assert transport.send_command(command) is True

    mock_port.write.assert_called_once_with(expected)
    mock_port.flush.assert_called_once()


@pytest.mark.parametrize("command,expected", [("CALL", b"NG1|CALL\n"), ("OK", b"NG1|OK\n")])
def test_call_and_ok_are_recorded_by_mock_serial(command, expected):
    mock_serial = MockSerialInterface()
    mock_serial.connect()

    assert mock_serial.send_command(command) is True
    assert mock_serial.sent_frames == [f"NG1|{command}\n"]
    assert mock_serial.sent_bytes == [expected]
    assert command_name(mock_serial.sent_commands[0]) == command


# ─────────────────────────────────────────────────────────
# F. End-to-end: recogniser output -> serial bytes
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("gesture,expected", [("CALL", b"NG1|CALL\n"), ("OK", b"NG1|OK\n")])
def test_call_and_ok_reach_the_serial_interface_through_the_pipeline(gesture, expected):
    """
    Full production path: recognition -> temporal stabilizer -> CommandValidator ->
    ProtocolEncoder -> SerialInterface.write().  This is the regression that proves
    CALL/OK are no longer recognised-but-untransmittable.
    """
    serial = MockSerialInterface()
    pipeline = _build_pipeline(_StubRecognizer(gesture), serial)

    try:
        # The stabilizer needs min_frames_required (5) consistent frames before it
        # reports a stable command, and emits exactly one command at that moment.
        emitting_frame = None
        for _ in range(8):
            _, _, viz_state = pipeline.process_frame()
            if viz_state is not None and viz_state.is_tx_permitted:
                emitting_frame = viz_state

        assert emitting_frame is not None, "command was never permitted for transmission"
        assert emitting_frame.command == gesture
        assert emitting_frame.is_stabilized is True
        assert emitting_frame.last_tx == f"NG1|{gesture}", "UI telemetry disagrees with the transmitted frame"

        assert expected in serial.sent_bytes, f"{expected!r} never reached the serial interface"
        assert serial.sent_frames.count(f"NG1|{gesture}\n") == 1, "command must be emitted exactly once"
        assert serial.sent_bytes == [expected]
    finally:
        pipeline.shutdown()


def test_no_command_never_writes_to_serial():
    """Safety regression: an unrecognised gesture must produce zero serial writes."""
    serial = MockSerialInterface()
    pipeline = _build_pipeline(_StubRecognizer("NO_COMMAND"), serial)

    try:
        for _ in range(8):
            _, _, viz_state = pipeline.process_frame()

        assert viz_state.command == "NO_COMMAND"
        assert serial.sent_bytes == []
        assert viz_state.is_tx_permitted is False
    finally:
        pipeline.shutdown()


def test_stop_safety_semantics_are_preserved():
    """STOP still bypasses the majority-vote window and is never de-duplicated."""
    config = AppConfig.default()
    stabilizer = TemporalStabilizer(config=config)
    result = stabilizer.update(RecognitionResult(label="STOP", confidence=1.0, model_used="RULE_BASED"))

    assert result.state == StabilizerState.STOP
    assert result.emitted_command == "STOP"

    validator = CommandValidator()
    assert validator.validate("STOP").is_valid is True
    assert validator.validate("STOP").is_valid is True  # never suppressed
