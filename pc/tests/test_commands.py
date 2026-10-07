"""
tests/test_commands.py
───────────────────────
Unit tests for NeuroGrip command vocabulary definitions and command validator.
"""
import pytest

from neurogrip.commands.definitions import (
    COMMAND_METADATA_MAP,
    CommandCategory,
    CommandMetadata,
    NeuroGripCommand,
)
from neurogrip.commands.validator import CommandValidator, PipelineState, ValidationResult


# ─────────────────────────────────────────────────────────
# Vocabulary & Enum Tests
# ─────────────────────────────────────────────────────────

def test_neurogrip_command_count():
    """Verify exactly 13 commands are defined in the locked vocabulary."""
    assert len(NeuroGripCommand) == 13


def test_neurogrip_command_members():
    """Verify all 13 locked commands exist with expected string values."""
    expected = {
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
    actual = {cmd.value for cmd in NeuroGripCommand}
    assert actual == expected


def test_neurogrip_command_from_string():
    """Test lookup of commands from string representations."""
    assert NeuroGripCommand.from_string("INDEX") == NeuroGripCommand.INDEX
    assert NeuroGripCommand.from_string("index") == NeuroGripCommand.INDEX
    assert NeuroGripCommand.from_string("  thumb_only  ") == NeuroGripCommand.THUMB_ONLY
    assert NeuroGripCommand.from_string("THREE_FINGER") == NeuroGripCommand.THREE_FINGER
    assert NeuroGripCommand.from_string("INVALID_CMD") is None
    assert NeuroGripCommand.from_string("") is None


def test_command_metadata_completeness():
    """Verify every command has a corresponding CommandMetadata entry."""
    for cmd in NeuroGripCommand:
        meta = cmd.metadata
        assert isinstance(meta, CommandMetadata)
        assert meta.canonical_name == cmd.value
        assert isinstance(meta.category, CommandCategory)
        assert isinstance(meta.is_safety_critical, bool)
        assert len(meta.description) > 0


def test_command_specific_categories_and_flags():
    """Verify specific category assignments and safety flags for key commands."""
    # STOP is safety-critical and in SAFETY category
    stop_meta = NeuroGripCommand.STOP.metadata
    assert stop_meta.category == CommandCategory.SAFETY
    assert stop_meta.is_safety_critical is True

    # GRAB is an action command
    grab_meta = NeuroGripCommand.GRAB.metadata
    assert grab_meta.category == CommandCategory.ACTION
    assert grab_meta.is_safety_critical is False

    # REST is system state
    rest_meta = NeuroGripCommand.REST.metadata
    assert rest_meta.category == CommandCategory.SYSTEM_STATE
    assert rest_meta.is_safety_critical is False

    # INDEX is finger config
    index_meta = NeuroGripCommand.INDEX.metadata
    assert index_meta.category == CommandCategory.FINGER_CONFIG
    assert index_meta.is_safety_critical is False


# ─────────────────────────────────────────────────────────
# Command Validator Tests
# ─────────────────────────────────────────────────────────

def test_validator_valid_commands():
    """Test validation of valid command candidates."""
    validator = CommandValidator()
    result = validator.validate("INDEX")
    assert result.is_valid is True
    assert result.command == NeuroGripCommand.INDEX
    assert result.was_deduplicated is False

    result2 = validator.validate(NeuroGripCommand.MIDDLE)
    assert result2.is_valid is True
    assert result2.command == NeuroGripCommand.MIDDLE


def test_validator_invalid_commands():
    """Test rejection of invalid, unknown, or None command candidates."""
    validator = CommandValidator()

    # Invalid string command
    res1 = validator.validate("UNKNOWN_GESTURE")
    assert res1.is_valid is False
    assert res1.command is None

    # None candidate
    res2 = validator.validate(None)
    assert res2.is_valid is False
    assert res2.command is None

    # Invalid type
    res3 = validator.validate(12345)  # type: ignore
    assert res3.is_valid is False
    assert res3.command is None


def test_validator_pipeline_state_blocking():
    """Test that NO_HAND, AMBIGUOUS, or ERROR states block command emission."""
    validator = CommandValidator()

    res_no_hand = validator.validate("INDEX", pipeline_state=PipelineState.NO_HAND)
    assert res_no_hand.is_valid is False
    assert res_no_hand.command is None

    res_ambiguous = validator.validate("INDEX", pipeline_state=PipelineState.AMBIGUOUS)
    assert res_ambiguous.is_valid is False

    res_error = validator.validate("INDEX", pipeline_state=PipelineState.ERROR)
    assert res_error.is_valid is False


def test_validator_deduplication():
    """Test duplicate command suppression and exception for STOP."""
    validator = CommandValidator()

    # Emit INDEX first time -> Valid
    r1 = validator.validate("INDEX")
    assert r1.is_valid is True

    # Emit INDEX second time -> Suppressed by de-duplication
    r2 = validator.validate("INDEX")
    assert r2.is_valid is False
    assert r2.was_deduplicated is True
    assert r2.command == NeuroGripCommand.INDEX

    # Emit MIDDLE -> Valid
    r3 = validator.validate("MIDDLE")
    assert r3.is_valid is True

    # Reset validator -> emitting MIDDLE again works
    validator.reset()
    r4 = validator.validate("MIDDLE")
    assert r4.is_valid is True


def test_validator_stop_never_deduplicated():
    """Verify STOP command is never suppressed by de-duplication."""
    validator = CommandValidator()

    r1 = validator.validate("STOP")
    assert r1.is_valid is True
    assert r1.command == NeuroGripCommand.STOP

    # Consecutive STOP -> MUST still be valid and not suppressed!
    r2 = validator.validate("STOP")
    assert r2.is_valid is True
    assert r2.was_deduplicated is False
    assert r2.command == NeuroGripCommand.STOP


def test_validator_stop_active_state_behavior():
    """Verify STOP command is validated and normal commands validate under STOP-as-normal-command architecture."""
    validator = CommandValidator()

    # Normal command validates normally (STOP has no latch/interlock)
    r_index = validator.validate("INDEX", pipeline_state=PipelineState.STOP_ACTIVE)
    assert r_index.is_valid is True
    assert r_index.command == NeuroGripCommand.INDEX

    # STOP command validates normally
    r_stop = validator.validate("STOP", pipeline_state=PipelineState.STOP_ACTIVE)
    assert r_stop.is_valid is True
    assert r_stop.command == NeuroGripCommand.STOP
