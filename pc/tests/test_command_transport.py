"""
tests/test_command_transport.py
────────────────────────────────
Unit tests for CommandTransport and DisplayConsoleTransport boundary layer.
Tests protocol frame formatting (NG1|<COMMAND>), idle handling, and REST rejection.
"""
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.transport import DisplayConsoleTransport


def test_display_console_transport_format_frame():
    """Verify protocol frame formatting adheres to NG1|<COMMAND>."""
    frame_index = DisplayConsoleTransport.format_protocol_frame(NeuroGripCommand.INDEX)
    assert frame_index == "NG1|INDEX_FINGER"

    frame_close = DisplayConsoleTransport.format_protocol_frame("CLOSED_FIST")
    assert frame_close == "NG1|CLOSED_FIST"

    frame_stop = DisplayConsoleTransport.format_protocol_frame(NeuroGripCommand.STOP)
    assert frame_stop == "NG1|STOP"


def test_display_console_transport_rest_is_idle():
    """Verify REST returns None for protocol frame and is treated as NO-COMMAND / IDLE."""
    frame_rest = DisplayConsoleTransport.format_protocol_frame(NeuroGripCommand.REST)
    assert frame_rest is None

    transport = DisplayConsoleTransport()
    res = transport.send_command(NeuroGripCommand.REST)
    assert res is False
    assert transport.latest_command is None
    assert transport.latest_frame is None


def test_display_console_transport_emission_count():
    """Verify active commands increment emission count and update latest_frame."""
    transport = DisplayConsoleTransport()
    assert transport.emission_count == 0

    tx_ok = transport.send_command(NeuroGripCommand.THREE_FINGER)
    assert tx_ok is True
    assert transport.emission_count == 1
    assert transport.latest_command == NeuroGripCommand.THREE_FINGER
    assert transport.latest_frame == "NG1|THREE_FINGERS"

    transport.send_idle()
    assert transport.latest_command is None
    assert transport.latest_frame is None
    assert transport.emission_count == 1
