"""
tests/test_serial_protocol.py
──────────────────────────────
Unit tests for serial communication protocol formatting, MockSerialInterface,
and RealSerialInterface error handling.
Does NOT require physical ESP32 hardware to run.
"""
from unittest.mock import MagicMock, patch
import pytest
import serial

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.base import SerialInterface, SerialState
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.communication.serial_port import RealSerialInterface
from neurogrip.config.settings import SerialConfig
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY


# ─────────────────────────────────────────────────────────
# Protocol Formatting Tests
# ─────────────────────────────────────────────────────────

def test_protocol_formatting_all_14_canonical_commands():
    """Verify all 14 locked canonical commands format correctly into 'NG1|<CMD>\\n'."""
    for cmd in NEUROGRIP_TAXONOMY:
        frame = SerialInterface.format_frame(cmd)
        assert frame is not None
        assert frame == f"NG1|{cmd}\n"
        assert frame.startswith("NG1|")
        assert frame.endswith("\n")
        assert len(frame.encode("ascii")) <= SerialInterface.MAX_FRAME_LENGTH


def test_protocol_formatting_case_insensitive_string():
    """Verify string lookups and legacy aliases format cleanly to canonical uppercase frames."""
    assert SerialInterface.format_frame("index") == "NG1|INDEX_FINGER\n"
    assert SerialInterface.format_frame("  thumb_only  ") == "NG1|THUMBS_UP\n"
    assert SerialInterface.format_frame("STOP") == "NG1|STOP\n"
    assert SerialInterface.format_frame("closed_fist") == "NG1|CLOSED_FIST\n"


def test_protocol_formatting_invalid_commands_rejected():
    """Verify invalid, empty, or wrong-type commands return None."""
    assert SerialInterface.format_frame("INVALID_COMMAND") is None
    assert SerialInterface.format_frame("one") is None
    assert SerialInterface.format_frame("peace") is None
    assert SerialInterface.format_frame("") is None
    assert SerialInterface.format_frame(None) is None  # type: ignore
    assert SerialInterface.format_frame(9999) is None  # type: ignore


def test_protocol_frame_max_length_bound():
    """Verify all 14 formatted command frames stay strictly within the 32-byte bound."""
    for cmd in NEUROGRIP_TAXONOMY:
        frame = SerialInterface.format_frame(cmd)
        assert frame is not None
        assert len(frame.encode("ascii")) <= 32


# ─────────────────────────────────────────────────────────
# MockSerialInterface Tests
# ─────────────────────────────────────────────────────────

def test_mock_serial_connection_flow():
    """Test MockSerialInterface connection, disconnection, and state."""
    mock_serial = MockSerialInterface()
    assert mock_serial.is_connected is False
    assert mock_serial.state == SerialState.DISCONNECTED

    # Connect
    assert mock_serial.connect() is True
    assert mock_serial.is_connected is True
    assert mock_serial.state == SerialState.MOCK

    # Disconnect
    mock_serial.disconnect()
    assert mock_serial.is_connected is False
    assert mock_serial.state == SerialState.DISCONNECTED


def test_mock_serial_command_transmission():
    """Test transmitting valid and invalid commands over MockSerialInterface."""
    mock_serial = MockSerialInterface()
    mock_serial.connect()

    # Transmit valid commands
    assert mock_serial.send_command("INDEX_FINGER") is True
    assert mock_serial.send_command("CLOSED_FIST") is True
    assert mock_serial.send_command(NeuroGripCommand.STOP) is True

    assert len(mock_serial.sent_frames) == 3
    assert mock_serial.sent_frames == ["NG1|INDEX_FINGER\n", "NG1|CLOSED_FIST\n", "NG1|STOP\n"]

    # Transmit invalid command -> rejected
    assert mock_serial.send_command("MALFORMED_CMD") is False
    assert len(mock_serial.sent_frames) == 3  # Not added

    # Clear history
    mock_serial.clear()
    assert len(mock_serial.sent_frames) == 0


def test_mock_serial_disconnected_transmission():
    """Verify transmitting while disconnected returns False."""
    mock_serial = MockSerialInterface()
    assert mock_serial.send_command("INDEX_FINGER") is False
    assert len(mock_serial.sent_frames) == 0


# ─────────────────────────────────────────────────────────
# RealSerialInterface Tests (Mocked PySerial)
# ─────────────────────────────────────────────────────────

@patch("serial.Serial")
def test_real_serial_connect_and_transmit(mock_pyserial_cls):
    """Test RealSerialInterface connection and frame writing with mocked PySerial."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_pyserial_cls.return_value = mock_port

    cfg = SerialConfig(port="COM4", baud_rate=115200, timeout_s=1.0)
    real_serial = RealSerialInterface(config=cfg)

    # Connect
    assert real_serial.connect() is True
    assert real_serial.is_connected is True
    assert real_serial.state == SerialState.CONNECTED
    mock_pyserial_cls.assert_called_once_with(port="COM4", baudrate=115200, timeout=1.0, write_timeout=1.0)

    # Transmit command
    assert real_serial.send_command("CLOSED_FIST") is True
    mock_port.write.assert_called_once_with(b"NG1|CLOSED_FIST\n")
    mock_port.flush.assert_called_once()

    # Disconnect
    real_serial.disconnect()
    assert real_serial.is_connected is False
    assert real_serial.state == SerialState.DISCONNECTED
    mock_port.close.assert_called_once()


@patch("serial.Serial", side_effect=serial.SerialException("Port not found"))
def test_real_serial_connection_failure(mock_pyserial_cls):
    """Verify RealSerialInterface catches SerialException on connect and enters ERROR state."""
    cfg = SerialConfig(port="COM99", baud_rate=115200)
    real_serial = RealSerialInterface(config=cfg)

    assert real_serial.connect() is False
    assert real_serial.is_connected is False
    assert real_serial.state == SerialState.ERROR


@patch("serial.Serial")
def test_real_serial_write_failure(mock_pyserial_cls):
    """Verify RealSerialInterface catches SerialException on write and enters ERROR state."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_port.write.side_effect = serial.SerialException("Write timeout")
    mock_pyserial_cls.return_value = mock_port

    real_serial = RealSerialInterface()
    real_serial.connect()

    assert real_serial.send_command("STOP") is False
    assert real_serial.state == SerialState.ERROR
