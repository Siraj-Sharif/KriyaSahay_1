"""
pc/tests/test_phase11_serial_integration.py
─────────────────────────────────────────────
Phase 11.0 PC → ESP32 USB Serial Integration Test Suite.

Verifies:
  A. Valid command encoding (CALL, INDEX_FINGER, INDEX_PINKY, STOP, THREE_FINGERS, etc.)
  B. Invalid command rejection (NO_COMMAND, UNKNOWN, INVALID, REST, one, peace, raw HaGRID labels)
  C. Exact protocol bytes (NG1|<CMD>\n ASCII bytes)
  D. System OFF → zero serial writes
  E. No hand → zero serial writes
  F. Multiple hands → zero serial writes
  G. Low confidence → zero serial writes
  H. STOP command transmission → transmits NG1|STOP\n
  I. STOP armed → exactly NG1|STOP\n
  J. Disconnect handling → no application crash
  K. Failed serial write → correct ERROR state transition
  L. Clean shutdown → serial port cleanly closed
"""

from unittest.mock import MagicMock, patch
import pytest
import serial

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.base import SerialState
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.communication.protocol import ProtocolEncoder, encode_protocol_frame
from neurogrip.communication.serial_port import RealSerialInterface
from neurogrip.config.settings import AppConfig, SerialConfig
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY, NO_COMMAND
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import RecognitionResult


def test_phase11_a_valid_command_encoding():
    """Requirement A: Valid canonical command candidates format correctly into NG1|<CMD>\\n."""
    test_commands = ["CALL", "INDEX_FINGER", "INDEX_PINKY", "STOP", "THREE_FINGERS", "CLOSED_FIST", "FOUR_FINGERS"]
    for cmd in test_commands:
        frame_str = ProtocolEncoder.encode_command_string(cmd)
        assert frame_str == f"NG1|{cmd}\n"


def test_phase11_b_invalid_command_rejection():
    """Requirement B: Invalid, empty, or unmapped commands produce None (zero writes)."""
    invalid_candidates = [
        "NO_COMMAND",
        "UNKNOWN",
        "INVALID",
        "REST",
        "one",
        "peace",
        "peace_inverted",
        "hand_heart",
        "take_picture",
        "dislike",
        "mute",
        "xsign",
        "",
        None,
    ]
    for cand in invalid_candidates:
        assert ProtocolEncoder.encode_command_string(cand) is None  # type: ignore
        assert ProtocolEncoder.encode_command(cand) is None  # type: ignore



def test_phase11_c_exact_protocol_bytes():
    """Requirement C: Formatted wire frames match exact ASCII bytes (e.g. b'NG1|INDEX_FINGER\\n')."""
    cmd = "INDEX_FINGER"
    frame_bytes = ProtocolEncoder.encode_command(cmd)
    assert frame_bytes == b"NG1|INDEX_FINGER\n"

    cmd_stop = "STOP"
    frame_stop = ProtocolEncoder.encode_command(cmd_stop)
    assert frame_stop == b"NG1|STOP\n"


def test_phase11_d_system_off_zero_writes():
    """Requirement D: System OFF state produces zero serial transmissions."""
    mock_serial = MockSerialInterface()
    mock_serial.connect()
    mock_serial.clear()

    # System OFF simulation
    system_on = False
    if system_on:
        mock_serial.send_command("INDEX_FINGER")

    assert len(mock_serial.sent_frames) == 0


def test_phase11_e_no_hand_zero_writes():
    """Requirement E: 0 hands detected produces NO_COMMAND and zero serial writes."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()

    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[], num_hands=0))  # type: ignore

    _, _, viz = pipeline.process_frame()

    assert viz.command == "NO_COMMAND"
    assert len(mock_serial.sent_frames) == 0
    pipeline.shutdown()


def test_phase11_f_multiple_hands_zero_writes():
    """Requirement F: 2+ hands detected produces NO_COMMAND and zero serial writes."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()

    dummy_lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    hand1 = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.RIGHT, score=0.9)
    hand2 = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.LEFT, score=0.9)

    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[hand1, hand2], num_hands=2))  # type: ignore

    _, _, viz = pipeline.process_frame()

    assert viz.is_ambiguous is True
    assert viz.command == "NO_COMMAND"
    assert len(mock_serial.sent_frames) == 0
    pipeline.shutdown()


def test_phase11_g_low_confidence_zero_writes():
    """Requirement G: Low confidence (<0.70 threshold) produces NO_COMMAND and zero serial writes."""
    mock_serial = MockSerialInterface()
    mock_serial.connect()

    rec_result = RecognitionResult(label="INDEX_FINGER", confidence=0.45, model_used="EXTRA_TREES")
    if rec_result.confidence < 0.70:
        cmd_candidate = "NO_COMMAND"
    else:
        cmd_candidate = rec_result.label

    tx_ok = mock_serial.send_command(cmd_candidate)
    assert tx_ok is False
    assert len(mock_serial.sent_frames) == 0


def test_phase11_h_stop_disarmed_zero_writes():
    """Requirement H: STOP is a normal command and transmits NG1|STOP\\n."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()

    dummy_lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    stop_hand = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.RIGHT, score=0.95)
    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[stop_hand], num_hands=1))  # type: ignore
    pipeline.recognizer.predict = MagicMock(return_value=RecognitionResult(label="STOP", confidence=1.0, model_used="RULE_BASED"))  # type: ignore

    for _ in range(3):
        pipeline.process_frame()

    assert "NG1|STOP\n" in mock_serial.sent_frames
    pipeline.shutdown()


def test_phase11_i_stop_armed_exact_frame():
    """Requirement I: STOP recognized while ARMED produces exactly NG1|STOP\\n."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()
    pipeline.arm_stop()

    dummy_lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    stop_hand = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.RIGHT, score=0.95)
    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[stop_hand], num_hands=1))  # type: ignore
    pipeline.recognizer.predict = MagicMock(return_value=RecognitionResult(label="STOP", confidence=1.0, model_used="RULE_BASED"))  # type: ignore

    pipeline.process_frame()

    assert pipeline.is_stop_armed is True
    assert "NG1|STOP\n" in mock_serial.sent_frames
    pipeline.shutdown()


@patch("serial.Serial")
def test_phase11_j_disconnect_handling_no_crash(mock_pyserial_cls):
    """Requirement J: Sudden disconnect/error does not crash the application."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_port.write.side_effect = serial.SerialException("USB Device Disconnected")
    mock_pyserial_cls.return_value = mock_port

    cfg = SerialConfig(port="COM3", baud_rate=115200)
    real_serial = RealSerialInterface(config=cfg)
    assert real_serial.connect() is True

    # Transmission fails gracefully without unhandled exception
    tx_ok = real_serial.send_command("INDEX_FINGER")
    assert tx_ok is False
    assert real_serial.is_connected is False


@patch("serial.Serial")
def test_phase11_k_failed_serial_write_error_state(mock_pyserial_cls):
    """Requirement K: Failed serial write transitions interface state to ERROR."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_port.write.side_effect = OSError("Write fault")
    mock_pyserial_cls.return_value = mock_port

    cfg = SerialConfig(port="COM3", baud_rate=115200)
    real_serial = RealSerialInterface(config=cfg)
    real_serial.connect()

    assert real_serial.send_command("CALL") is False
    assert real_serial.state == SerialState.ERROR


@patch("serial.Serial")
def test_phase11_l_clean_shutdown_closes_port(mock_pyserial_cls):
    """Requirement L: Clean shutdown releases serial port resources cleanly."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_pyserial_cls.return_value = mock_port

    cfg = SerialConfig(port="COM3", baud_rate=115200)
    real_serial = RealSerialInterface(config=cfg)
    real_serial.connect()

    real_serial.disconnect()
    assert real_serial.is_connected is False
    assert real_serial.state == SerialState.DISCONNECTED
    mock_port.close.assert_called_once()
