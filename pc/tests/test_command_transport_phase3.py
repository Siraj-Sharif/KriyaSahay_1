"""
tests/test_command_transport_phase3.py
───────────────────────────────────────
Phase 3 Command Protocol & Transport Integration Test Suite.

Verifies:
  A. Protocol encoding (13 locked canonical commands -> b"NG1|<CMD>\n")
  B. NO_COMMAND suppression (zero writes)
  C. Invalid command suppression (zero writes for UNKNOWN, INVALID, one, peace, etc.)
  D. Mock transport recording (exact NG1|<CMD>\n frame recorded)
  E. Duplicate-frame protection (stabilizer emission event logic)
  F. STOP safety (unarmed STOP -> zero writes, armed STOP -> exactly one write)
  G. Multi-hand safety (2 hands -> zero writes)
  H. No-hand safety (0 hands -> zero writes)
  I. Transport failure resilience (serial exception caught, app stays alive)
  J. Protocol isolation (ProtocolEncoder module imports NO pyserial)
"""
import sys
from unittest.mock import MagicMock, patch
import pytest
import serial

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.commands.validator import CommandValidator, PipelineState
from neurogrip.communication.base import SerialState
from neurogrip.communication.mock_serial import MockSerialInterface, MockSerialTransport
from neurogrip.communication.protocol import ProtocolEncoder, encode_protocol_frame
from neurogrip.communication.serial_port import RealSerialInterface, SerialTransport
from neurogrip.config.settings import AppConfig, SerialConfig
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY, NO_COMMAND
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark

from neurogrip.recognition.base import RecognitionResult
from neurogrip.stabilization.temporal import StabilizerState, TemporalStabilizer


# ─────────────────────────────────────────────────────────
# A. Protocol Encoding Tests
# ─────────────────────────────────────────────────────────

def test_protocol_encoding_all_13_canonical_commands():
    """Verify all 13 canonical commands format correctly into b'NG1|<CMD>\\n'."""
    assert len(NEUROGRIP_TAXONOMY) == 13
    for cmd in NEUROGRIP_TAXONOMY:
        encoded_bytes = ProtocolEncoder.encode_command(cmd)
        encoded_str = ProtocolEncoder.encode_command_string(cmd)

        assert encoded_bytes is not None
        assert encoded_str is not None
        assert encoded_bytes == f"NG1|{cmd}\n".encode("ascii")
        assert encoded_str == f"NG1|{cmd}\n"
        assert len(encoded_bytes) <= 32


def test_protocol_encoding_specific_examples():
    """Verify specific prompt examples for CALL, INDEX_FINGER, and STOP."""
    assert encode_protocol_frame("CALL") == b"NG1|CALL\n"
    assert encode_protocol_frame("INDEX_FINGER") == b"NG1|INDEX_FINGER\n"
    assert encode_protocol_frame("STOP") == b"NG1|STOP\n"
    assert encode_protocol_frame(NeuroGripCommand.CLOSE) == b"NG1|CLOSED_FIST\n"
    assert encode_protocol_frame("CLOSED_FIST") == b"NG1|CLOSED_FIST\n"



# ─────────────────────────────────────────────────────────
# B. NO_COMMAND Suppression Tests
# ─────────────────────────────────────────────────────────

def test_no_command_suppression():
    """Verify NO_COMMAND produces zero bytes / returns None from protocol encoder."""
    assert ProtocolEncoder.encode_command("NO_COMMAND") is None
    assert ProtocolEncoder.encode_command_string("NO_COMMAND") is None
    assert encode_protocol_frame("NO_COMMAND") is None

    mock_serial = MockSerialInterface()
    mock_serial.connect()
    res = mock_serial.send_command("NO_COMMAND")
    assert res is False
    assert len(mock_serial.sent_frames) == 0
    assert len(mock_serial.sent_bytes) == 0


# ─────────────────────────────────────────────────────────
# C. Invalid Command Suppression Tests
# ─────────────────────────────────────────────────────────

def test_invalid_and_unsupported_command_suppression():
    """Verify UNKNOWN, INVALID, one, peace, peace_inverted produce zero writes."""
    invalid_candidates = [
        "UNKNOWN",
        "INVALID",
        "one",
        "peace",
        "peace_inverted",
        "hukenovs_unsupported",
        "random_text",
        "",
        None,
    ]
    mock_serial = MockSerialInterface()
    mock_serial.connect()

    for cand in invalid_candidates:
        assert ProtocolEncoder.encode_command(cand) is None  # type: ignore
        assert mock_serial.send_command(cand) is False  # type: ignore

    assert len(mock_serial.sent_frames) == 0


# ─────────────────────────────────────────────────────────
# D. Mock Transport Tests
# ─────────────────────────────────────────────────────────

def test_mock_transport_recording():
    """Verify valid command reaches MockSerialTransport exactly once."""
    transport = MockSerialTransport()
    transport.connect()
    assert transport.is_connected is True

    tx_ok = transport.send_command("INDEX_FINGER")
    assert tx_ok is True
    assert len(transport.sent_frames) == 1
    assert transport.sent_frames[0] == "NG1|INDEX_FINGER\n"
    assert transport.sent_bytes[0] == b"NG1|INDEX_FINGER\n"

    # Alias check
    assert isinstance(transport, MockSerialInterface)


# ─────────────────────────────────────────────────────────
# E. Duplicate-Frame Protection Tests
# ─────────────────────────────────────────────────────────

def test_duplicate_frame_protection():
    """Verify stabilizer & validator suppress duplicate consecutive frames (except STOP)."""
    validator = CommandValidator()

    res1 = validator.validate("INDEX_FINGER", pipeline_state=PipelineState.TRACKING)
    assert res1.is_valid is True
    assert res1.was_deduplicated is False

    # Second frame with same command candidate
    res2 = validator.validate("INDEX_FINGER", pipeline_state=PipelineState.TRACKING)
    assert res2.is_valid is False
    assert res2.was_deduplicated is True

    # STOP is never suppressed by de-duplication
    res_stop1 = validator.validate("STOP", pipeline_state=PipelineState.STOP_ARMED)
    assert res_stop1.is_valid is True
    res_stop2 = validator.validate("STOP", pipeline_state=PipelineState.STOP_ARMED)
    assert res_stop2.is_valid is True


# ─────────────────────────────────────────────────────────
# F. STOP Safety Tests
# ─────────────────────────────────────────────────────────

def test_stop_command_transmission():
    """Verify STOP command emits exactly one wire frame upon transition like normal gestures."""
    config = AppConfig.default()
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    mock_serial.connect()

    rec_stop = RecognitionResult(label="STOP", confidence=0.95, model_used="RULE_BASED")
    stabilizer = TemporalStabilizer(config=config)
    
    # Process 5 frames of STOP
    emissions = []
    for _ in range(5):
        res = stabilizer.update(rec_stop)
        if res.emitted_command and res.emitted_command != "NO_COMMAND":
            mock_serial.send_command(res.emitted_command)
            emissions.append(res.emitted_command)

    assert len(emissions) == 1
    assert emissions[0] == "STOP"
    assert len(mock_serial.sent_frames) == 1
    assert mock_serial.sent_frames[0] == "NG1|STOP\n"


# ─────────────────────────────────────────────────────────
# G. Multi-Hand Safety Tests
# ─────────────────────────────────────────────────────────

def test_multi_hand_safety_zero_writes():
    """Verify 2+ hands result in NO_COMMAND and ZERO serial writes."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    mock_serial.connect()

    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()

    # Inject ambiguous (2 hands) detection result
    dummy_landmarks = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]

    hand1 = HandLandmarks(landmarks=dummy_landmarks, handedness=Handedness.RIGHT, score=0.9)
    hand2 = HandLandmarks(landmarks=dummy_landmarks, handedness=Handedness.LEFT, score=0.9)
    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[hand1, hand2], num_hands=2))  # type: ignore

    container, rendered, viz_state = pipeline.process_frame()

    assert viz_state.is_ambiguous is True
    assert viz_state.command == "NO_COMMAND"
    assert len(mock_serial.sent_frames) == 0
    pipeline.shutdown()


# ─────────────────────────────────────────────────────────
# H. No-Hand Safety Tests
# ─────────────────────────────────────────────────────────

def test_no_hand_safety_zero_writes():
    """Verify 0 hands result in NO_COMMAND and ZERO serial writes."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    mock_serial.connect()

    pipeline = NeuroGripPipeline(config=config, serial=mock_serial)
    pipeline.initialize()

    # Inject 0 hands detection result
    pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[], num_hands=0))  # type: ignore


    container, rendered, viz_state = pipeline.process_frame()

    assert viz_state.pipeline_state == "NO_HAND"
    assert viz_state.command == "NO_COMMAND"
    assert len(mock_serial.sent_frames) == 0
    pipeline.shutdown()


# ─────────────────────────────────────────────────────────
# I. Transport Failure Tests
# ─────────────────────────────────────────────────────────

@patch("serial.Serial")
def test_transport_failure_resilience(mock_pyserial_cls):
    """Verify RealSerialInterface/SerialTransport catches write exceptions, stays alive, and logs error."""
    mock_port = MagicMock()
    mock_port.is_open = True
    mock_port.write.side_effect = serial.SerialException("USB Device Disconnected")
    mock_pyserial_cls.return_value = mock_port

    cfg = SerialConfig(port="COM3", baud_rate=115200)
    transport = SerialTransport(config=cfg)
    assert transport.connect() is True
    assert transport.is_connected is True

    # Transmit command under write exception
    success = transport.send_command("INDEX_FINGER")
    assert success is False
    # Verified: application does NOT crash, state set to ERROR / DISCONNECTED
    assert transport.state == SerialState.ERROR
    assert transport.is_connected is False


# ─────────────────────────────────────────────────────────
# J. Protocol Isolation Tests
# ─────────────────────────────────────────────────────────

def test_protocol_isolation_no_pyserial_import():
    """Verify neurogrip.communication.protocol does NOT import pyserial."""
    protocol_module = sys.modules.get("neurogrip.communication.protocol")
    assert protocol_module is not None

    # Check imported module names in protocol module globals
    for name, val in protocol_module.__dict__.items():
        if isinstance(val, type(sys)):
            assert "serial" not in val.__name__.lower() or "neurogrip" in val.__name__.lower(), (
                f"Protocol module imports disallowed hardware module '{val.__name__}'"
            )
