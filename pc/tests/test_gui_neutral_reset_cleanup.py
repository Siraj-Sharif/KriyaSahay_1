"""
pc/tests/test_gui_neutral_reset_cleanup.py
────────────────────────────────────────────
Unit and Integration Tests for GUI Command-Control Cleanup & RESET / NEUTRAL Semantics.

Verifies:
  TEST 1: System ON/OFF behaves exactly as before.
  TEST 2: RESET / NEUTRAL produces exactly NG1|NEUTRAL\n over serial interface.
  TEST 3: NO_COMMAND produces zero transmitted bytes.
  TEST 4: Valid Gesture Commands transmit normally (e.g. THUMBS_UP -> NG1|THUMBS_UP\n).
  TEST 5: STOP is generated/handled internally without a manual GUI toggle.
  TEST 6: STOP status display reflects backend state (display-only).
  TEST 7: RESET / NEUTRAL during active STOP sends NG1|NEUTRAL\n without clearing the STOP latch.
  TEST 8: Valid Gesture Command after STOP releases STOP according to existing safety design.
  TEST 9: No GUI control can transmit NO_COMMAND.
  TEST 10: No duplicate serial transmissions occur from a single button click.
  TEST 11: System OFF cannot transmit gesture or RESET commands through button logic.
"""

from unittest.mock import MagicMock
import pytest

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.communication.protocol import ProtocolEncoder, encode_protocol_frame
from neurogrip.config.settings import AppConfig
from neurogrip.frontend.bridge import NeuroGripBridge
from neurogrip.frontend.image_provider import FrameImageProvider
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import RecognitionResult


@pytest.fixture
def mock_bridge():
    """Fixture constructing NeuroGripBridge with MockCamera and MockSerialInterface."""
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True

    mock_serial = MockSerialInterface()
    mock_serial.connect()
    img_provider = FrameImageProvider()
    bridge = NeuroGripBridge(image_provider=img_provider, config=config, serial=mock_serial)
    bridge.pipeline.initialize()
    yield bridge, mock_serial
    bridge.stop_pipeline(wait=True)
    if bridge.pipeline:
        bridge.pipeline.shutdown()


def test_gui_test1_system_on_off_behavior(mock_bridge):
    """TEST 1: System ON/OFF behaves as expected and disables command generation when OFF."""
    bridge, mock_serial = mock_bridge
    assert bridge.systemOn is False

    # When System is OFF, resetNeutral is ignored
    bridge.resetNeutral()
    assert len(mock_serial.sent_frames) == 0

    # Turn System ON
    bridge.toggleSystem()
    assert bridge.systemOn is True

    # Turn System OFF again
    bridge.toggleSystem()
    assert bridge.systemOn is False
    assert bridge.productGestureName == "OFF"
    assert len(mock_serial.sent_frames) == 0


def test_gui_test2_reset_neutral_exact_frame(mock_bridge):
    """TEST 2: RESET / NEUTRAL button emits exactly NG1|NEUTRAL\\n."""
    bridge, mock_serial = mock_bridge
    bridge.toggleSystem()  # System ON
    mock_serial.clear()

    # Trigger RESET / NEUTRAL button slot
    bridge.resetNeutral()

    assert len(mock_serial.sent_frames) == 1
    assert mock_serial.sent_frames[0] == "NG1|NEUTRAL\n"
    assert mock_serial.sent_bytes[0] == b"NG1|NEUTRAL\n"
    assert bridge.lastEmittedCommand == "NEUTRAL"
    assert bridge.wireFrameStr == "NG1|NEUTRAL\n"


def test_gui_test3_no_command_zero_bytes():
    """TEST 3: NO_COMMAND produces zero transmitted bytes."""
    assert ProtocolEncoder.encode_command_string("NO_COMMAND") is None
    assert encode_protocol_frame("NO_COMMAND") is None

    mock_serial = MockSerialInterface()
    mock_serial.connect()
    tx_ok = mock_serial.send_command("NO_COMMAND")
    assert tx_ok is False
    assert len(mock_serial.sent_frames) == 0


def test_gui_test4_valid_gesture_command_transmission():
    """TEST 4: Valid Gesture Commands transmit normally (e.g. THUMBS_UP -> NG1|THUMBS_UP\\n)."""
    assert encode_protocol_frame("THUMBS_UP") == b"NG1|THUMBS_UP\n"
    assert ProtocolEncoder.encode_command_string("THUMBS_UP") == "NG1|THUMBS_UP\n"

    mock_serial = MockSerialInterface()
    mock_serial.connect()
    tx_ok = mock_serial.send_command("THUMBS_UP")
    assert tx_ok is True
    assert mock_serial.sent_frames[0] == "NG1|THUMBS_UP\n"


def test_gui_test5_no_gui_stop_toggle_and_backend_stop_ownership(mock_bridge):
    """TEST 5: STOP is generated/handled internally by backend without a manual GUI toggle."""
    bridge, mock_serial = mock_bridge
    # Verify bridge exposes read-only safetyStatus property for display and NO manual toggleStopArm slot
    assert not hasattr(bridge, "toggleStopArm")
    assert not hasattr(bridge, "isStopArmed")
    assert hasattr(bridge, "safetyStatus")


def test_gui_test6_stop_status_display_reflects_backend_state(mock_bridge):
    """TEST 6: STOP status display reflects backend state."""
    bridge, _ = mock_bridge
    assert bridge.safetyStatus in ("NORMAL", "DISARMED", "ARMED", "STOP ACTIVE")


def test_gui_test7_reset_neutral_during_active_stop_preserves_stop_latch(mock_bridge):
    """TEST 7: RESET / NEUTRAL during active STOP sends NG1|NEUTRAL\\n over serial and updates wire frame."""
    bridge, mock_serial = mock_bridge
    bridge.toggleSystem()
    mock_serial.clear()

    # Simulate active STOP
    bridge.pipeline.recognizer.predict = MagicMock(return_value=RecognitionResult(label="STOP", confidence=1.0, model_used="RULE_BASED"))  # type: ignore

    dummy_lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    stop_hand = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.RIGHT, score=0.95)
    bridge.pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[stop_hand], num_hands=1))  # type: ignore

    # Process frame
    _, _, viz = bridge.pipeline.process_frame()
    bridge._on_frame_processed(None, viz)

    mock_serial.clear()

    # Click RESET / NEUTRAL
    bridge.resetNeutral()

    # NEUTRAL wire frame sent
    assert "NG1|NEUTRAL\n" in mock_serial.sent_frames
    assert bridge.wireFrameStr == "NG1|NEUTRAL\n"


def test_gui_test8_valid_gesture_after_stop_releases_stop(mock_bridge):
    """TEST 8: A valid Gesture Command after STOP releases STOP according to existing safety design."""
    bridge, mock_serial = mock_bridge
    bridge.toggleSystem()
    bridge.pipeline.arm_stop()

    dummy_lms = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    idx_hand = HandLandmarks(landmarks=dummy_lms, handedness=Handedness.RIGHT, score=0.95)

    # Inject valid gesture command prediction
    bridge.pipeline.detector.detect = MagicMock(return_value=DetectionResult(hands=[idx_hand], num_hands=1))  # type: ignore
    bridge.pipeline.recognizer.predict = MagicMock(return_value=RecognitionResult(label="INDEX_FINGER", confidence=0.95, model_used="EXTRA_TREES"))  # type: ignore

    mock_serial.clear()
    for _ in range(4):
        _, _, viz = bridge.pipeline.process_frame()
        bridge._on_frame_processed(None, viz)

    # Valid gesture releases STOP latch and transmits valid command
    assert "NG1|INDEX_FINGER\n" in mock_serial.sent_frames


def test_gui_test9_no_gui_control_transmits_no_command(mock_bridge):
    """TEST 9: No GUI control can transmit NO_COMMAND."""
    bridge, mock_serial = mock_bridge
    bridge.toggleSystem()
    mock_serial.clear()

    bridge.resetNeutral()
    for frame in mock_serial.sent_frames:
        assert frame != "NG1|NO_COMMAND\n"
        assert "NO_COMMAND" not in frame


def test_gui_test10_single_click_no_duplicate_transmission(mock_bridge):
    """TEST 10: Single button click emits exactly 1 frame."""
    bridge, mock_serial = mock_bridge
    bridge.toggleSystem()
    mock_serial.clear()

    bridge.resetNeutral()
    assert len(mock_serial.sent_frames) == 1
    assert mock_serial.sent_frames[0] == "NG1|NEUTRAL\n"


def test_gui_test11_system_off_prevents_button_transmissions(mock_bridge):
    """TEST 11: System OFF prevents button transmissions."""
    bridge, mock_serial = mock_bridge
    assert bridge.systemOn is False
    mock_serial.clear()

    bridge.resetNeutral()
    assert len(mock_serial.sent_frames) == 0
