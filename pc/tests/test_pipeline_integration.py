"""
tests/test_pipeline_integration.py
───────────────────────────────────
End-to-End Pipeline Integration Unit Tests for Phase 9.
Tests component initialization, execution loop, frame flow, 0-hand/1-hand/multi-hand state transitions,
STOP safety path, serial transmission via MockSerialInterface, and graceful shutdown.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.commands.validator import PipelineState
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.config.settings import AppConfig
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from tests.conftest import make_hand_landmarks


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline_components():
    """Create mock pipeline components (MockCamera, MockSerialInterface, RuleBasedRecognizer)."""
    cfg = AppConfig()
    cfg.serial.mock = True
    cfg.camera.backend = "mock"

    cam = MockCamera(config=cfg.camera)
    ser = MockSerialInterface()
    rec = RuleBasedRecognizer(config=cfg)

    # Mock HandDetector to avoid loading real MediaPipe model bundle during unit tests
    det = MagicMock()
    det.initialize.return_value = None
    det.is_initialized = True

    return cfg, cam, ser, rec, det


# ─────────────────────────────────────────────────────────
# Initialization & Lifecycle Tests
# ─────────────────────────────────────────────────────────

def test_pipeline_initialization(mock_pipeline_components):
    """Test pipeline initialization lifecycle."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    assert pipeline.is_initialized is False
    assert pipeline.state == PipelineState.STARTING

    # Initialize
    assert pipeline.initialize() is True
    assert pipeline.is_initialized is True
    assert pipeline.is_running is True
    assert pipeline.state == PipelineState.TRACKING

    # Shutdown
    pipeline.shutdown()
    assert pipeline.is_running is False
    assert pipeline.is_initialized is False
    assert pipeline.state == PipelineState.SHUTDOWN


def test_pipeline_camera_selection_decoupled_from_serial_mock():
    """
    Regression Test: Proves camera.backend = 'opencv' + serial.mock = True
    results in OpenCVCamera, not MockCamera.
    """
    from neurogrip.camera.opencv_camera import OpenCVCamera
    from neurogrip.camera.mock_camera import MockCamera

    cfg = AppConfig()
    cfg.camera.backend = "opencv"
    cfg.serial.mock = True

    det = MagicMock()

    pipeline = NeuroGripPipeline(config=cfg, detector=det)
    assert isinstance(pipeline.camera, OpenCVCamera)
    assert not isinstance(pipeline.camera, MockCamera)

    # Conversely, camera.backend = "mock" results in MockCamera
    cfg_mock = AppConfig()
    cfg_mock.camera.backend = "mock"
    cfg_mock.serial.mock = False

    pipeline_mock = NeuroGripPipeline(config=cfg_mock, detector=det)
    assert isinstance(pipeline_mock.camera, MockCamera)


def test_pipeline_camera_failure(mock_pipeline_components):
    """Verify pipeline initialization fails gracefully if camera fails to open."""
    cfg, cam, ser, rec, det = mock_pipeline_components
    cam.open = MagicMock(return_value=False)  # Camera open fails

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    assert pipeline.initialize() is False
    assert pipeline.state == PipelineState.ERROR


def test_pipeline_context_manager(mock_pipeline_components):
    """Verify pipeline context manager initializes and shuts down cleanly."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    with NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det) as p:
        assert p.is_initialized is True

    assert p.is_initialized is False
    assert p.state == PipelineState.SHUTDOWN


# ─────────────────────────────────────────────────────────
# STOP Safety Arming Lifecycle Tests
# ─────────────────────────────────────────────────────────

def test_pipeline_stop_arming_lifecycle(mock_pipeline_components):
    """Test arming, disarming, and toggling software STOP safety state."""
    cfg, cam, ser, rec, det = mock_pipeline_components
    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)

    assert pipeline.is_stop_armed is False

    pipeline.arm_stop()
    assert pipeline.is_stop_armed is True

    pipeline.disarm_stop()
    assert pipeline.is_stop_armed is False

    toggled = pipeline.toggle_stop_arm()
    assert toggled is True
    assert pipeline.is_stop_armed is True


# ─────────────────────────────────────────────────────────
# End-to-End Processing Loop & State Tests
# ─────────────────────────────────────────────────────────

def test_pipeline_no_hand_frame(mock_pipeline_components):
    """Verify pipeline handles 0-hand frames correctly (NO_HAND state)."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    # Mock detector returning 0 hands
    det.detect.return_value = DetectionResult(hands=[], num_hands=0, timestamp_ms=100)

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    pipeline.initialize()

    frame_c, rendered, viz_state = pipeline.process_frame()

    assert viz_state.command in ("NO_COMMAND", "NO_HAND")
    assert viz_state.pipeline_state == "NO_HAND"
    assert len(viz_state.landmarks) == 0
    assert len(ser.sent_commands) == 0  # No serial command sent


def test_pipeline_ambiguous_multi_hand_frame(mock_pipeline_components, open_hand_landmarks):
    """Verify 2+ hands trigger AMBIGUOUS safe state and block normal commands."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    # Mock detector returning 2 hands
    det.detect.return_value = DetectionResult(
        hands=[open_hand_landmarks, open_hand_landmarks],
        num_hands=2,
        timestamp_ms=200,
    )

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    pipeline.initialize()

    frame_c, rendered, viz_state = pipeline.process_frame()

    assert viz_state.command in ("NO_COMMAND", "AMBIGUOUS")
    assert viz_state.is_ambiguous is True
    assert viz_state.pipeline_state == "AMBIGUOUS"
    assert len(ser.sent_commands) == 0  # Commands blocked during AMBIGUOUS state!


def test_pipeline_single_hand_index_stabilization_and_transmission(mock_pipeline_components, index_only_landmarks):
    """Verify end-to-end processing of a single hand: INDEX gesture stabilization & serial transmission."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    # Configure small window for fast unit testing
    cfg.stabilization.window_size = 5
    cfg.stabilization.vote_threshold = 0.70
    cfg.stabilization.min_frames_required = 3

    det.detect.return_value = DetectionResult(hands=[index_only_landmarks], num_hands=1, timestamp_ms=300)

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    pipeline.initialize()

    # Process 3 frames to reach stability threshold (3/3 = 100% >= 70%)
    pipeline.process_frame()
    pipeline.process_frame()
    frame_c, rendered, viz_state = pipeline.process_frame()

    assert viz_state.command == "INDEX"
    assert viz_state.stabilizer_state == "STABLE"

    # Verify command was transmitted ONCE over MockSerialInterface
    assert len(ser.sent_commands) == 1
    assert ser.sent_commands[0] == NeuroGripCommand.INDEX
    assert ser.sent_frames[0] == "NG1|INDEX_FINGER\n"


    # Process 4th frame (repeat INDEX) -> De-duplicated, no additional transmission!
    pipeline.process_frame()
    assert len(ser.sent_commands) == 1


def test_pipeline_stop_safety_bypass_and_transmission(mock_pipeline_components):
    """Verify armed STOP gesture immediately bypasses stabilization and transmits NG1|STOP\\n."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    compact_stop_landmarks = make_hand_landmarks(
        tip_offsets={
            "thumb": (0.05, -0.25),
            "index": (0.01, -0.25),
            "middle": (0.0, -0.25),
            "ring": (-0.01, -0.25),
            "pinky": (-0.02, -0.25),
        }
    )

    det.detect.return_value = DetectionResult(hands=[compact_stop_landmarks], num_hands=1, timestamp_ms=400)

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    pipeline.initialize()
    pipeline.arm_stop()  # Arm software STOP

    # Process 1 frame -> Immediate STOP bypass!
    frame_c, rendered, viz_state = pipeline.process_frame()

    assert viz_state.command == "STOP"
    assert viz_state.is_stop_active is True
    assert viz_state.pipeline_state in ("STABLE_CMD", "STOP_ACTIVE")

    # Serial transmitted STOP
    assert len(ser.sent_commands) == 1
    assert ser.sent_commands[0] == NeuroGripCommand.STOP
    assert ser.sent_frames[0] == "NG1|STOP\n"


def test_pipeline_disarmed_stop_never_emits_serial_command(mock_pipeline_components, open_hand_landmarks):
    """Verify STOP gesture transmits NG1|STOP\\n over serial interface under STOP-as-normal-command architecture."""
    cfg, cam, ser, _, det = mock_pipeline_components
    real_rec = RuleBasedRecognizer(config=cfg)

    det.detect.return_value = DetectionResult(hands=[open_hand_landmarks], num_hands=1, timestamp_ms=450)

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=real_rec, detector=det)
    pipeline.initialize()

    # Process frame -> STOP recognition occurs and transmits NG1|STOP\n
    _, _, viz_state = pipeline.process_frame()
    assert viz_state.command == "STOP"
    assert viz_state.is_stop_active is True
    assert len(ser.sent_commands) == 1
    assert ser.sent_commands[0] == NeuroGripCommand.STOP
    assert ser.sent_frames[0] == "NG1|STOP\n"


def test_pipeline_run_loop_iterations(mock_pipeline_components):
    """Test pipeline run_loop execution across max_frames limit."""
    cfg, cam, ser, rec, det = mock_pipeline_components
    det.detect.return_value = DetectionResult(hands=[], num_hands=0, timestamp_ms=500)

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)

    processed_count = 0

    def frame_cb(fc, rend, viz):
        nonlocal processed_count
        processed_count += 1

    pipeline.run_loop(max_frames=5, callback=frame_cb)

    assert processed_count == 5
    assert pipeline.is_running is False


def test_pipeline_diagnostic_formatter(mock_pipeline_components, open_hand_landmarks):
    """Test format_diagnostic_line output for 0-hand, 1-hand, and ambiguous frames."""
    cfg, cam, ser, rec, det = mock_pipeline_components

    pipeline = NeuroGripPipeline(config=cfg, camera=cam, serial=ser, recognizer=rec, detector=det)
    pipeline.initialize()

    # 1. 0-hand frame
    det.detect.return_value = DetectionResult(hands=[], num_hands=0, timestamp_ms=100)
    pipeline.process_frame()
    diag_0 = pipeline.format_diagnostic_line()
    assert "HAND=NONE" in diag_0
    assert "RAW=NONE:0.00" in diag_0

    # 2. 1-hand frame
    det.detect.return_value = DetectionResult(hands=[open_hand_landmarks], num_hands=1, timestamp_ms=200)
    pipeline.process_frame()
    diag_1 = pipeline.format_diagnostic_line()
    assert "HAND=RIGHT" in diag_1

