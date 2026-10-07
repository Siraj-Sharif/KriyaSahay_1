"""Unit and Integration Tests for Final Production Hybrid CV Pipeline.

Tests cover all 20 required production pipeline verification items:
 1. HaGRID class -> canonical mapping
 2. Unauthorized HaGRID class -> NO_COMMAND
 3. 'one' -> NO_COMMAND (No ONE_FINGER)
 4. stop + stop_inverted -> STOP
 5. two_up + two_up_inverted -> TWO_FINGERS
 6. three/three2/three3/three_gun -> THREE_FINGERS
 7. Hybrid routing for INDEX_FINGER (routes to Extra Trees)
 8. Hybrid routing for TWO_FINGERS (routes to Extra Trees)
 9. Hybrid routing for PINKY (routes to Extra Trees)
10. HaGRID normal gesture routing (routes directly to HaGRID ResNet18)
11. 0 hands -> NO_COMMAND
12. >1 hands -> NO_COMMAND
13. Invalid feature vector -> NO_COMMAND
14. Missing model -> safe startup failure
15. Extra Trees model expects exactly 68-D features
16. NO_COMMAND never reaches transport boundary
17. Temporal stabilizer receives FINAL hybrid command
18. STOP safety path preserved and authoritative
19. Model loading happens once at startup
20. End-to-end pipeline processes a mock frame without crashing
"""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.transport import CommandTransport
from neurogrip.config.settings import AppConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import (
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    map_hagrid_to_neurogrip,
)
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)
from neurogrip.recognition.base import RecognitionResult
from neurogrip.recognition.hybrid import (
    HybridRecognizer,
    map_extratrees_to_neurogrip,
)
from neurogrip.recognition.ml_recognizer import MLRecognizer, MLRecognizerError
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from neurogrip.stabilization.temporal import StabilizerResult, StabilizerState, TemporalStabilizer


def _build_index_only_landmarks() -> HandLandmarks:
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    lms[5] = NormalizedLandmark(x=0.45, y=0.5, z=0.0)
    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[13] = NormalizedLandmark(x=0.55, y=0.5, z=0.0)
    lms[17] = NormalizedLandmark(x=0.6, y=0.5, z=0.0)
    lms[6] = NormalizedLandmark(x=0.45, y=0.38, z=0.0)
    lms[7] = NormalizedLandmark(x=0.45, y=0.25, z=0.0)
    lms[8] = NormalizedLandmark(x=0.45, y=0.1, z=0.0)
    for tip_idx in (12, 16, 20):
        lms[tip_idx - 2] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
        lms[tip_idx - 1] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.56, z=0.0)
        lms[tip_idx] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
    lms[1] = NormalizedLandmark(x=0.48, y=0.7, z=0.0)
    lms[2] = NormalizedLandmark(x=0.47, y=0.65, z=0.0)
    lms[3] = NormalizedLandmark(x=0.46, y=0.6, z=0.0)
    lms[4] = NormalizedLandmark(x=0.45, y=0.58, z=0.0)
    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def _build_two_fingers_landmarks() -> HandLandmarks:
    lms = _build_index_only_landmarks().landmarks
    lms[10] = NormalizedLandmark(x=0.5, y=0.38, z=0.0)
    lms[11] = NormalizedLandmark(x=0.5, y=0.25, z=0.0)
    lms[12] = NormalizedLandmark(x=0.5, y=0.1, z=0.0)
    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def _build_pinky_only_landmarks() -> HandLandmarks:
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    lms[5] = NormalizedLandmark(x=0.45, y=0.5, z=0.0)
    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[13] = NormalizedLandmark(x=0.55, y=0.5, z=0.0)
    lms[17] = NormalizedLandmark(x=0.6, y=0.5, z=0.0)
    for tip_idx in (8, 12, 16):
        lms[tip_idx - 2] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
        lms[tip_idx - 1] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.56, z=0.0)
        lms[tip_idx] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
    lms[18] = NormalizedLandmark(x=0.6, y=0.38, z=0.0)
    lms[19] = NormalizedLandmark(x=0.6, y=0.25, z=0.0)
    lms[20] = NormalizedLandmark(x=0.6, y=0.1, z=0.0)
    lms[1] = NormalizedLandmark(x=0.48, y=0.7, z=0.0)
    lms[2] = NormalizedLandmark(x=0.47, y=0.65, z=0.0)
    lms[3] = NormalizedLandmark(x=0.46, y=0.6, z=0.0)
    lms[4] = NormalizedLandmark(x=0.45, y=0.58, z=0.0)
    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def _build_fist_landmarks() -> HandLandmarks:
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    lms[5] = NormalizedLandmark(x=0.45, y=0.5, z=0.0)
    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[13] = NormalizedLandmark(x=0.55, y=0.5, z=0.0)
    lms[17] = NormalizedLandmark(x=0.6, y=0.5, z=0.0)
    for tip_idx in (8, 12, 16, 20):
        lms[tip_idx - 2] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
        lms[tip_idx - 1] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.56, z=0.0)
        lms[tip_idx] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
    lms[1] = NormalizedLandmark(x=0.48, y=0.7, z=0.0)
    lms[2] = NormalizedLandmark(x=0.47, y=0.65, z=0.0)
    lms[3] = NormalizedLandmark(x=0.46, y=0.6, z=0.0)
    lms[4] = NormalizedLandmark(x=0.45, y=0.58, z=0.0)
    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def test_1_hagrid_class_to_canonical_mapping():
    """1. Verify valid HaGRID raw class mapping to canonical taxonomy."""
    assert map_hagrid_to_neurogrip("call") == "CALL"
    assert map_hagrid_to_neurogrip("fist") == "CLOSED_FIST"
    assert map_hagrid_to_neurogrip("four") == "FOUR_FINGERS"
    assert map_hagrid_to_neurogrip("grabbing") == "GRABBING"
    assert map_hagrid_to_neurogrip("grip") == "GRIP"
    assert map_hagrid_to_neurogrip("like") == "THUMBS_UP"
    assert map_hagrid_to_neurogrip("little_finger") == "PINKY"
    assert map_hagrid_to_neurogrip("middle_finger") == "MIDDLE_FINGER"
    assert map_hagrid_to_neurogrip("ok") == "OK"
    assert map_hagrid_to_neurogrip("point") == "INDEX_FINGER"
    assert map_hagrid_to_neurogrip("rock") == "INDEX_PINKY"


def test_2_unauthorized_hagrid_class_maps_to_no_command():
    """2. Verify unauthorized / unknown HaGRID classes map to NO_COMMAND."""
    assert map_hagrid_to_neurogrip("holy") == NO_COMMAND
    assert map_hagrid_to_neurogrip("timeout") == NO_COMMAND
    assert map_hagrid_to_neurogrip("xsign") == NO_COMMAND
    assert map_hagrid_to_neurogrip("hand_heart") == NO_COMMAND
    assert map_hagrid_to_neurogrip("dislike") == NO_COMMAND
    assert map_hagrid_to_neurogrip("take_picture") == NO_COMMAND
    assert map_hagrid_to_neurogrip("no_gesture") == NO_COMMAND


def test_3_one_maps_to_no_command_no_one_finger():
    """3. Verify HaGRID 'one' maps strictly to NO_COMMAND and no ONE_FINGER exists."""
    assert map_hagrid_to_neurogrip("one") == NO_COMMAND
    assert "ONE_FINGER" not in NEUROGRIP_TAXONOMY
    assert "ONE" not in NEUROGRIP_TAXONOMY


def test_4_stop_and_stop_inverted_map_to_stop():
    """4. Verify stop and stop_inverted both map to STOP."""
    assert map_hagrid_to_neurogrip("stop") == "STOP"
    assert map_hagrid_to_neurogrip("stop_inverted") == "STOP"


def test_5_two_up_and_two_up_inverted_map_to_two_fingers():
    """5. Verify two_up and two_up_inverted both map to TWO_FINGERS."""
    assert map_hagrid_to_neurogrip("two_up") == "TWO_FINGERS"
    assert map_hagrid_to_neurogrip("two_up_inverted") == "TWO_FINGERS"


def test_6_three_variants_map_to_three_fingers():
    """6. Verify three, three2, three3, and three_gun all map to THREE_FINGERS."""
    assert map_hagrid_to_neurogrip("three") == "THREE_FINGERS"
    assert map_hagrid_to_neurogrip("three2") == "THREE_FINGERS"
    assert map_hagrid_to_neurogrip("three3") == "THREE_FINGERS"
    assert map_hagrid_to_neurogrip("three_gun") == "THREE_FINGERS"


def test_7_hybrid_routing_index_finger_routes_to_extra_trees():
    """7. Verify INDEX_FINGER landmark posture routes to Extra Trees and skips HaGRID inference."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True
    mock_et.predict.return_value = MagicMock(label="INDEX", confidence=0.88, all_scores={"INDEX": 0.88})

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)
    index_hand = _build_index_only_landmarks()

    res = recognizer.predict(dummy_feats, hand_landmarks=index_hand, is_stop_armed=False, raw_frame=dummy_frame)

    assert res.model_used == "EXTRA_TREES"
    assert res.label == "INDEX_FINGER"
    assert "LANDMARK_GATE_EXTRA_TREES" in res.reason
    assert mock_et.predict.called
    assert not mock_hagrid.predict.called  # Skips HaGRID CPU ResNet18!


def test_8_hybrid_routing_two_fingers_routes_to_extra_trees():
    """8. Verify TWO_FINGERS landmark posture routes to Extra Trees and skips HaGRID inference."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True
    mock_et.predict.return_value = MagicMock(label="TWO_FINGER", confidence=0.82, all_scores={"TWO_FINGER": 0.82})

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)
    two_hand = _build_two_fingers_landmarks()

    res = recognizer.predict(dummy_feats, hand_landmarks=two_hand, is_stop_armed=False, raw_frame=dummy_frame)

    assert res.model_used == "EXTRA_TREES"
    assert res.label == "TWO_FINGERS"
    assert mock_et.predict.called
    assert not mock_hagrid.predict.called  # Skips HaGRID CPU ResNet18!


def test_9_hybrid_routing_pinky_routes_to_extra_trees():
    """9. Verify PINKY landmark posture routes to Extra Trees and skips HaGRID inference."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True
    mock_et.predict.return_value = MagicMock(label="PINKY", confidence=0.84, all_scores={"PINKY": 0.84})

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)
    pinky_hand = _build_pinky_only_landmarks()

    res = recognizer.predict(dummy_feats, hand_landmarks=pinky_hand, is_stop_armed=False, raw_frame=dummy_frame)

    assert res.model_used == "EXTRA_TREES"
    assert res.label == "PINKY"
    assert mock_et.predict.called
    assert not mock_hagrid.predict.called  # Skips HaGRID CPU ResNet18!


def test_10_hagrid_normal_gesture_routing():
    """10. Verify HaGRID normal gestures (e.g. GRIP, THUMBS_UP) route directly to HaGRID ResNet18."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {
        "raw_class": "grip",
        "mapped_command": "GRIP",
        "confidence": 0.94,
        "top_k": [{"mapped_command": "GRIP", "probability": 0.94}],
    }

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)
    fist_hand = _build_fist_landmarks()

    res = recognizer.predict(dummy_feats, hand_landmarks=fist_hand, is_stop_armed=False, raw_frame=dummy_frame)

    assert res.model_used == "HAGRID"
    assert res.label == "GRIP"
    assert res.confidence == pytest.approx(0.94)
    assert not mock_et.predict.called


def test_11_zero_hands_emits_no_command():
    """11. Verify 0 hands detected outputs NO_COMMAND."""
    cfg = AppConfig.default()
    mock_transport = MagicMock(spec=CommandTransport)

    pipeline = NeuroGripPipeline(config=cfg, transport=mock_transport)
    pipeline.initialize()

    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[], num_hands=0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.camera.read_frame = MagicMock(return_value=MagicMock(is_valid=True, frame=dummy_frame, timestamp_ms=0))

        _, _, viz_state = pipeline.process_frame()

        assert viz_state.command == "NO_COMMAND"
        assert viz_state.hand_count == "0"
        assert mock_transport.send_idle.called
        assert not mock_transport.send_command.called


def test_12_multiple_hands_emits_no_command():
    """12. Verify 2+ hands detected outputs NO_COMMAND / AMBIGUOUS."""
    cfg = AppConfig.default()
    mock_transport = MagicMock(spec=CommandTransport)

    pipeline = NeuroGripPipeline(config=cfg, transport=mock_transport)
    pipeline.initialize()

    with patch.object(pipeline.detector, "detect") as mock_detect:
        h1 = _build_fist_landmarks()
        h2 = _build_fist_landmarks()
        mock_detect.return_value = DetectionResult(hands=[h1, h2], num_hands=2)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.camera.read_frame = MagicMock(return_value=MagicMock(is_valid=True, frame=dummy_frame, timestamp_ms=0))

        _, _, viz_state = pipeline.process_frame()

        assert viz_state.command == "NO_COMMAND"
        assert viz_state.is_ambiguous
        assert viz_state.hand_count == "2+"
        assert mock_transport.send_idle.called
        assert not mock_transport.send_command.called


def test_13_invalid_feature_vector_emits_no_command():
    """13. Verify invalid feature vector length returns NO_COMMAND safely."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {
        "raw_class": "point",
        "mapped_command": "INDEX_FINGER",
        "confidence": 0.90,
        "top_k": [],
    }

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    invalid_feats = np.zeros(67, dtype=np.float32)  # Wrong dimension (67 instead of 68)
    index_hand = _build_index_only_landmarks()

    res = recognizer.predict(invalid_feats, hand_landmarks=index_hand, is_stop_armed=False, raw_frame=dummy_frame)

    assert res.label == NO_COMMAND
    assert res.model_used == "EXTRA_TREES"


def test_14_missing_model_bundle_safe_failure():
    """14. Verify missing Extra Trees model path fails safely without crashing."""
    cfg = AppConfig.default()
    ml_rec = MLRecognizer(config=cfg)
    loaded = ml_rec.load_model("pc/models/NON_EXISTENT_MODEL.pkl")
    assert not loaded
    assert not ml_rec.is_ready


def test_15_extra_trees_feature_dimension_verification():
    """15. Verify FeatureExtractor.FEATURE_DIM is exactly 68."""
    assert FeatureExtractor.FEATURE_DIM == 68
    feats = FeatureExtractor().extract(_build_fist_landmarks())
    assert feats.shape == (68,)


def test_16_no_command_never_reaches_transport():
    """16. Verify NO_COMMAND is never emitted via transport.send_command."""
    mock_transport = MagicMock(spec=CommandTransport)
    pipeline = NeuroGripPipeline(transport=mock_transport)
    pipeline.initialize()

    # Simulate process_frame with NO_HAND
    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[], num_hands=0)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.camera.read_frame = MagicMock(return_value=MagicMock(is_valid=True, frame=dummy_frame, timestamp_ms=0))
        pipeline.process_frame()

    # Verify send_command was never called with NO_COMMAND
    for call_item in mock_transport.send_command.call_args_list:
        cmd_arg = call_item[0][0]
        assert cmd_arg.value != "NO_COMMAND"


def test_17_temporal_stabilizer_receives_final_hybrid_command():
    """17. Verify TemporalStabilizer operates on the final hybrid command output."""
    mock_stabilizer = MagicMock(spec=TemporalStabilizer)
    mock_stabilizer.update.return_value = StabilizerResult(
        state=StabilizerState.STABLE,
        stable_command="GRIP",
        emitted_command="GRIP",
        confidence=0.95,
        window_fill=5,
        winning_count=5,
    )

    pipeline = NeuroGripPipeline()
    pipeline.stabilizer = mock_stabilizer

    rec_res = RecognitionResult(
        label="GRIP", confidence=0.95, model_used="HAGRID", raw_prediction="grip"
    )
    mock_rec = MagicMock()
    mock_rec.predict.return_value = rec_res

    pipeline.recognizer = mock_rec
    pipeline.initialize()

    dummy_hand = _build_fist_landmarks()
    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[dummy_hand], num_hands=1)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline.camera.read_frame = MagicMock(return_value=MagicMock(is_valid=True, frame=dummy_frame, timestamp_ms=0))

        pipeline.process_frame()

    assert mock_stabilizer.update.called
    rec_arg = mock_stabilizer.update.call_args[0][0]
    assert rec_arg.label == "GRIP"


def test_18_stop_safety_path_preserved_and_authoritative():
    """18. Verify deterministic rule-based STOP safety path is authoritative."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {"raw_class": "fist", "mapped_command": "CLOSED_FIST", "confidence": 0.99, "top_k": []}

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    mock_rule = MagicMock(spec=RuleBasedRecognizer)
    mock_rule.finger_detector = RuleBasedRecognizer().finger_detector
    mock_rule.predict.return_value = RecognitionResult(
        label="STOP", confidence=1.0, recognizer_type="rule_based", model_used="RULE_BASED", raw_prediction="STOP"
    )

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et, rule_recognizer=mock_rule)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)
    dummy_hand = _build_fist_landmarks()

    # RuleBasedRecognizer returning STOP triggers authoritative STOP safety override
    res = recognizer.predict(dummy_feats, hand_landmarks=dummy_hand, is_stop_armed=True, raw_frame=dummy_frame)
    assert res.label == "STOP"
    assert res.model_used == "RULE_BASED"

    # Even when disarmed, recognition still identifies STOP (decoupled from hardware transmission)
    res_disarmed = recognizer.predict(dummy_feats, hand_landmarks=dummy_hand, is_stop_armed=False, raw_frame=dummy_frame)
    assert res_disarmed.label == "STOP"
    assert res_disarmed.model_used == "RULE_BASED"


def test_19_model_loading_happens_once_at_startup():
    """19. Verify model instances are loaded once during initialization."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)

    # Execute multiple predictions
    for _ in range(5):
        recognizer.predict(dummy_feats, raw_frame=dummy_frame)

    # Verify no additional load_model calls occurred
    assert not mock_et.load_model.called


def test_20_pipeline_mock_frame_processing():
    """20. Verify end-to-end pipeline processes a mock camera frame cleanly without crashing."""
    cfg = AppConfig.default()
    cfg.camera.backend = "mock"
    mock_transport = MagicMock(spec=CommandTransport)

    pipeline = NeuroGripPipeline(config=cfg, transport=mock_transport)
    pipeline.initialize()

    dummy_hand = _build_fist_landmarks()
    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[dummy_hand], num_hands=1)
        frame_c, rendered, viz_state = pipeline.process_frame()

        assert frame_c.is_valid
        assert viz_state.hand_count == "1"
        assert viz_state.latency_ms > 0.0

    pipeline.shutdown()


def test_21_extra_trees_cannot_leak_non_owned_commands():
    """21. Verify Extra Trees raw prediction mapping behavior for owned and unmapped classes."""
    assert map_extratrees_to_neurogrip("INDEX") == "INDEX_FINGER"
    assert map_extratrees_to_neurogrip("INDEX_FINGER") == "INDEX_FINGER"
    assert map_extratrees_to_neurogrip("TWO_FINGER") == "TWO_FINGERS"
    assert map_extratrees_to_neurogrip("TWO_FINGERS") == "TWO_FINGERS"
    assert map_extratrees_to_neurogrip("PINKY") == "PINKY"

    # Unmapped Extra Trees raw labels map to NO_COMMAND
    assert map_extratrees_to_neurogrip("REST") == NO_COMMAND
    assert map_extratrees_to_neurogrip("RING") == NO_COMMAND
    assert map_extratrees_to_neurogrip("NO_COMMAND") == NO_COMMAND
    assert map_extratrees_to_neurogrip("UNKNOWN") == NO_COMMAND


def test_22_call_pose_strictly_routes_to_hagrid_not_extra_trees():
    """22. Verify CALL landmark posture (Thumb OPEN, Pinky OPEN) returns None from gate and routes to HaGRID."""
    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {
        "raw_class": "call",
        "mapped_command": "CALL",
        "confidence": 0.96,
        "top_k": [{"mapped_command": "CALL", "probability": 0.96}],
    }

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    recognizer = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et)

    # Build synthetic CALL landmarks (Thumb OPEN, Pinky OPEN, Index/Middle/Ring CLOSED)
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    # Thumb OPEN (extended laterally)
    lms[4] = NormalizedLandmark(x=0.2, y=0.6, z=0.0)
    lms[2] = NormalizedLandmark(x=0.35, y=0.7, z=0.0)
    lms[3] = NormalizedLandmark(x=0.25, y=0.65, z=0.0)
    # Pinky OPEN (extended up)
    lms[17] = NormalizedLandmark(x=0.6, y=0.5, z=0.0)
    lms[18] = NormalizedLandmark(x=0.62, y=0.35, z=0.0)
    lms[20] = NormalizedLandmark(x=0.65, y=0.2, z=0.0)
    # Index/Middle/Ring CLOSED
    for tip_idx in (8, 12, 16):
        lms[tip_idx - 2] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)
        lms[tip_idx - 1] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.56, z=0.0)
        lms[tip_idx] = NormalizedLandmark(x=lms[tip_idx - 3].x, y=0.55, z=0.0)

    call_hand = HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)

    gate_res = recognizer._check_landmark_routing_gate(call_hand)
    assert gate_res is None, f"CALL pose should return None from routing gate, got '{gate_res}'"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_feats = np.zeros(68, dtype=np.float32)

    res = recognizer.predict(dummy_feats, hand_landmarks=call_hand, is_stop_armed=False, raw_frame=dummy_frame)
    assert res.model_used == "HAGRID"
    assert res.label == "CALL"
    assert not mock_et.predict.called


def test_23_stop_safety_state_resets_on_tracking_loss():
    """23. Verify STOP safety state and temporal stabilizer reset on tracking loss / 0 hands."""
    stabilizer = TemporalStabilizer()

    # Feed STOP frame when armed -> Immediate STOP safety bypass
    rec_stop = RecognitionResult(label="STOP", confidence=1.0, recognizer_type="rule_based", model_used="RULE_BASED", raw_prediction="STOP")
    res = stabilizer.update(rec_stop, is_stop_armed=True)

    assert res.state == StabilizerState.STOP
    assert res.emitted_command == "STOP"

    # Reset on tracking loss / zero hands
    stabilizer.reset()
    assert stabilizer.current_state == StabilizerState.UNKNOWN_STATE
    assert stabilizer.last_emitted_command is None


def test_24_unknown_thumb_state_fails_closed_to_hagrid():
    """24. Verify that an UNKNOWN thumb state fails closed (returns None) from PINKY routing gate."""
    recognizer = HybridRecognizer()

    # Create synthetic Pinky-like landmarks but force non-palm-facing (making finger states UNKNOWN)
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    # Set degenerate/edge-on coordinates to force FingerState.UNKNOWN for thumb
    unknown_thumb_hand = HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)

    gate_res = recognizer._check_landmark_routing_gate(unknown_thumb_hand)
    assert gate_res is None, f"UNKNOWN thumb state should return None from routing gate, got '{gate_res}'"


def _build_stop_landmarks(is_left: bool = False, is_inverted: bool = False) -> HandLandmarks:
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    # Main fingers extended UP
    # Index (5, 6, 7, 8)
    lms[5] = NormalizedLandmark(x=0.42, y=0.5, z=0.0)
    lms[6] = NormalizedLandmark(x=0.42, y=0.38, z=0.0)
    lms[7] = NormalizedLandmark(x=0.42, y=0.25, z=0.0)
    lms[8] = NormalizedLandmark(x=0.42, y=0.1, z=0.0)
    # Middle (9, 10, 11, 12)
    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[10] = NormalizedLandmark(x=0.5, y=0.37, z=0.0)
    lms[11] = NormalizedLandmark(x=0.5, y=0.23, z=0.0)
    lms[12] = NormalizedLandmark(x=0.5, y=0.08, z=0.0)
    # Ring (13, 14, 15, 16)
    lms[13] = NormalizedLandmark(x=0.58, y=0.5, z=0.0)
    lms[14] = NormalizedLandmark(x=0.58, y=0.38, z=0.0)
    lms[15] = NormalizedLandmark(x=0.58, y=0.25, z=0.0)
    lms[16] = NormalizedLandmark(x=0.58, y=0.12, z=0.0)
    # Pinky (17, 18, 19, 20)
    lms[17] = NormalizedLandmark(x=0.66, y=0.52, z=0.0)
    lms[18] = NormalizedLandmark(x=0.66, y=0.41, z=0.0)
    lms[19] = NormalizedLandmark(x=0.66, y=0.30, z=0.0)
    lms[20] = NormalizedLandmark(x=0.66, y=0.18, z=0.0)
    # Thumb (1, 2, 3, 4) — spread widely out laterally
    lms[1] = NormalizedLandmark(x=0.45, y=0.72, z=0.0)
    lms[2] = NormalizedLandmark(x=0.35, y=0.68, z=0.0)
    lms[3] = NormalizedLandmark(x=0.22, y=0.64, z=0.0)
    lms[4] = NormalizedLandmark(x=0.10, y=0.60, z=0.0)

    if is_inverted:
        for i in range(len(lms)):
            lms[i] = NormalizedLandmark(x=lms[i].x, y=lms[i].y, z=-lms[i].z)

    handedness = Handedness.LEFT if is_left else Handedness.RIGHT
    return HandLandmarks(landmarks=lms, handedness=handedness, score=0.95)


def test_25_front_stop_recognition():
    """25. Verify RuleBasedRecognizer and HybridRecognizer recognize front-facing STOP gesture."""
    rule_rec = RuleBasedRecognizer()
    stop_hand = _build_stop_landmarks(is_left=False, is_inverted=False)

    rule_res = rule_rec.predict(np.zeros(68, dtype=np.float32), hand_landmarks=stop_hand)
    assert rule_res.label == "STOP"
    assert rule_res.confidence == 1.0

    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    hybrid = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et, rule_recognizer=rule_rec)
    res = hybrid.predict(np.zeros(68, dtype=np.float32), hand_landmarks=stop_hand, is_stop_armed=False)
    assert res.label == "STOP"
    assert res.model_used == "RULE_BASED"


def test_26_inverted_stop_recognition():
    """26. Verify RuleBasedRecognizer and HybridRecognizer recognize inverted/back-of-hand STOP gesture."""
    rule_rec = RuleBasedRecognizer()
    inv_stop_hand = _build_stop_landmarks(is_left=False, is_inverted=True)

    rule_res = rule_rec.predict(np.zeros(68, dtype=np.float32), hand_landmarks=inv_stop_hand)
    assert rule_res.label == "STOP"
    assert rule_res.confidence == 1.0

    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    hybrid = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et, rule_recognizer=rule_rec)
    res = hybrid.predict(np.zeros(68, dtype=np.float32), hand_landmarks=inv_stop_hand, is_stop_armed=False)
    assert res.label == "STOP"
    assert res.model_used == "RULE_BASED"


def test_27_ordinary_palm_to_no_command():
    """27. Verify HaGRID prediction 'palm' maps strictly to NO_COMMAND and never triggers STOP."""
    assert map_hagrid_to_neurogrip("palm") == NO_COMMAND

    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_hagrid.predict.return_value = {"raw_class": "palm", "mapped_command": NO_COMMAND, "confidence": 0.98, "top_k": []}

    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True
    mock_rule = MagicMock(spec=RuleBasedRecognizer)
    mock_rule.finger_detector = RuleBasedRecognizer().finger_detector
    mock_rule.predict.return_value = RecognitionResult(label="UNKNOWN", confidence=0.0, recognizer_type="rule_based")

    hybrid = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et, rule_recognizer=mock_rule)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_hand = _build_index_only_landmarks()
    # Modify dummy_hand so routing gate returns None
    dummy_hand.landmarks[8] = NormalizedLandmark(x=0.45, y=0.55, z=0.0)

    res = hybrid.predict(np.zeros(68, dtype=np.float32), hand_landmarks=dummy_hand, is_stop_armed=True, raw_frame=dummy_frame)
    assert res.label == NO_COMMAND
    assert res.model_used == "HAGRID"


def test_28_stop_canonical_command_emission():
    """28. Verify STOP gesture prediction produces STOP pose and emits STOP command on transition."""
    stop_hand = _build_stop_landmarks()
    rule_rec = RuleBasedRecognizer()

    mock_hagrid = MagicMock(spec=HaGRIDClassifier)
    mock_hagrid.is_loaded = True
    mock_et = MagicMock(spec=MLRecognizer)
    mock_et.is_ready = True

    hybrid = HybridRecognizer(hagrid_classifier=mock_hagrid, ml_recognizer=mock_et, rule_recognizer=rule_rec)
    stabilizer = TemporalStabilizer()

    rec_res = hybrid.predict(np.zeros(68, dtype=np.float32), hand_landmarks=stop_hand)
    assert rec_res.label == "STOP"

    stab_res = stabilizer.update(rec_res)
    assert stab_res.state == StabilizerState.STOP
    assert stab_res.stable_command == "STOP"
    assert stab_res.emitted_command == "STOP"


def test_29_zero_hand_and_two_hand_safety_behavior():
    """29. Verify 0 hands and 2+ hands output NO_COMMAND with 0 transport emissions."""
    mock_camera = MockCamera()
    mock_transport = MagicMock(spec=CommandTransport)

    pipeline = NeuroGripPipeline(camera=mock_camera, transport=mock_transport)
    pipeline.initialize()

    # 0 Hands
    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[], num_hands=0)
        _, _, viz_state_0 = pipeline.process_frame()

    assert viz_state_0.command == "NO_COMMAND"
    assert mock_transport.send_idle.called
    assert not mock_transport.send_command.called

    mock_transport.reset_mock()

    # 2+ Hands (Ambiguous)
    hand1 = _build_stop_landmarks()
    hand2 = _build_stop_landmarks()
    with patch.object(pipeline.detector, "detect") as mock_detect:
        mock_detect.return_value = DetectionResult(hands=[hand1, hand2], num_hands=2)
        _, _, viz_state_2 = pipeline.process_frame()

    assert viz_state_2.command == "NO_COMMAND"
    assert viz_state_2.is_ambiguous is True
    assert mock_transport.send_idle.called
    assert not mock_transport.send_command.called


