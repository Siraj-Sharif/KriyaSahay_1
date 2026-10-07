"""
tests/test_dataset_collector.py
────────────────────────────────
Unit tests for Phase 10A Dataset Collector script, schema writer, and validation filters.
Supports testing for Feature Version v2 pilot collection as well as legacy v1 compatibility.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts directory is accessible for importing collect_dataset
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from collect_dataset import DatasetCSVWriter, DatasetSampleValidator, build_collector_parser, run_collector

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.detector import HandDetector
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult


def create_dummy_hand_landmarks(handedness: Handedness = Handedness.RIGHT) -> HandLandmarks:
    """Helper fixture to create a valid 21-landmark HandLandmarks instance."""
    lms = [NormalizedLandmark(x=0.5 + i * 0.01, y=0.5, z=0.0) for i in range(21)]
    return HandLandmarks(landmarks=lms, handedness=handedness)


class MockValidDetector(HandDetector):
    """Mock HandDetector returning 1 valid hand landmark set for unit testing."""

    def initialize(self) -> None:
        self._is_initialized = True

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> DetectionResult:
        hand = create_dummy_hand_landmarks(Handedness.RIGHT)
        return DetectionResult(hands=[hand], num_hands=1, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self._is_initialized = False


class MockLabelRecognizer(GestureRecognizer):
    """Mock recognizer returning a configurable prediction label."""

    def __init__(self, label: str = "CLOSE") -> None:
        self.label = label

    @property
    def is_ready(self) -> bool:
        return True

    def predict(self, features, hand_landmarks=None, is_stop_armed=False):
        return RecognitionResult(label=self.label, confidence=1.0)


def test_dataset_validator_valid_sample():
    hand = create_dummy_hand_landmarks(Handedness.RIGHT)
    det_res = DetectionResult(hands=[hand], num_hands=1)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="INDEX", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is True
    assert reason == "VALID"


def test_dataset_validator_rejects_different_recognizer_label():
    """Ground truth alignment requirement: validator rejects sample if recognizer output differs from human target label."""
    hand = create_dummy_hand_landmarks(Handedness.RIGHT)
    det_res = DetectionResult(hands=[hand], num_hands=1)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    # Recognizer predicts CLOSE, but human ground truth label is INDEX
    rec_res = RecognitionResult(label="CLOSE", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "REJECT_MISMATCH" in reason


def test_dataset_validator_reject_no_hand():
    det_res = DetectionResult(hands=[], num_hands=0)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="INDEX", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "NO_HAND" in reason


def test_dataset_validator_reject_ambiguous():
    h1 = create_dummy_hand_landmarks(Handedness.RIGHT)
    h2 = create_dummy_hand_landmarks(Handedness.LEFT)
    det_res = DetectionResult(hands=[h1, h2], num_hands=2)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="INDEX", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "AMBIGUOUS" in reason


def test_dataset_validator_reject_non_finite_features():
    hand = create_dummy_hand_landmarks()
    det_res = DetectionResult(hands=[hand], num_hands=1)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    features[10] = np.nan
    rec_res = RecognitionResult(label="INDEX", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "NON_FINITE_FEATURES" in reason


def test_dataset_validator_reject_invalid_dim():
    hand = create_dummy_hand_landmarks()
    det_res = DetectionResult(hands=[hand], num_hands=1)
    features = np.zeros(50, dtype=np.float32)
    rec_res = RecognitionResult(label="INDEX", confidence=1.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "INVALID_DIM" in reason


def test_dataset_validator_reject_unknown():
    hand = create_dummy_hand_landmarks()
    det_res = DetectionResult(hands=[hand], num_hands=1)
    features = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    rec_res = RecognitionResult(label="UNKNOWN", confidence=0.0)
    target_cmd = NeuroGripCommand.INDEX

    is_valid, reason = DatasetSampleValidator.validate(det_res, features, rec_res, target_cmd)
    assert is_valid is False
    assert "REJECT_UNKNOWN" in reason


def test_dataset_csv_writer(tmp_path):
    csv_file = tmp_path / "test_dataset.csv"

    with DatasetCSVWriter(csv_file) as writer:
        features = np.full(FeatureExtractor.FEATURE_DIM, 0.123456, dtype=np.float32)
        writer.write_sample(
            label="INDEX",
            handedness="RIGHT",
            session_id="test_sess",
            sample_id=1,
            features=features,
        )

    assert csv_file.exists()

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 2  # Header + 1 sample row
        header = reader[0]
        row = reader[1]

        assert len(header) == 4 + FeatureExtractor.FEATURE_DIM
        assert header[0:4] == ["label", "handedness", "session_id", "sample_id"]
        assert header[4] == "feature_0"
        assert header[-1] == f"feature_{FeatureExtractor.FEATURE_DIM - 1}"

        assert row[0] == "INDEX"
        assert row[1] == "RIGHT"
        assert row[2] == "test_sess"
        assert row[3] == "1"
        assert len(row[4:]) == FeatureExtractor.FEATURE_DIM


def test_dataset_collector_cli_parser():
    parser = build_collector_parser()
    args = parser.parse_args(["--label", "INDEX", "--samples", "100", "--interval", "0.05", "--no-gui"])
    assert args.label == "INDEX"
    assert args.samples == 100
    assert args.interval == 0.05
    assert args.no_gui is True
    assert args.feature_version == "v2"


def test_v2_pilot_collector_defaults(tmp_path):
    """Verify v2 pilot dataset collection output format, path, schema, session, and sample_id."""
    pilot_dir = tmp_path / "raw_v2_pilot"
    mock_detector = MockValidDetector()
    mock_recognizer = MockLabelRecognizer("INDEX")

    ret = run_collector(
        label="INDEX",
        target_samples=3,
        output_dir=pilot_dir,
        feature_version="v2",
        interval_s=0.001,
        no_gui=True,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=mock_recognizer,
        max_frames=10,
    )
    assert ret == 0

    assert pilot_dir.exists()

    csv_files = list(pilot_dir.glob("dataset_v2_index_*.csv"))
    assert len(csv_files) == 1
    csv_file = csv_files[0]

    with open(csv_file, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

        # 72 columns: 4 metadata + 68 features
        header = rows[0]
        assert len(header) == 72
        assert header[0:4] == ["label", "handedness", "session_id", "sample_id"]
        assert header[4] == "feature_0"
        assert header[-1] == "feature_67"

        # 3 valid samples written
        assert len(rows) == 4

        # session_id is non-empty and uniform
        sess_id = rows[1][2]
        assert len(sess_id) > 0
        assert rows[2][2] == sess_id
        assert rows[3][2] == sess_id

        # sample_id begins at 1 and increments
        assert rows[1][3] == "1"
        assert rows[2][3] == "2"
        assert rows[3][3] == "3"

        # Human label is ground truth
        assert rows[1][0] == "INDEX"
        assert rows[2][0] == "INDEX"
        assert rows[3][0] == "INDEX"

        # Exactly 68 features in row
        assert len(rows[1][4:]) == 68


def test_v1_legacy_collector_compatibility(tmp_path):
    """Verify passing feature_version='v1' uses raw dir and legacy dataset_ naming."""
    raw_v1_dir = tmp_path / "raw"
    mock_detector = MockValidDetector()
    mock_recognizer = MockLabelRecognizer("INDEX")

    ret = run_collector(
        label="INDEX",
        target_samples=2,
        output_dir=raw_v1_dir,
        feature_version="v1",
        interval_s=0.001,
        no_gui=True,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=mock_recognizer,
        max_frames=5,
    )
    assert ret == 0
    assert raw_v1_dir.exists()

    csv_files = list(raw_v1_dir.glob("dataset_index_*.csv"))
    assert len(csv_files) == 1
    csv_file = csv_files[0]

    with open(csv_file, mode="r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 3  # Header + 2 samples
        assert len(rows[0]) == 71  # 4 metadata + 67 features
        assert rows[0][-1] == "feature_66"


def test_unique_session_id_per_invocation(tmp_path):
    """Verify each run_collector invocation generates a unique session_id."""
    pilot_dir = tmp_path / "raw_v2_pilot"
    mock_detector = MockValidDetector()
    mock_recognizer = MockLabelRecognizer("INDEX")

    run_collector(
        label="INDEX",
        target_samples=1,
        output_dir=pilot_dir,
        feature_version="v2",
        interval_s=0.001,
        no_gui=True,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=mock_recognizer,
        max_frames=3,
    )

    run_collector(
        label="INDEX",
        target_samples=1,
        output_dir=pilot_dir,
        feature_version="v2",
        interval_s=0.001,
        no_gui=True,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=mock_recognizer,
        max_frames=3,
    )

    csv_files = list(pilot_dir.glob("dataset_v2_index_*.csv"))
    assert len(csv_files) == 2  # Two distinct session files created

    sess_ids = set()
    for f_path in csv_files:
        with open(f_path, "r", encoding="utf-8") as f:
            r = list(csv.reader(f))
            sess_ids.add(r[1][2])

    assert len(sess_ids) == 2


def test_stop_collection_starts_disarmed(tmp_path):
    """Verify STOP collection starts with is_stop_armed = False."""
    out_file = tmp_path / "stop_dataset.csv"
    mock_detector = MockValidDetector()

    stop_armed_observations = []

    class StopCheckRecognizer(GestureRecognizer):
        @property
        def is_ready(self) -> bool:
            return True

        def predict(self, features, hand_landmarks=None, is_stop_armed=False):
            stop_armed_observations.append(is_stop_armed)
            return RecognitionResult(label="REST", confidence=1.0)

    ret = run_collector(
        label="STOP",
        target_samples=2,
        output_file=out_file,
        interval_s=0.001,
        no_gui=True,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=StopCheckRecognizer(),
        max_frames=5,
    )
    assert ret == 0
    assert len(stop_armed_observations) > 0
    assert all(armed is False for armed in stop_armed_observations)


def test_visualization_state_constructor_compatibility(tmp_path):
    """Regression test: verify VisualizationState constructor compatibility during collector GUI loop execution."""
    out_file = tmp_path / "viz_test.csv"
    mock_detector = MockValidDetector()
    mock_recognizer = MockLabelRecognizer("INDEX")

    # 1. Direct constructor test with all keyword arguments used by collect_dataset
    from neurogrip.visualization.base import VisualizationState
    viz_state = VisualizationState(
        command="INDEX",
        confidence=1.0,
        stabilizer_state="SAVED",
        pipeline_state="COLLECTING (1/10)",
        handedness="RIGHT",
        landmarks=[create_dummy_hand_landmarks()],
        fps=30.0,
        is_stop_active=False,
        is_ambiguous=False,
        message="Target: INDEX [1/10]",
    )
    assert viz_state.command == "INDEX"
    assert viz_state.confidence == 1.0
    assert viz_state.stabilizer_state == "SAVED"
    assert viz_state.pipeline_state == "COLLECTING (1/10)"
    assert viz_state.handedness == "RIGHT"

    # 2. Integration test running collector loop with no_gui=False (rendering active)
    ret = run_collector(
        label="INDEX",
        target_samples=1,
        output_file=out_file,
        interval_s=0.001,
        no_gui=False,
        mock_camera=True,
        detector_override=mock_detector,
        recognizer_override=mock_recognizer,
        max_frames=3,
    )
    assert ret == 0
    assert out_file.exists()

