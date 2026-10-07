"""
scripts/collect_dataset.py
───────────────────────────
Dataset Collector for NeuroGrip.
Supports Feature Version v2 (default 68-D vector to pc/data/raw_v2_pilot/)
and Feature Version v1 (legacy 67-D vector to pc/data/raw/).

Flow:
Camera -> HandDetector -> FeatureExtractor -> RuleBasedRecognizer -> Dataset Validation -> CSV Writer

Usage:
    python pc/scripts/collect_dataset.py --label INDEX --samples 100
    python pc/scripts/collect_dataset.py --label STOP --samples 100 --output-dir pc/data/raw_v2_pilot
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# Ensure src directory is in sys.path if run directly
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurogrip.camera.base import CameraInterface
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, configure_logging
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.detector import HandDetector
from neurogrip.hand_tracking.landmarks import DetectionResult
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from neurogrip.visualization.base import VisualizationState
from neurogrip.visualization.overlay import OverlayWindow

logger = logging.getLogger("neurogrip.collect_dataset")


class DatasetSampleValidator:
    """
    Validates candidate dataset samples against strict quality and schema constraints.
    """

    @staticmethod
    def validate(
        detection_res: Optional[DetectionResult],
        features: Optional[np.ndarray],
        rec_result: Optional[RecognitionResult],
        target_command: NeuroGripCommand,
        expected_dim: int = FeatureExtractor.FEATURE_DIM,
    ) -> Tuple[bool, str]:
        """
        Validate a dataset sample.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, rejection_reason)
        """
        # 1. Detection check: exactly 1 hand required
        if detection_res is None or detection_res.is_empty:
            return False, "NO_HAND: 0 hands detected."

        if detection_res.is_ambiguous or detection_res.num_hands > 1:
            return False, f"AMBIGUOUS: {detection_res.num_hands} hands detected."

        if detection_res.num_hands != 1 or detection_res.primary_hand is None:
            return False, "INVALID_DETECTION: Primary hand missing."

        # 2. Feature vector check: must be expected_dim and all finite values
        if features is None:
            return False, "MISSING_FEATURES: Feature vector is None."

        if len(features) != expected_dim:
            return False, f"INVALID_DIM: Expected {expected_dim} features, got {len(features)}."

        if not np.all(np.isfinite(features)):
            return False, "NON_FINITE_FEATURES: Feature vector contains NaN or Inf."

        # 3. Quality check: reject UNKNOWN predictions
        if rec_result is None or rec_result.is_unknown or rec_result.label.upper() == "UNKNOWN":
            return False, "REJECT_UNKNOWN: Recognizer output is UNKNOWN."

        # 4. Strict Ground-Truth alignment check: recognizer prediction must match target command
        if target_command is not None and rec_result.label.upper() != target_command.value.upper():
            return False, f"REJECT_MISMATCH: Recognizer predicted '{rec_result.label}', expected '{target_command.value}'."

        return True, "VALID"


class DatasetCSVWriter:
    """
    Handles CSV dataset schema formatting and incremental sample writing.
    """

    def __init__(self, output_path: Path, feature_dim: int = FeatureExtractor.FEATURE_DIM) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_dim = feature_dim
        self.HEADER: list[str] = [
            "label",
            "handedness",
            "session_id",
            "sample_id",
        ] + [f"feature_{i}" for i in range(self.feature_dim)]

        self._file = None
        self._writer = None

    def open(self) -> None:
        """Open CSV file for appending and write header if file is new."""
        file_exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        self._file = open(self.output_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(self.HEADER)
            self._file.flush()

    def write_sample(
        self,
        label: str,
        handedness: str,
        session_id: str,
        sample_id: int,
        features: np.ndarray,
    ) -> None:
        """Write a single validated sample row to the CSV file."""
        if self._writer is None or self._file is None:
            raise RuntimeError("DatasetCSVWriter is not open.")

        row = [label, handedness, session_id, sample_id] + [f"{float(val):.6f}" for val in features]
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close underlying file handles cleanly."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            finally:
                self._file = None
                self._writer = None

    def __enter__(self) -> DatasetCSVWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def build_collector_parser() -> argparse.ArgumentParser:
    """Build argument parser for dataset collector CLI."""
    parser = argparse.ArgumentParser(
        prog="collect_dataset",
        description="NeuroGrip Dataset Collector (v2 default / v1 legacy option).",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Target command label to collect (must be one of the locked official commands).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Target number of samples to collect (default: 100).",
    )
    parser.add_argument(
        "--feature-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Feature extractor version to use: 'v2' (68-D, default) or 'v1' (67-D legacy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for raw CSV datasets (default: pc/data/raw_v2_pilot for v2, pc/data/raw for v1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit CSV file path override.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Minimum sampling interval between saved frames in seconds (default: 0.1s).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headlessly without OpenCV visualization window.",
    )
    parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Use MockCamera instead of physical webcam (for automated testing).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file.",
    )
    return parser


def run_collector(
    label: str,
    target_samples: int = 100,
    output_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    interval_s: float = 0.1,
    no_gui: bool = False,
    mock_camera: bool = False,
    config_path: Optional[Path] = None,
    feature_version: str = "v2",
    camera_override: Optional[CameraInterface] = None,
    detector_override: Optional[HandDetector] = None,
    extractor_override: Optional[FeatureExtractor] = None,
    recognizer_override: Optional[GestureRecognizer] = None,
    max_frames: Optional[int] = None,
) -> int:
    """
    Execute dataset collection loop.
    """
    # 1. Validate target label
    target_cmd = NeuroGripCommand.from_string(label)
    if target_cmd is None:
        valid_cmds = ", ".join([c.value for c in NeuroGripCommand])
        logger.error("Invalid command label '%s'. Must be one of: %s", label, valid_cmds)
        return 1

    # 2. Load config
    if config_path is None:
        config = AppConfig.default()
    else:
        config = AppConfig.from_yaml(config_path)

    configure_logging(config.logging)

    version_clean = feature_version.lower()
    extractor = extractor_override or FeatureExtractor(config=config.features, version=version_clean)
    feature_dim = 67 if version_clean == "v1" else 68

    # 3. Setup output CSV path
    session_id = uuid.uuid4().hex[:8]
    if output_file is not None:
        csv_path = Path(output_file)
    else:
        if output_dir is not None:
            out_dir = Path(output_dir)
        else:
            if version_clean == "v2":
                out_dir = Path(__file__).resolve().parent.parent / "data" / "raw_v2_pilot"
            else:
                out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

        if version_clean == "v2":
            csv_path = out_dir / f"dataset_v2_{target_cmd.value.lower()}_{session_id}.csv"
        else:
            csv_path = out_dir / f"dataset_{target_cmd.value.lower()}_{session_id}.csv"

    # 4. Initialize components
    if camera_override is not None:
        camera = camera_override
    elif mock_camera or config.camera.backend == "mock":
        camera = MockCamera(config=config.camera)
    else:
        camera = OpenCVCamera(config=config.camera)

    detector = detector_override or HandDetector(config=config.hand_tracking)
    recognizer = recognizer_override or RuleBasedRecognizer(config=config)
    overlay = OverlayWindow(config=config.visualization) if not no_gui and config.visualization.enabled else None

    # STOP safety path is ALWAYS disarmed by default during dataset collection
    is_stop_armed = False

    logger.info("Starting Dataset Collector...")
    logger.info("  Target Label     : %s", target_cmd.value)
    logger.info("  Target Samples   : %d", target_samples)
    logger.info("  Feature Version  : %s (%d-D)", version_clean, feature_dim)
    logger.info("  Sampling Rate    : %.2f s/sample", interval_s)
    logger.info("  Output CSV       : %s", csv_path)
    logger.info("  Session ID       : %s", session_id)
    logger.info("  STOP Armed       : %s", is_stop_armed)

    if not camera.open():
        logger.error("Failed to open camera.")
        return 1

    try:
        detector.initialize()
    except Exception as e:
        logger.error("Failed to initialize HandDetector: %s", e)
        camera.close()
        return 1

    writer = DatasetCSVWriter(csv_path, feature_dim=feature_dim)
    writer.open()

    samples_collected = 0
    total_frames_processed = 0
    last_sample_time = 0.0

    try:
        while samples_collected < target_samples:
            if max_frames is not None and total_frames_processed >= max_frames:
                logger.info("Max frames reached (%d). Exiting collection loop.", max_frames)
                break

            frame_container = camera.read_frame()
            if not frame_container.is_valid or frame_container.frame is None:
                total_frames_processed += 1
                time.sleep(0.001)
                continue

            total_frames_processed += 1
            raw_frame = frame_container.frame
            timestamp_ms = frame_container.timestamp_ms
            detection_res = detector.detect(raw_frame, timestamp_ms)

            features = None
            rec_res = None
            primary_hand = None
            is_sample_valid = False
            status_reason = "NO_HAND"

            if detection_res.num_hands == 1 and detection_res.primary_hand is not None:
                primary_hand = detection_res.primary_hand
                features = extractor.extract(primary_hand)
                rec_res = recognizer.predict(features, hand_landmarks=primary_hand, is_stop_armed=is_stop_armed)

                is_sample_valid, status_reason = DatasetSampleValidator.validate(
                    detection_res, features, rec_res, target_cmd, expected_dim=feature_dim
                )

                now = time.time()
                if is_sample_valid and (now - last_sample_time >= interval_s):
                    samples_collected += 1
                    writer.write_sample(
                        label=target_cmd.value,
                        handedness=primary_hand.handedness.value,
                        session_id=session_id,
                        sample_id=samples_collected,
                        features=features,
                    )
                    last_sample_time = now
                    logger.info(
                        "Collected sample %d/%d [%s, %s]",
                        samples_collected,
                        target_samples,
                        target_cmd.value,
                        primary_hand.handedness.value,
                    )

            # Render HUD and handle GUI inputs
            if overlay is not None:
                viz_state = VisualizationState(
                    command=rec_res.label if rec_res else "NO_HAND",
                    confidence=rec_res.confidence if rec_res else 0.0,
                    stabilizer_state="SAVED" if is_sample_valid else status_reason,
                    pipeline_state=f"COLLECTING ({samples_collected}/{target_samples})",
                    handedness=primary_hand.handedness.value if primary_hand else "UNKNOWN",
                    landmarks=[primary_hand] if primary_hand else [],
                    fps=30.0,
                    is_stop_active=is_stop_armed,
                    is_ambiguous=detection_res.is_ambiguous if detection_res else False,
                    message=f"Target: {target_cmd.value} [{samples_collected}/{target_samples}]",
                )
                key = overlay.show(raw_frame, viz_state)
                if key in (ord("q"), 27):
                    logger.info("User requested exit via window close or ESC key.")
                    break

    finally:
        writer.close()
        detector.close()
        camera.close()
        if overlay is not None:
            overlay.close()

    logger.info(
        "Dataset collection finished. Collected %d/%d samples to '%s'.",
        samples_collected,
        target_samples,
        csv_path,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point."""
    parser = build_collector_parser()
    args = parser.parse_args(argv)

    return run_collector(
        label=args.label,
        target_samples=args.samples,
        output_dir=args.output_dir,
        output_file=args.output,
        interval_s=args.interval,
        no_gui=args.no_gui,
        mock_camera=args.mock_camera,
        config_path=args.config,
        feature_version=args.feature_version,
    )


if __name__ == "__main__":
    sys.exit(main())
