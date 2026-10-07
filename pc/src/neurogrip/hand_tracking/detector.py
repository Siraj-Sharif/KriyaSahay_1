"""
neurogrip/hand_tracking/detector.py
───────────────────────────────────
MediaPipe HandLandmarker wrapper using the modern Tasks API in VIDEO mode.
Keeps MediaPipe dependencies encapsulated away from downstream CV modules.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from neurogrip.config.settings import HandTrackingConfig, resolve_model_path
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)

logger = logging.getLogger(__name__)


class HandDetectorError(Exception):
    """Raised when HandDetector initialization or processing fails."""
    pass


class HandDetector:
    """
    Encapsulates MediaPipe Tasks HandLandmarker for real-time video processing.
    Runs in VIDEO mode (synchronous per-frame detection with internal tracking).
    """

    def __init__(self, config: Optional[HandTrackingConfig] = None) -> None:
        self.config = config or HandTrackingConfig()
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """True if the MediaPipe landmarker instance is initialized and ready."""
        return self._is_initialized

    def initialize(self) -> None:
        """Initialize the underlying MediaPipe HandLandmarker instance."""
        if self._is_initialized:
            return

        model_path = resolve_model_path(self.config.model_path)


        if not model_path.exists():
            raise HandDetectorError(
                f"MediaPipe model file not found at '{model_path}'. "
                "Run 'python scripts/download_model.py' to download it."
            )

        try:
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.config.num_hands,
                min_hand_detection_confidence=self.config.min_hand_detection_confidence,
                min_hand_presence_confidence=self.config.min_hand_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            logger.info("HandDetector initialized successfully with model '%s'.", model_path)
        except Exception as e:
            raise HandDetectorError(f"Failed to initialize MediaPipe HandLandmarker: {e}") from e

    def detect(self, frame: np.ndarray, timestamp_ms: int) -> DetectionResult:
        """
        Detect hands in an RGB or BGR OpenCV image frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV image array (BGR or RGB).
        timestamp_ms : int
            Monotonically increasing timestamp in milliseconds.

        Returns
        -------
        DetectionResult
            Clean internal representation of detected hand landmarks.
        """
        if not self._is_initialized or self._landmarker is None:
            self.initialize()
            assert self._landmarker is not None

        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to HandDetector.")
            return DetectionResult(timestamp_ms=timestamp_ms)

        # Convert BGR (default OpenCV format) to RGB if 3 channels
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarker = self._landmarker
        if landmarker is None:
            return DetectionResult(timestamp_ms=timestamp_ms)

        try:
            mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.error("Error during detect_for_video: %s", e)
            return DetectionResult(timestamp_ms=timestamp_ms)

        return self._convert_result(mp_result, timestamp_ms)

    def _convert_result(
        self, mp_result: vision.HandLandmarkerResult, timestamp_ms: int
    ) -> DetectionResult:
        """Convert MediaPipe result to clean internal DetectionResult representation."""
        if not mp_result.hand_landmarks:
            return DetectionResult(timestamp_ms=timestamp_ms)

        detected_hands: list[HandLandmarks] = []
        for i, raw_landmarks in enumerate(mp_result.hand_landmarks):
            converted_landmarks = [
                NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in raw_landmarks
            ]

            handedness_enum = Handedness.UNKNOWN
            score = 1.0
            if mp_result.handedness and i < len(mp_result.handedness):
                categories = mp_result.handedness[i]
                if categories:
                    handedness_enum = Handedness.from_str(categories[0].category_name)
                    score = getattr(categories[0], "score", 1.0)

            world_lms: Optional[list[NormalizedLandmark]] = None
            if mp_result.hand_world_landmarks and i < len(mp_result.hand_world_landmarks):
                raw_world = mp_result.hand_world_landmarks[i]
                world_lms = [NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in raw_world]

            detected_hands.append(
                HandLandmarks(
                    landmarks=converted_landmarks,
                    handedness=handedness_enum,
                    score=score,
                    world_landmarks=world_lms,
                )
            )

        num_hands = len(detected_hands)
        return DetectionResult(
            hands=detected_hands,
            num_hands=num_hands,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        landmarker = self._landmarker
        self._landmarker = None
        self._is_initialized = False
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as e:
                logger.warning("Error closing MediaPipe landmarker: %s", e)
        logger.info("HandDetector closed.")

    def __enter__(self) -> "HandDetector":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
