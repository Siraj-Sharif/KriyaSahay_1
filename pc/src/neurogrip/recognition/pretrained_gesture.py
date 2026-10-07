"""
pc/src/neurogrip/recognition/pretrained_gesture.py
────────────────────────────────────────────────────
Phase 10K — Pretrained Gesture Recognizer Adapter & HaGRID Mapping Layer.
ISOLATED EXPERIMENTAL MODULE.
Does NOT replace or modify MLRecognizer or HybridRecognizer.
Does NOT send active hardware/serial/TCP commands.

Encapsulates open-source pretrained hand gesture models (MediaPipe Tasks GestureRecognizer / HaGRID weights).
Maps pretrained gesture taxonomy to NeuroGrip's 13 official gesture commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, resolve_model_path
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult

logger = logging.getLogger(__name__)


class PretrainedClass(Enum):
    """Native category labels supported by HaGRID / MediaPipe pretrained gesture models."""
    UNRECOGNIZED = "Unrecognized"
    CLOSED_FIST = "Closed_Fist"
    OPEN_PALM = "Open_Palm"
    POINTING_UP = "Pointing_Up"
    THUMB_DOWN = "Thumb_Down"
    THUMB_UP = "Thumb_Up"
    VICTORY = "Victory"
    ILOVEYOU = "ILoveYou"
    # Extended HaGRID taxonomy
    FOUR = "four"
    THREE = "three"
    OK = "ok"
    CALL = "call"


@dataclass(frozen=True)
class PretrainedRecognitionResult:
    """
    Experimental result container for pretrained gesture recognition inference.
    """
    raw_pretrained_class: str
    pretrained_confidence: float
    mapped_neurogrip_command: str
    correspondence_level: str  # "DIRECT", "APPROXIMATE", "NONE"
    handedness: str
    handedness_confidence: float
    inference_latency_ms: float
    is_hand_detected: bool = True
    reason: Optional[str] = None


class HaGRIDNeuroGripMapper:
    """
    Taxonomy mapping layer translating HaGRID / MediaPipe pretrained gesture categories
    into NeuroGrip's 13 official command labels.
    """

    # Direct mapping dictionary
    DIRECT_MAPPING: dict[str, str] = {
        "Pointing_Up": NeuroGripCommand.INDEX.value,
        "one": NeuroGripCommand.INDEX.value,
        "Victory": NeuroGripCommand.TWO_FINGER.value,
        "peace": NeuroGripCommand.TWO_FINGER.value,
        "two_up": NeuroGripCommand.TWO_FINGER.value,
        "three": NeuroGripCommand.THREE_FINGER.value,
        "four": NeuroGripCommand.FOUR_FINGERS.value,
        "Closed_Fist": NeuroGripCommand.CLOSE.value,
        "fist": NeuroGripCommand.CLOSE.value,
        "Open_Palm": NeuroGripCommand.STOP.value,  # Display only; non-authoritative
        "palm": NeuroGripCommand.STOP.value,
        "stop": NeuroGripCommand.STOP.value,
        "Unrecognized": NeuroGripCommand.REST.value,
        "no_gesture": NeuroGripCommand.REST.value,
        "neutral": NeuroGripCommand.REST.value,
    }

    # Approximate mapping dictionary
    APPROX_MAPPING: dict[str, str] = {
        "Thumb_Up": NeuroGripCommand.THUMB_ONLY.value,
        "like": NeuroGripCommand.THUMB_ONLY.value,
        "Thumb_Down": NeuroGripCommand.THUMB_ONLY.value,
        "dislike": NeuroGripCommand.THUMB_ONLY.value,
        "ILoveYou": NeuroGripCommand.INDEX_PINKY.value,
        "rock": NeuroGripCommand.INDEX_PINKY.value,
        "ok": NeuroGripCommand.GRAB.value,
        "call": NeuroGripCommand.PINKY.value,  # Approximate (includes thumb)
    }

    # Gestures missing from pretrained models
    UNSUPPORTED_NEUROGRIP_GESTURES: list[str] = [
        NeuroGripCommand.MIDDLE.value,
        NeuroGripCommand.RING.value,
        NeuroGripCommand.PINKY.value,
    ]

    @classmethod
    def map_class(cls, pretrained_label: str) -> tuple[str, str]:
        """
        Map a pretrained gesture label to NeuroGrip label and correspondence level.

        Returns
        -------
        tuple[str, str]
            (mapped_neurogrip_command, correspondence_level)
            correspondence_level: "DIRECT", "APPROXIMATE", or "NONE"
        """
        lbl_clean = pretrained_label.strip()

        if lbl_clean in cls.DIRECT_MAPPING:
            return cls.DIRECT_MAPPING[lbl_clean], "DIRECT"

        if lbl_clean in cls.APPROX_MAPPING:
            return cls.APPROX_MAPPING[lbl_clean], "APPROXIMATE"

        # Search case-insensitive
        for k, v in cls.DIRECT_MAPPING.items():
            if k.lower() == lbl_clean.lower():
                return v, "DIRECT"

        for k, v in cls.APPROX_MAPPING.items():
            if k.lower() == lbl_clean.lower():
                return v, "APPROXIMATE"

        return "UNKNOWN", "NONE"


class PretrainedGestureRecognizer:
    """
    Experimental recognizer wrapping pretrained HaGRID / MediaPipe gesture models.
    Operates in VIDEO mode for real-time webcam frame processing.
    """

    def __init__(self, model_path: Optional[str | Path] = None, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.default()
        default_p = Path(__file__).resolve().parent.parent.parent.parent / "models" / "mediapipe" / "gesture_recognizer.task"
        self.model_path = resolve_model_path(model_path or default_p)

        self._recognizer: Optional[vision.GestureRecognizer] = None
        self._is_ready: bool = False
        self.mapper = HaGRIDNeuroGripMapper()

        if self.model_path.exists():
            self.initialize()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def initialize(self) -> bool:
        """Initialize MediaPipe Tasks GestureRecognizer instance."""
        if self._is_ready:
            return True

        if not self.model_path.exists():
            logger.warning("Pretrained gesture model not found at '%s'.", self.model_path)
            self._is_ready = False
            return False

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.config.hand_tracking.num_hands,
                min_hand_detection_confidence=self.config.hand_tracking.min_hand_detection_confidence,
                min_hand_presence_confidence=self.config.hand_tracking.min_hand_presence_confidence,
                min_tracking_confidence=self.config.hand_tracking.min_tracking_confidence,
            )
            self._recognizer = vision.GestureRecognizer.create_from_options(options)
            self._is_ready = True
            logger.info("PretrainedGestureRecognizer initialized successfully from '%s'.", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to initialize PretrainedGestureRecognizer: %s", e)
            self._is_ready = False
            return False

    def predict_frame(self, frame: np.ndarray, timestamp_ms: int) -> PretrainedRecognitionResult:
        """
        Run inference on an OpenCV BGR frame and return clean diagnostic predictions.
        """
        if not self._is_ready or self._recognizer is None:
            self.initialize()
            if not self._is_ready or self._recognizer is None:
                return PretrainedRecognitionResult(
                    raw_pretrained_class="UNINITIALIZED",
                    pretrained_confidence=0.0,
                    mapped_neurogrip_command="UNKNOWN",
                    correspondence_level="NONE",
                    handedness="UNKNOWN",
                    handedness_confidence=0.0,
                    inference_latency_ms=0.0,
                    is_hand_detected=False,
                    reason="MODEL_UNINITIALIZED",
                )

        if frame is None or frame.size == 0:
            return PretrainedRecognitionResult(
                raw_pretrained_class="NO_FRAME",
                pretrained_confidence=0.0,
                mapped_neurogrip_command="UNKNOWN",
                correspondence_level="NONE",
                handedness="UNKNOWN",
                handedness_confidence=0.0,
                inference_latency_ms=0.0,
                is_hand_detected=False,
                reason="EMPTY_FRAME",
            )

        # Convert BGR frame to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        t0 = time.perf_counter()
        try:
            mp_res = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.error("Error during recognize_for_video: %s", e)
            return PretrainedRecognitionResult(
                raw_pretrained_class="ERROR",
                pretrained_confidence=0.0,
                mapped_neurogrip_command="UNKNOWN",
                correspondence_level="NONE",
                handedness="UNKNOWN",
                handedness_confidence=0.0,
                inference_latency_ms=0.0,
                is_hand_detected=False,
                reason=str(e),
            )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if not mp_res.gestures or len(mp_res.gestures) == 0 or len(mp_res.gestures[0]) == 0:
            return PretrainedRecognitionResult(
                raw_pretrained_class="No_Hand",
                pretrained_confidence=0.0,
                mapped_neurogrip_command="UNKNOWN",
                correspondence_level="NONE",
                handedness="UNKNOWN",
                handedness_confidence=0.0,
                inference_latency_ms=latency_ms,
                is_hand_detected=False,
                reason="NO_HAND_DETECTED",
            )

        top_gesture = mp_res.gestures[0][0]
        raw_class = top_gesture.category_name or "Unrecognized"
        conf = float(top_gesture.score)

        handedness_str = "UNKNOWN"
        h_score = 1.0
        if mp_res.handedness and len(mp_res.handedness) > 0 and len(mp_res.handedness[0]) > 0:
            h_cat = mp_res.handedness[0][0]
            handedness_str = Handedness.from_str(h_cat.category_name).value
            h_score = float(getattr(h_cat, "score", 1.0))

        mapped_cmd, corr_level = self.mapper.map_class(raw_class)

        return PretrainedRecognitionResult(
            raw_pretrained_class=raw_class,
            pretrained_confidence=conf,
            mapped_neurogrip_command=mapped_cmd,
            correspondence_level=corr_level,
            handedness=handedness_str,
            handedness_confidence=h_score,
            inference_latency_ms=latency_ms,
            is_hand_detected=True,
            reason=None,
        )

    def close(self) -> None:
        """Release MediaPipe recognizer resources."""
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None
        self._is_ready = False
        logger.info("PretrainedGestureRecognizer closed.")

    def __enter__(self) -> "PretrainedGestureRecognizer":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
