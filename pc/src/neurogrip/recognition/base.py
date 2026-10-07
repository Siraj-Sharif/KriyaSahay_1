"""
neurogrip/recognition/base.py
──────────────────────────────
Abstract base class and result container for gesture recognizers.
Decoupled from MediaPipe and specific ML frameworks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from neurogrip.hand_tracking.landmarks import HandLandmarks


@dataclass(frozen=True)
class RecognitionResult:
    """
    Standardized result emitted by any GestureRecognizer implementation.

    Attributes
    ----------
    label : str
        Predicted command name (e.g. "INDEX_FINGER", "CLOSED_FIST") or "NO_COMMAND" / "UNKNOWN".
    confidence : float
        Prediction confidence score in range [0.0, 1.0].
    all_scores : dict[str, float]
        Mapping of command names to probability/confidence scores.
    recognizer_type : str
        Identifier of the recognizer that produced this result (e.g. "hybrid_hagrid", "hybrid_extra_trees").
    reason : Optional[str]
        Optional internal classification or rejection reason code.
    model_used : str
        Model architecture identifier ("HAGRID", "EXTRA_TREES", "RULE_BASED", "UNKNOWN").
    raw_prediction : str
        Original raw prediction string before taxonomy mapping.
    """
    label: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)
    recognizer_type: str = "unknown"
    reason: Optional[str] = None
    model_used: str = "UNKNOWN"
    raw_prediction: str = "UNKNOWN"

    @property
    def is_unknown(self) -> bool:
        """True if the prediction is low-confidence or unclassifiable."""
        return self.label.upper() == "UNKNOWN"


class GestureRecognizer(ABC):
    """
    Abstract interface for gesture recognizers.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if the recognizer is initialized and ready to perform inference."""
        pass

    @abstractmethod
    def predict(
        self,
        features: np.ndarray,
        hand_landmarks: Optional[HandLandmarks] = None,
        is_stop_armed: bool = False,
        raw_frame: Optional[np.ndarray] = None,
    ) -> RecognitionResult:
        """
        Predict a gesture command from a 67-dimensional feature vector.

        Parameters
        ----------
        features : np.ndarray
            Shape (67,), float32 feature vector from FeatureExtractor.
        hand_landmarks : Optional[HandLandmarks]
            Optional HandLandmarks object (used by rule-based recognizer for finger states).
        is_stop_armed : bool
            Flag indicating whether STOP detection is currently armed in software.

        Returns
        -------
        RecognitionResult
            Predicted command, confidence score, per-class probabilities, and recognizer type.
        """
        pass
