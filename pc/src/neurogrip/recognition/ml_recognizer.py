"""
neurogrip/recognition/ml_recognizer.py
───────────────────────────────────────
ML-based Primary Gesture Recognizer.
Loads a locally trained scikit-learn model bundle (.pkl) and performs CPU-based real-time inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, RecognitionConfig, resolve_model_path
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.landmarks import HandLandmarks
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult

logger = logging.getLogger(__name__)


class MLRecognizerError(Exception):
    """Raised when model bundle loading or validation fails."""
    pass


class MLRecognizer(GestureRecognizer):
    """
    Primary machine-learning gesture recognizer using scikit-learn (RandomForest or MLP).
    Loads a joblib serialized model bundle from local storage.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.default()
        self.rec_cfg: RecognitionConfig = self.config.recognition

        self._classifier: Any = None
        self._preprocessor: Any = None
        self._classes: list[str] = []
        self._is_ready: bool = False
        self._recognizer_type: str = "ml_uninitialized"

        # Attempt to load model at construction if file exists
        if self.rec_cfg.model_path:
            self.load_model(self.rec_cfg.model_path)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_model(self, model_path: str | Path) -> bool:
        """
        Load and validate a joblib model bundle (.pkl).

        Returns
        -------
        bool
            True if model loaded and validated successfully, False otherwise.
        """
        path = resolve_model_path(model_path)


        if not path.exists():
            logger.warning("ML model bundle not found at '%s'. Recognizer marked NOT ready.", path)
            self._is_ready = False
            return False

        try:
            bundle = joblib.load(path)
            self._validate_bundle(bundle, path)

            self._classifier = bundle["classifier"]
            self._preprocessor = bundle.get("preprocessor")
            self._classes = list(bundle["classes"])

            classifier_name = type(self._classifier).__name__.lower()
            if "randomforest" in classifier_name:
                self._recognizer_type = "ml_random_forest"
            elif "mlp" in classifier_name:
                self._recognizer_type = "ml_mlp"
            else:
                self._recognizer_type = f"ml_{classifier_name}"

            self._is_ready = True
            logger.info("MLRecognizer loaded '%s' successfully (%s).", path.name, self._recognizer_type)
            return True

        except Exception as e:
            logger.error("Failed to load ML model bundle from '%s': %s", path, e)
            self._is_ready = False
            return False

        return False

    def _validate_bundle(self, bundle: Any, path: Path) -> None:
        """Validate metadata contained in the model bundle."""
        if not isinstance(bundle, dict):
            raise MLRecognizerError("Model bundle must be a dictionary object.")

        if "classifier" not in bundle:
            raise MLRecognizerError("Model bundle missing 'classifier' key.")

        if "classes" not in bundle:
            raise MLRecognizerError("Model bundle missing 'classes' key.")

        # Check feature version compatibility
        bundle_version = bundle.get("feature_version")
        expected_version = FeatureExtractor.FEATURE_VERSION
        if bundle_version != expected_version:
            raise MLRecognizerError(
                f"Feature version mismatch: bundle has '{bundle_version}', "
                f"expected '{expected_version}'."
            )

        # Check feature dimension compatibility
        bundle_dim = bundle.get("feature_dim")
        expected_dim = FeatureExtractor.FEATURE_DIM
        if bundle_dim != expected_dim:
            raise MLRecognizerError(
                f"Feature dimension mismatch: bundle has {bundle_dim}, "
                f"expected {expected_dim}."
            )

    def predict(
        self,
        features: np.ndarray,
        hand_landmarks: Optional[HandLandmarks] = None,
        is_stop_armed: bool = False,
        raw_frame: Optional[np.ndarray] = None,
    ) -> RecognitionResult:
        """
        Predict a gesture command from a 67-dimensional feature vector.
        """
        if not self._is_ready or self._classifier is None:
            return RecognitionResult(
                label="UNKNOWN",
                confidence=0.0,
                recognizer_type="ml_uninitialized",
            )

        if features is None or len(features) != FeatureExtractor.FEATURE_DIM:
            logger.warning(
                "Invalid feature vector passed to MLRecognizer. Expected dim %d.",
                FeatureExtractor.FEATURE_DIM,
            )
            return RecognitionResult(
                label="UNKNOWN",
                confidence=0.0,
                recognizer_type=self._recognizer_type,
            )

        # Preprocess features (e.g. StandardScaler if required)
        feat_input = features.reshape(1, -1)
        if self._preprocessor is not None:
            try:
                feat_input = self._preprocessor.transform(feat_input)
            except Exception as e:
                logger.error("Error preprocessing feature vector: %s", e)
                return RecognitionResult(
                    label="UNKNOWN",
                    confidence=0.0,
                    recognizer_type=self._recognizer_type,
                )

        # Compute prediction probabilities
        try:
            if hasattr(self._classifier, "predict_proba"):
                probs = self._classifier.predict_proba(feat_input)[0]
                top_idx = int(np.argmax(probs))
                raw_label = str(self._classes[top_idx])
                confidence = float(probs[top_idx])
                all_scores = {str(cls_name): float(p) for cls_name, p in zip(self._classes, probs)}
            else:
                top_idx = int(self._classifier.predict(feat_input)[0])
                raw_label = str(self._classes[top_idx])
                confidence = 1.0
                all_scores = {raw_label: 1.0}
        except Exception as e:
            logger.error("Error during ML model inference: %s", e)
            return RecognitionResult(
                label="UNKNOWN",
                confidence=0.0,
                recognizer_type=self._recognizer_type,
            )

        # Apply confidence gate
        min_conf = self.rec_cfg.normal_min_confidence
        if confidence < min_conf:
            final_label = "UNKNOWN"
        else:
            final_label = raw_label

        # Apply STOP software-armed security restriction:
        # If classifier predicts STOP but STOP detection is NOT armed in software, override to UNKNOWN
        if final_label == NeuroGripCommand.STOP.value and not is_stop_armed:
            logger.debug("STOP predicted by ML model but STOP detection is NOT armed. Overriding to UNKNOWN.")
            final_label = "UNKNOWN"

        return RecognitionResult(
            label=final_label,
            confidence=confidence,
            all_scores=all_scores,
            recognizer_type=self._recognizer_type,
        )
