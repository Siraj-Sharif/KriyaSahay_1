"""
neurogrip/recognition/hybrid.py
───────────────────────────────
Production Hybrid Gesture Recognizer for NeuroGrip.
Integrates:
  1. HaGRID ResNet18 Full-Frame Classifier (Primary model for 11 canonical gestures)
  2. Extra Trees 68-D Feature Classifier (Specialized model for INDEX_FINGER, TWO_FINGERS, PINKY)
  3. Deterministic Rule-Based Safety Detector (Authoritative STOP safety detector)

LOCKED CANONICAL TAXONOMY (13 commands):
  CALL, CLOSED_FIST, FOUR_FINGERS, GRIP, THUMBS_UP, PINKY,
  MIDDLE_FINGER, OK, INDEX_FINGER, INDEX_PINKY, STOP, TWO_FINGERS, THREE_FINGERS.
Internal: NO_COMMAND
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neurogrip.config.settings import AppConfig, resolve_model_path
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import FingerState
from neurogrip.hagrid import (
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    HaGRIDClassifier,
    HaGRIDConfig,
    map_hagrid_to_neurogrip,
)
from neurogrip.hand_tracking.landmarks import HandLandmarks
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult
from neurogrip.recognition.ml_recognizer import MLRecognizer
from neurogrip.recognition.rule_based import RuleBasedRecognizer

logger = logging.getLogger(__name__)

# Extra Trees Raw Class -> Canonical NeuroGrip Command Mapping
EXTRATREES_TO_NEUROGRIP_MAP: Dict[str, str] = {
    "INDEX": "INDEX_FINGER",
    "INDEX_FINGER": "INDEX_FINGER",
    "TWO_FINGER": "TWO_FINGERS",
    "TWO_FINGERS": "TWO_FINGERS",
    "PINKY": "PINKY",
    "CLOSE": "CLOSED_FIST",
    "CLOSED_FIST": "CLOSED_FIST",
    "FOUR_FINGERS": "FOUR_FINGERS",
    "GRAB": "GRIP",
    "INDEX_PINKY": "INDEX_PINKY",
    "MIDDLE": "MIDDLE_FINGER",
    "MIDDLE_FINGER": "MIDDLE_FINGER",
    "THREE_FINGER": "THREE_FINGERS",
    "THREE_FINGERS": "THREE_FINGERS",
    "THUMB_ONLY": "THUMBS_UP",
    "THUMBS_UP": "THUMBS_UP",
    "STOP": "STOP",
    "REST": NO_COMMAND,
    "RING": NO_COMMAND,
    "NO_COMMAND": NO_COMMAND,
}

ROUTED_TARGET_COMMANDS: set[str] = {"INDEX_FINGER", "TWO_FINGERS", "PINKY"}


def map_extratrees_to_neurogrip(raw_label: str) -> str:
    """Map Extra Trees raw prediction label to canonical 13-gesture taxonomy."""
    if not isinstance(raw_label, str):
        return NO_COMMAND
    clean = raw_label.strip().upper()
    return EXTRATREES_TO_NEUROGRIP_MAP.get(clean, NO_COMMAND)


class HybridRecognizer(GestureRecognizer):
    """
    Production Hybrid Gesture Recognizer.
    Loads both HaGRID ResNet18 and Extra Trees 68-D models at startup.
    Routes frames deterministically:
    - HaGRID ResNet18 predicts primary gesture.
    - If HaGRID predicts INDEX_FINGER, TWO_FINGERS, or PINKY -> routes 68-D landmarks to Extra Trees.
    - Deterministic Rule-Based STOP check acts as authoritative safety override.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        hagrid_classifier: Optional[HaGRIDClassifier] = None,
        ml_recognizer: Optional[MLRecognizer] = None,
        rule_recognizer: Optional[RuleBasedRecognizer] = None,
        extra_trees_model_path: Optional[str | Path] = None,
    ) -> None:
        self.config = config or AppConfig.default()
        self.rule_recognizer = rule_recognizer or RuleBasedRecognizer(config=self.config)

        # 1. Initialize HaGRID ResNet18 classifier
        if hagrid_classifier is not None:
            self.hagrid_classifier = hagrid_classifier
        else:
            hagrid_cfg = HaGRIDConfig(model_name="ResNet18")
            self.hagrid_classifier = HaGRIDClassifier(config=hagrid_cfg, auto_load=True)

        # 2. Initialize Extra Trees 68-D MLRecognizer
        if ml_recognizer is not None:
            self.ml_recognizer = ml_recognizer
        else:
            model_p = extra_trees_model_path or "pc/models/NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl"
            self.ml_recognizer = MLRecognizer(config=self.config)
            self.ml_recognizer.load_model(model_p)

        # 3. Verify Extra Trees feature dimension (Must be 68)
        if self.ml_recognizer.is_ready:
            expected_dim = FeatureExtractor.FEATURE_DIM
            if expected_dim != 68:
                logger.error("FeatureExtractor.FEATURE_DIM mismatch! Expected 68, got %d", expected_dim)
                raise ValueError(f"Feature dimension mismatch: expected 68, got {expected_dim}")

        logger.info(
            "HybridRecognizer initialized: HaGRID loaded=%s, ExtraTrees ready=%s (68-D).",
            self.hagrid_classifier.is_loaded,
            self.ml_recognizer.is_ready,
        )

    @property
    def is_ready(self) -> bool:
        return self.hagrid_classifier.is_loaded or self.ml_recognizer.is_ready or self.rule_recognizer.is_ready

    def _check_landmark_routing_gate(self, hand_landmarks: Optional[HandLandmarks]) -> Optional[str]:
        """
        Evaluate candidate physical postures from MediaPipe landmarks BEFORE model inference:
        - INDEX-ONLY: Index OPEN, Middle/Ring/Pinky CLOSED
        - TWO-FINGER: Index & Middle OPEN, Ring & Pinky CLOSED
        - PINKY-ONLY: Pinky OPEN, Index/Middle/Ring CLOSED (Thumb NOT OPEN to prevent CALL mis-routing)

        Returns candidate command ("INDEX_FINGER", "TWO_FINGERS", "PINKY") if matched, or None if ambiguous/unmatched.
        """
        if hand_landmarks is None:
            return None

        fs = self.rule_recognizer.finger_detector.detect(hand_landmarks)
        t, i, m, r, p = fs.thumb, fs.index, fs.middle, fs.ring, fs.pinky

        # INDEX-ONLY: Index OPEN, Middle CLOSED, Ring CLOSED, Pinky CLOSED
        if i == FingerState.OPEN and m == FingerState.CLOSED and r == FingerState.CLOSED and p == FingerState.CLOSED:
            return "INDEX_FINGER"

        # TWO-FINGER: Index OPEN, Middle OPEN, Ring CLOSED, Pinky CLOSED
        if i == FingerState.OPEN and m == FingerState.OPEN and r == FingerState.CLOSED and p == FingerState.CLOSED:
            return "TWO_FINGERS"

        # PINKY-ONLY: Pinky OPEN, Index CLOSED, Middle CLOSED, Ring CLOSED, Thumb CLOSED
        if p == FingerState.OPEN and i == FingerState.CLOSED and m == FingerState.CLOSED and r == FingerState.CLOSED and t == FingerState.CLOSED:
            return "PINKY"

        return None

    def predict(
        self,
        features: np.ndarray,
        hand_landmarks: Optional[HandLandmarks] = None,
        is_stop_armed: bool = True,
        raw_frame: Optional[np.ndarray] = None,
    ) -> RecognitionResult:
        """
        Predict canonical gesture command using production hybrid routing.
        """
        # Step 1: Deterministic Rule-Based STOP Safety Check FIRST (when 1 valid hand present)
        if hand_landmarks is not None:
            rule_res = self.rule_recognizer.predict(
                features, hand_landmarks=hand_landmarks, is_stop_armed=is_stop_armed
            )
            if rule_res.label in ("STOP", "STOP_ACTIVE"):
                logger.debug("Authoritative Rule-Based STOP safety triggered.")
                return RecognitionResult(
                    label="STOP",
                    confidence=1.0,
                    all_scores={"STOP": 1.0},
                    recognizer_type="rule_based_stop_authoritative",
                    reason="DETERMINISTIC_STOP_SAFETY",
                    model_used="RULE_BASED",
                    raw_prediction="STOP",
                )

        # Step 2: Landmark Routing Gate (PRE-CNN evaluation)
        gate_target = self._check_landmark_routing_gate(hand_landmarks)

        if gate_target is not None:
            # ROUTE TO EXTRA TREES DIRECTLY (Skip HaGRID ResNet18 CPU CNN)
            if features is not None and len(features) == FeatureExtractor.FEATURE_DIM and self.ml_recognizer.is_ready:
                ml_res = self.ml_recognizer.predict(
                    features, hand_landmarks=hand_landmarks, is_stop_armed=is_stop_armed
                )
                et_raw = ml_res.label
                final_cmd = map_extratrees_to_neurogrip(et_raw)

                # Targeted INDEX_FINGER fallback: If landmark gate verified INDEX_FINGER but Extra Trees returned UNKNOWN/NO_COMMAND
                if gate_target == "INDEX_FINGER" and final_cmd == NO_COMMAND:
                    final_cmd = "INDEX_FINGER"

                conf_val = float(ml_res.confidence) if (final_cmd != "INDEX_FINGER" or ml_res.confidence > 0.0) else 1.0
                raw_pred = et_raw if (et_raw != "UNKNOWN" or final_cmd != "INDEX_FINGER") else "INDEX"

                return RecognitionResult(
                    label=final_cmd,
                    confidence=conf_val,
                    all_scores=ml_res.all_scores,
                    recognizer_type="hybrid_extra_trees",
                    reason=f"LANDMARK_GATE_EXTRA_TREES({gate_target} -> ET predicted {et_raw}->{final_cmd})",
                    model_used="EXTRA_TREES",
                    raw_prediction=raw_pred,
                )
            else:
                return RecognitionResult(
                    label=NO_COMMAND,
                    confidence=0.0,
                    recognizer_type="hybrid_extra_trees_failed",
                    reason=f"LANDMARK_GATE_EXTRA_TREES_BUT_INVALID_FEATURES({gate_target})",
                    model_used="EXTRA_TREES",
                    raw_prediction="INVALID_FEATURES",
                )

        # Step 3: HaGRID ResNet18 Full-Frame Inference for ordinary/unambiguous non-gate postures
        if raw_frame is not None and self.hagrid_classifier.is_loaded:
            try:
                hagrid_res = self.hagrid_classifier.predict(raw_frame, is_bgr=True, top_k=5)
                hagrid_raw_cls = hagrid_res["raw_class"]
                hagrid_mapped_cmd = hagrid_res["mapped_command"]
                hagrid_conf = float(hagrid_res["confidence"])
                hagrid_top_k = hagrid_res.get("top_k", [])

                if hagrid_mapped_cmd in NEUROGRIP_TAXONOMY and hagrid_mapped_cmd != NO_COMMAND:
                    final_cmd = hagrid_mapped_cmd

                    all_sc = {p["mapped_command"]: p["probability"] for p in hagrid_top_k if p["mapped_command"] != NO_COMMAND}
                    return RecognitionResult(
                        label=final_cmd,
                        confidence=hagrid_conf,
                        all_scores=all_sc,
                        recognizer_type="hybrid_hagrid",
                        reason=f"HAGRID_DIRECT({hagrid_raw_cls}->{final_cmd})",
                        model_used="HAGRID",
                        raw_prediction=hagrid_raw_cls,
                    )
                else:
                    return RecognitionResult(
                        label=NO_COMMAND,
                        confidence=hagrid_conf,
                        recognizer_type="hybrid_unsupported",
                        reason=f"UNSUPPORTED_HAGRID_CLASS({hagrid_raw_cls})",
                        model_used="HAGRID",
                        raw_prediction=hagrid_raw_cls,
                    )

            except Exception as e:
                logger.error("HaGRID prediction failed during frame processing: %s", e)

        # Fallback when raw_frame is None (e.g. feature-only unit testing)
        if features is not None and len(features) == FeatureExtractor.FEATURE_DIM and self.ml_recognizer.is_ready:
            ml_res = self.ml_recognizer.predict(
                features, hand_landmarks=hand_landmarks, is_stop_armed=is_stop_armed
            )
            et_raw = ml_res.label
            final_cmd = map_extratrees_to_neurogrip(et_raw)
            return RecognitionResult(
                label=final_cmd,
                confidence=float(ml_res.confidence),
                all_scores=ml_res.all_scores,
                recognizer_type="extra_trees_feature_only",
                reason=f"EXTRA_TREES_FEATURE_ONLY({et_raw}->{final_cmd})",
                model_used="EXTRA_TREES",
                raw_prediction=et_raw,
            )

        return RecognitionResult(
            label=NO_COMMAND,
            confidence=0.0,
            recognizer_type="hybrid_uninitialized",
            reason="NO_FRAME_OR_FEATURES_AVAILABLE",
            model_used="UNKNOWN",
            raw_prediction="NONE",
        )


# Alias for explicit name compatibility
HaGRIDExtraTreesHybridRecognizer = HybridRecognizer

