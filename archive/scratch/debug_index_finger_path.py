"""
pc/scratch/debug_index_finger_path.py
──────────────────────────────────────
Trace exact INDEX_FINGER prediction through HybridRecognizer, ExtraTrees, Routing Gate, Taxonomy, Validator, Stabilizer.
"""
import sys
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, 'pc/src')

from neurogrip.config.settings import AppConfig
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.hybrid import HybridRecognizer
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from neurogrip.commands.validator import CommandValidator, PipelineState
from neurogrip.stabilization.temporal import TemporalStabilizer


def build_index_finger_landmarks(middle_ring_pinky_state="curled"):
    """
    Build 21 landmarks for an index finger extended pointing gesture.
    """
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)

    # Index finger extended UP
    lms[5] = NormalizedLandmark(x=0.45, y=0.5, z=0.0)
    lms[6] = NormalizedLandmark(x=0.45, y=0.38, z=0.0)
    lms[7] = NormalizedLandmark(x=0.45, y=0.25, z=0.0)
    lms[8] = NormalizedLandmark(x=0.45, y=0.10, z=0.0)  # Extended Tip

    # Middle, Ring, Pinky curled
    for mcp_idx, tip_idx in [(9, 12), (13, 16), (17, 20)]:
        x_pos = 0.5 + (mcp_idx - 9) * 0.02
        lms[mcp_idx] = NormalizedLandmark(x=x_pos, y=0.5, z=0.0)
        lms[mcp_idx + 1] = NormalizedLandmark(x=x_pos, y=0.55, z=0.0)
        lms[mcp_idx + 2] = NormalizedLandmark(x=x_pos, y=0.58, z=0.0)
        lms[tip_idx] = NormalizedLandmark(x=x_pos, y=0.55, z=0.0)  # Curled tip near MCP

    # Thumb tucked across palm
    lms[1] = NormalizedLandmark(x=0.48, y=0.7, z=0.0)
    lms[2] = NormalizedLandmark(x=0.47, y=0.65, z=0.0)
    lms[3] = NormalizedLandmark(x=0.46, y=0.6, z=0.0)
    lms[4] = NormalizedLandmark(x=0.45, y=0.58, z=0.0)

    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def trace_index_finger():
    print("==================================================")
    print("TRACING INDEX_FINGER END-TO-END PIPELINE")
    print("==================================================")

    config = AppConfig.default()
    hybrid = HybridRecognizer(config=config)
    rule_rec = RuleBasedRecognizer(config=config)

    index_hand = build_index_finger_landmarks()

    # 1. Inspect Finger States
    fs = rule_rec.finger_detector.detect(index_hand)
    print(f"\n1. FingerStateDetector output:")
    print(f"   Thumb:  {fs.thumb}")
    print(f"   Index:  {fs.index}")
    print(f"   Middle: {fs.middle}")
    print(f"   Ring:   {fs.ring}")
    print(f"   Pinky:  {fs.pinky}")

    # 2. Inspect Landmark Routing Gate
    gate_target = hybrid._check_landmark_routing_gate(index_hand)
    print(f"\n2. Routing Gate Result: '{gate_target}'")

    # 3. Inspect Feature Extractor & Extra Trees Predictor
    from neurogrip.features.extractor import FeatureExtractor
    extractor = FeatureExtractor(config.features)
    features = extractor.extract(index_hand)
    print(f"\n3. Extracted Feature Dimension: {len(features)}")

    et_res = hybrid.ml_recognizer.predict(features, hand_landmarks=index_hand)
    print(f"\n4. Extra Trees Raw Prediction:")
    print(f"   et_raw:       '{et_res.label}'")
    print(f"   confidence:   {et_res.confidence:.4f}")
    from neurogrip.recognition.hybrid import map_extratrees_to_neurogrip
    mapped_et = map_extratrees_to_neurogrip(et_res.label)
    print(f"   mapped_cmd:   '{mapped_et}'")

    # 5. Hybrid Prediction Result (with & without raw_frame)
    res_no_frame = hybrid.predict(features, hand_landmarks=index_hand)
    print(f"\n5. HybridRecognizer.predict (features + landmarks):")
    print(f"   label:           '{res_no_frame.label}'")
    print(f"   model_used:      '{res_no_frame.model_used}'")
    print(f"   recognizer_type: '{res_no_frame.recognizer_type}'")
    print(f"   reason:          '{res_no_frame.reason}'")

    # 6. Command Validator check
    validator = CommandValidator()
    val_res = validator.validate(res_no_frame.label)
    print(f"\n6. CommandValidator validation:")
    print(f"   is_valid: '{val_res.is_valid}'")
    print(f"   command:  '{val_res.command}'")
    print(f"   reason:   '{val_res.reason}'")

    # 7. Temporal Stabilizer check
    stabilizer = TemporalStabilizer(config=config)
    for i in range(10):
        stab_res = stabilizer.update(res_no_frame)
    print(f"\n7. TemporalStabilizer (10 frames):")
    print(f"   state:           '{stab_res.state}'")
    print(f"   stable_command:  '{stab_res.stable_command}'")
    print(f"   emitted_command: '{stab_res.emitted_command}'")

    print("\n==================================================")


if __name__ == "__main__":
    trace_index_finger()
