"""Unit Tests for HaGRID Phase 1 Model Verification, Taxonomy Mapping, and Preprocessing.

Tests cover:
1. Exact 34-class list and index ordering verification.
2. Taxonomy mapping for all 14 NeuroGrip locked commands and multi-class aliases.
3. Explicit mapping of HaGRID class 'one' to NO_COMMAND.
4. Mapping of unauthorized/invalid classes to NO_COMMAND.
5. Preprocessing pipeline transformation to (1, 3, 224, 224) float tensor.
6. Checkpoint loading and inference dimensions (skips gracefully if checkpoint unavailable).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from neurogrip.hagrid.config import HaGRIDConfig
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import (
    HAGRID_CLASSES,
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    is_valid_neurogrip_command,
    map_hagrid_to_neurogrip,
)


def test_hagrid_class_count_and_ordering():
    """Verify that HAGRID_CLASSES has exactly 34 classes in verified official order."""
    assert len(HAGRID_CLASSES) == 34, f"Expected 34 classes, got {len(HAGRID_CLASSES)}"

    # Check key index positions verified from official HaGRID constants.py
    assert HAGRID_CLASSES[0] == "grabbing"
    assert HAGRID_CLASSES[1] == "grip"
    assert HAGRID_CLASSES[3] == "point"
    assert HAGRID_CLASSES[4] == "call"
    assert HAGRID_CLASSES[10] == "little_finger"
    assert HAGRID_CLASSES[11] == "middle_finger"
    assert HAGRID_CLASSES[14] == "fist"
    assert HAGRID_CLASSES[15] == "four"
    assert HAGRID_CLASSES[16] == "like"
    assert HAGRID_CLASSES[18] == "ok"
    assert HAGRID_CLASSES[19] == "one", "Class 'one' must be at index 19"
    assert HAGRID_CLASSES[23] == "rock"
    assert HAGRID_CLASSES[24] == "stop"
    assert HAGRID_CLASSES[25] == "stop_inverted"
    assert HAGRID_CLASSES[26] == "three"
    assert HAGRID_CLASSES[27] == "three2"
    assert HAGRID_CLASSES[28] == "two_up"
    assert HAGRID_CLASSES[29] == "two_up_inverted"
    assert HAGRID_CLASSES[30] == "three_gun"
    assert HAGRID_CLASSES[33] == "no_gesture", "Class 'no_gesture' must be at index 33"


def test_neurogrip_locked_taxonomy():
    """Verify the 14 locked NeuroGrip taxonomy commands."""
    assert len(NEUROGRIP_TAXONOMY) == 14
    expected_taxonomy = {
        "CALL",
        "CLOSED_FIST",
        "FOUR_FINGERS",
        "GRABBING",
        "GRIP",
        "THUMBS_UP",
        "PINKY",
        "MIDDLE_FINGER",
        "OK",
        "INDEX_FINGER",
        "INDEX_PINKY",
        "STOP",
        "TWO_FINGERS",
        "THREE_FINGERS",
    }
    assert set(NEUROGRIP_TAXONOMY) == expected_taxonomy
    assert NO_COMMAND not in NEUROGRIP_TAXONOMY


def test_taxonomy_mapping_locked_commands():
    """Verify official HaGRID class mappings to NeuroGrip locked taxonomy."""
    mapping_tests = {
        "call": "CALL",
        "fist": "CLOSED_FIST",
        "four": "FOUR_FINGERS",
        "grabbing": "GRABBING",
        "grip": "GRIP",
        "like": "THUMBS_UP",
        "little_finger": "PINKY",
        "middle_finger": "MIDDLE_FINGER",
        "ok": "OK",
        "point": "INDEX_FINGER",
        "rock": "INDEX_PINKY",
        "stop": "STOP",
        "stop_inverted": "STOP",
        "two_up": "TWO_FINGERS",
        "two_up_inverted": "TWO_FINGERS",
        "three": "THREE_FINGERS",
        "three2": "THREE_FINGERS",
        "three3": "THREE_FINGERS",
        "three_gun": "THREE_FINGERS",
    }

    for raw_cls, expected_cmd in mapping_tests.items():
        mapped = map_hagrid_to_neurogrip(raw_cls)
        assert mapped == expected_cmd, f"Expected {raw_cls} -> {expected_cmd}, got {mapped}"
        assert is_valid_neurogrip_command(mapped), f"Mapped command {mapped} must be valid"


def test_taxonomy_mapping_one_explicitly_removed():
    """Verify that HaGRID 'one' explicitly maps to NO_COMMAND."""
    mapped = map_hagrid_to_neurogrip("one")
    assert mapped == NO_COMMAND
    assert not is_valid_neurogrip_command(mapped)


def test_taxonomy_mapping_invalid_and_unauthorized_classes():
    """Verify that all non-taxonomical classes map to NO_COMMAND."""
    unauthorized_classes = [
        "dislike",
        "hand_heart",
        "hand_heart2",
        "holy",
        "mute",
        "no_gesture",
        "palm",
        "peace",
        "peace_inverted",
        "take_picture",
        "thumb_index",
        "thumb_index2",
        "timeout",
        "xsign",
        "random_unknown_class",
        "",
        None,
    ]

    for raw_cls in unauthorized_classes:
        mapped = map_hagrid_to_neurogrip(raw_cls)
        assert mapped == NO_COMMAND, f"Class '{raw_cls}' should map to NO_COMMAND, got '{mapped}'"


def test_preprocessing_tensor_shape_and_normalization():
    """Verify that image preprocessing produces (1, 3, 224, 224) tensor with correct normalization."""
    config = HaGRIDConfig()
    clf = HaGRIDClassifier(config=config, auto_load=False)

    # Create dummy 480x640 BGR image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[:, :] = (100, 150, 200)  # BGR

    tensor = clf.preprocess(dummy_img, is_bgr=True)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_model_loading_and_inference_if_checkpoint_exists():
    """Test model loading and prediction if local checkpoint exists."""
    config = HaGRIDConfig()
    checkpoint_path = config.checkpoint_path

    if not checkpoint_path.exists():
        pytest.skip(f"Skipping test because checkpoint {checkpoint_path} is not present locally.")

    clf = HaGRIDClassifier(config=config, auto_load=True)
    assert clf.is_loaded
    assert clf.model is not None

    # Test single frame prediction
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = clf.predict(dummy_frame, is_bgr=True, top_k=5)

    assert "raw_class" in res
    assert res["raw_class"] in HAGRID_CLASSES
    assert "mapped_command" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 5
    assert res["latency_ms"] > 0.0
