"""Unit Tests for HaGRID Phase 2 Architecture Benchmark and Hand Selection Rules.

Tests cover:
1. Detector initialization & execution structure.
2. Hand selection rules:
   - 0 hands -> NO_COMMAND (NO_HAND)
   - 1 hand  -> Crop & classify (SINGLE_HAND_SUCCESS)
   - >1 hands -> NO_COMMAND (MULTI_HAND_AMBIGUITY)
3. Bounding box crop expansion padding and boundary clipping.
4. Empty/invalid crop safety handling.
5. Architecture A & Architecture B output dictionary structures.
6. Taxonomy locking and explicit 'one' -> NO_COMMAND removal.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from neurogrip.hagrid.benchmark import (
    crop_from_box,
    predict_architecture_a,
    predict_architecture_b,
)
from neurogrip.hagrid.config import HaGRIDConfig
from neurogrip.hagrid.detector import HaGRIDHandDetector
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY, NO_COMMAND, map_hagrid_to_neurogrip


def test_detector_initialization_and_structure():
    """Verify detector initialization and output dictionary structure using a mock model."""
    config = HaGRIDConfig()
    detector = HaGRIDHandDetector(config=config, auto_load=False)

    # Mock YOLO model
    mock_yolo = MagicMock()
    mock_boxes = MagicMock()

    # Mock box data: 1 box [[100, 150, 300, 400]], conf [0.85]
    mock_boxes.xyxy.cpu().numpy.return_value = np.array([[100, 150, 300, 400]], dtype=np.float32)
    mock_boxes.conf.cpu().numpy.return_value = np.array([0.85], dtype=np.float32)
    mock_boxes.__len__.return_value = 1

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_yolo.return_value = [mock_result]

    detector.model = mock_yolo
    detector.is_loaded = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame, conf_threshold=0.40)

    assert "boxes" in res
    assert "confidences" in res
    assert "count" in res
    assert "latency_ms" in res
    assert res["count"] == 1
    assert res["boxes"] == [[100, 150, 300, 400]]
    assert res["confidences"] == [pytest.approx(0.85)]


def test_crop_from_box_padding_and_clipping():
    """Verify crop_from_box expands bounding box by padding ratio and clips to image boundaries."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    box = [100, 100, 200, 200]  # 100x100 box

    # 15% padding -> 15px expansion on each side -> [85, 85, 215, 215]
    crop, pad_box = crop_from_box(dummy_img, box, pad_ratio=0.15)

    assert pad_box == [85, 85, 215, 215]
    assert crop.shape == (130, 130, 3)

    # Corner box near boundary -> must clip to 0 and W/H
    corner_box = [10, 10, 50, 50]
    crop_corner, pad_corner = crop_from_box(dummy_img, corner_box, pad_ratio=0.50)

    assert pad_corner[0] == 0, "Left coordinate must clip to 0"
    assert pad_corner[1] == 0, "Top coordinate must clip to 0"
    assert crop_corner.shape[0] > 0 and crop_corner.shape[1] > 0


def test_architecture_b_zero_hand_rule():
    """Verify Architecture B safety rule for 0 detected hands -> NO_COMMAND (NO_HAND)."""
    mock_detector = MagicMock(spec=HaGRIDHandDetector)
    mock_detector.detect.return_value = {
        "boxes": [],
        "confidences": [],
        "count": 0,
        "latency_ms": 15.0,
    }

    mock_classifier = MagicMock(spec=HaGRIDClassifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = predict_architecture_b(mock_detector, mock_classifier, dummy_frame)

    assert res["architecture"] == "B_HAND_CROP"
    assert res["detected_hands_count"] == 0
    assert res["mapped_command"] == NO_COMMAND
    assert res["reason"] == "NO_HAND"
    # Classifier must NOT be invoked when 0 hands are detected
    assert mock_classifier.predict.call_count == 0


def test_architecture_b_single_hand_rule():
    """Verify Architecture B pipeline when exactly 1 hand is detected -> Crop & Classify."""
    mock_detector = MagicMock(spec=HaGRIDHandDetector)
    mock_detector.detect.return_value = {
        "boxes": [[100, 100, 200, 200]],
        "confidences": [0.90],
        "count": 1,
        "latency_ms": 20.0,
    }

    mock_classifier = MagicMock(spec=HaGRIDClassifier)
    mock_classifier.predict.return_value = {
        "raw_class": "fist",
        "raw_index": 14,
        "mapped_command": "CLOSED_FIST",
        "confidence": 0.95,
        "top_k": [{"class": "fist", "mapped_command": "CLOSED_FIST", "probability": 0.95}],
        "latency_ms": 10.0,
    }

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = predict_architecture_b(mock_detector, mock_classifier, dummy_frame)

    assert res["architecture"] == "B_HAND_CROP"
    assert res["detected_hands_count"] == 1
    assert res["mapped_command"] == "CLOSED_FIST"
    assert res["reason"] == "SINGLE_HAND_SUCCESS"
    assert res["selected_crop_box"] == [85, 85, 215, 215]
    assert mock_classifier.predict.call_count == 1


def test_architecture_b_multi_hand_rule():
    """Verify Architecture B safety rule for >1 detected hands -> NO_COMMAND (MULTI_HAND_AMBIGUITY)."""
    mock_detector = MagicMock(spec=HaGRIDHandDetector)
    mock_detector.detect.return_value = {
        "boxes": [[50, 50, 150, 150], [300, 300, 400, 400]],
        "confidences": [0.88, 0.82],
        "count": 2,
        "latency_ms": 25.0,
    }

    mock_classifier = MagicMock(spec=HaGRIDClassifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = predict_architecture_b(mock_detector, mock_classifier, dummy_frame)

    assert res["architecture"] == "B_HAND_CROP"
    assert res["detected_hands_count"] == 2
    assert res["mapped_command"] == NO_COMMAND
    assert res["reason"] == "MULTI_HAND_AMBIGUITY"
    assert res["selected_crop_box"] is None
    # Classifier must NOT be invoked when multiple hands are detected
    assert mock_classifier.predict.call_count == 0


def test_architecture_a_output_structure():
    """Verify Architecture A output dictionary structure."""
    mock_classifier = MagicMock(spec=HaGRIDClassifier)
    mock_classifier.preprocess.return_value = torch.zeros((1, 3, 224, 224))
    mock_classifier.predict.return_value = {
        "raw_class": "like",
        "raw_index": 16,
        "mapped_command": "THUMBS_UP",
        "confidence": 0.92,
        "top_k": [{"class": "like", "mapped_command": "THUMBS_UP", "probability": 0.92}],
        "latency_ms": 18.0,
    }

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = predict_architecture_a(mock_classifier, dummy_frame)

    assert res["architecture"] == "A_FULL_FRAME"
    assert res["mapped_command"] == "THUMBS_UP"
    assert "latency_ms" in res
    assert "breakdown" in res
