"""HaGRID Architecture Benchmark Module (Architecture A vs Architecture B).

Phase 2.1 Timing Validation update:
Detailed component-level latency measurement separating:
- Preprocessing latency
- Detector latency
- Crop calculation latency
- Crop preprocessing latency
- ResNet18 classifier inference latency
- Postprocessing / mapping latency
- Total pipeline latency

Strict Hand Selection Rules enforced for Architecture B:
- 0 hands detected  -> NO_COMMAND (reason: NO_HAND)
- 1 hand detected   -> Crop with configurable pad_ratio -> ResNet18 -> Mapped Command
- >1 hands detected -> NO_COMMAND (reason: MULTI_HAND_AMBIGUITY; draw all boxes in GUI)
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from neurogrip.hagrid.config import HaGRIDConfig
from neurogrip.hagrid.detector import HaGRIDHandDetector
from neurogrip.hagrid.model import HaGRIDClassifier
from neurogrip.hagrid.taxonomy import NO_COMMAND, map_hagrid_to_neurogrip


def crop_from_box(
    image: np.ndarray, box: List[int], pad_ratio: float = 0.15
) -> Tuple[np.ndarray, List[int]]:
    """Crop hand region from image using bounding box expanded by padding ratio.

    Args:
        image: Input image array (H, W, 3).
        box: Bounding box [x1, y1, x2, y2].
        pad_ratio: Padding expansion margin percentage (default 0.15 = 15%).

    Returns:
        Tuple of (cropped_image_array, clipped_pad_box_coords [x1, y1, x2, y2]).
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot crop from empty image.")

    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = box

    w_box = max(1, x2 - x1)
    h_box = max(1, y2 - y1)

    pad_w = int(round(w_box * pad_ratio))
    pad_h = int(round(h_box * pad_ratio))

    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w_img, x2 + pad_w)
    crop_y2 = min(h_img, y2 + pad_h)

    # Ensure valid non-empty slice
    if crop_x2 <= crop_x1:
        crop_x1 = 0
        crop_x2 = w_img
    if crop_y2 <= crop_y1:
        crop_y1 = 0
        crop_y2 = h_img

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, [crop_x1, crop_y1, crop_x2, crop_y2]


def predict_architecture_a(
    classifier: HaGRIDClassifier, image: np.ndarray, is_bgr: bool = True
) -> Dict[str, Any]:
    """Run Architecture A — Full-Frame Baseline Pipeline with Phase 2.1 component timing.

    Webcam/Image -> ResNet18 -> Raw Class -> Taxonomy Mapper

    Args:
        classifier: Initialized HaGRIDClassifier instance.
        image: NumPy frame (H, W, 3).
        is_bgr: True if frame is BGR (OpenCV format).

    Returns:
        Dict containing prediction details, latency, and component timing breakdown.
    """
    t0 = time.perf_counter()

    # 1. Preprocessing timing
    input_tensor = classifier.preprocess(image, is_bgr=is_bgr)
    t1 = time.perf_counter()

    # 2. ResNet18 inference timing
    clf_res = classifier.predict(image, is_bgr=is_bgr, top_k=5)
    t2 = time.perf_counter()

    # 3. Postprocessing / mapping timing
    t_post_start = time.perf_counter()
    raw_class = clf_res["raw_class"]
    mapped_cmd = clf_res["mapped_command"]
    t_post_end = time.perf_counter()

    prep_ms = (t1 - t0) * 1000.0
    inference_ms = clf_res["latency_ms"]
    postproc_ms = (t_post_end - t_post_start) * 1000.0
    total_ms = (t2 - t0) * 1000.0 + postproc_ms

    return {
        "architecture": "A_FULL_FRAME",
        "raw_class": raw_class,
        "raw_index": clf_res["raw_index"],
        "mapped_command": mapped_cmd,
        "confidence": clf_res["confidence"],
        "top_k": clf_res["top_k"],
        "latency_ms": total_ms,
        "breakdown": {
            "preprocessing_ms": prep_ms,
            "inference_ms": inference_ms,
            "postprocessing_ms": postproc_ms,
            "total_ms": total_ms,
        },
    }


def predict_architecture_b(
    detector: HaGRIDHandDetector,
    classifier: HaGRIDClassifier,
    image: np.ndarray,
    is_bgr: bool = True,
    conf_threshold: float = 0.40,
    pad_ratio: float = 0.15,
    force_hand_box: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run Architecture B — Hand-Crop Pipeline with Phase 2.1 component timing.

    Webcam/Image -> YOLOv10n -> Hand Count Validation -> Crop (if 1) -> ResNet18 -> Taxonomy

    Strict Rules:
    - 0 hands: mapped_command = NO_COMMAND, reason = NO_HAND
    - 1 hand:  crop + classify -> mapped_command, reason = SINGLE_HAND_SUCCESS
    - >1 hands: mapped_command = NO_COMMAND, reason = MULTI_HAND_AMBIGUITY

    Args:
        detector: Initialized HaGRIDHandDetector instance.
        classifier: Initialized HaGRIDClassifier instance.
        image: NumPy frame (H, W, 3).
        is_bgr: True if frame is BGR (OpenCV format).
        conf_threshold: Detector confidence threshold.
        pad_ratio: Bounding box expansion ratio.
        force_hand_box: Optional forced bounding box for benchmark timing validation.

    Returns:
        Dict containing prediction details, hand count, reason, latency, and component breakdown.
    """
    t0 = time.perf_counter()

    # 1. Run YOLOv10n hand detector (or forced box for benchmark testing)
    if force_hand_box is not None:
        t_det_0 = time.perf_counter()
        det_res = {
            "boxes": [force_hand_box],
            "confidences": [0.90],
            "count": 1,
            "latency_ms": 50.0,
        }
        t_det_1 = time.perf_counter()
        detector_ms = (t_det_1 - t_det_0) * 1000.0
    else:
        det_res = detector.detect(image, conf_threshold=conf_threshold)
        detector_ms = det_res["latency_ms"]

    hand_count = det_res["count"]
    boxes = det_res["boxes"]
    confidences = det_res["confidences"]

    crop_calc_ms = 0.0
    crop_prep_ms = 0.0
    classifier_ms = 0.0
    postproc_ms = 0.0

    if hand_count == 0:
        # Rule 1: Zero detected hands
        raw_class = "no_hand"
        mapped_cmd = NO_COMMAND
        confidence = 0.0
        reason = "NO_HAND"
        selected_crop_box = None
        top_k = []
        t_end = time.perf_counter()
    elif hand_count == 1:
        # Rule 2: Exactly one detected hand -> crop & classify
        t_crop_calc_start = time.perf_counter()
        target_box = boxes[0]
        cropped_img, selected_crop_box = crop_from_box(image, target_box, pad_ratio=pad_ratio)
        t_crop_calc_end = time.perf_counter()
        crop_calc_ms = (t_crop_calc_end - t_crop_calc_start) * 1000.0

        # Crop Preprocessing timing
        t_crop_prep_start = time.perf_counter()
        input_tensor = classifier.preprocess(cropped_img, is_bgr=is_bgr)
        t_crop_prep_end = time.perf_counter()
        crop_prep_ms = (t_crop_prep_end - t_crop_prep_start) * 1000.0

        # Classifier Inference timing
        clf_res = classifier.predict(cropped_img, is_bgr=is_bgr, top_k=5)
        classifier_ms = clf_res["latency_ms"]

        # Postprocessing / mapping timing
        t_post_start = time.perf_counter()
        raw_class = clf_res["raw_class"]
        mapped_cmd = clf_res["mapped_command"]
        confidence = clf_res["confidence"]
        top_k = clf_res["top_k"]
        reason = "SINGLE_HAND_SUCCESS"
        t_post_end = time.perf_counter()
        postproc_ms = (t_post_end - t_post_start) * 1000.0
        t_end = time.perf_counter()
    else:
        # Rule 3: Multiple detected hands -> NO_COMMAND (Multi-hand ambiguity)
        raw_class = "multi_hand"
        mapped_cmd = NO_COMMAND
        confidence = 0.0
        reason = "MULTI_HAND_AMBIGUITY"
        selected_crop_box = None
        top_k = []
        t_end = time.perf_counter()

    total_ms = (t_end - t0) * 1000.0

    return {
        "architecture": "B_HAND_CROP",
        "detected_hands_count": hand_count,
        "detected_boxes": boxes,
        "detected_confidences": confidences,
        "selected_crop_box": selected_crop_box,
        "raw_class": raw_class,
        "mapped_command": mapped_cmd,
        "confidence": confidence,
        "reason": reason,
        "top_k": top_k,
        "latency_ms": total_ms,
        "breakdown": {
            "detector_ms": detector_ms,
            "crop_calc_ms": crop_calc_ms,
            "crop_prep_ms": crop_prep_ms,
            "classifier_ms": classifier_ms,
            "postprocessing_ms": postproc_ms,
            "total_ms": total_ms,
        },
    }
