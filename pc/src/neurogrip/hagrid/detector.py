"""Official HaGRID YOLOv10n Hand Detector Wrapper for NeuroGrip Phase 2."""

import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from neurogrip.hagrid.config import HaGRIDConfig


class HaGRIDHandDetector:
    """Official HaGRIDv2 YOLOv10n Hand Detector wrapper."""

    def __init__(self, config: Optional[HaGRIDConfig] = None, auto_load: bool = True):
        """Initialize the HaGRID hand detector.

        Args:
            config: HaGRIDConfig instance. Uses default settings if None.
            auto_load: If True, attempts to download (if needed) and load model on init.
        """
        self.config = config or HaGRIDConfig()
        self.model: Optional[YOLO] = None
        self.is_loaded: bool = False

        if auto_load:
            self.load_model()

    def download_checkpoint_if_missing(self) -> Path:
        """Download official pretrained YOLOv10n hand detector checkpoint if missing locally.

        Returns:
            Path to the verified checkpoint file.
        """
        detector_path = self.config.detector_path
        if detector_path.exists() and detector_path.stat().st_size > 0:
            return detector_path

        self.config.ensure_checkpoint_dir()
        url = self.config.official_detector_url
        print(f"[HaGRIDHandDetector] Downloading official detector checkpoint from: {url}")
        print(f"[HaGRIDHandDetector] Saving to: {detector_path}")

        req = urllib.request.Request(url, headers={"User-Agent": "NeuroGrip-Phase2/1.0"})
        with urllib.request.urlopen(req) as response, open(detector_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(
                        f"\r[HaGRIDHandDetector] Progress: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({percent:.1f}%)",
                        end="",
                    )
            print("\n[HaGRIDHandDetector] Download complete.")

        return detector_path

    def load_model(self) -> None:
        """Instantiate YOLOv10n hand detector model."""
        detector_path = self.download_checkpoint_if_missing()
        if not detector_path.exists():
            raise RuntimeError(f"Hand detector checkpoint not found at: {detector_path}")

        try:
            self.model = YOLO(str(detector_path))
            self.is_loaded = True
            print(f"[HaGRIDHandDetector] Successfully loaded YOLOv10n hand detector from: {detector_path}")
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Failed to load official YOLOv10n hand detector: {e}") from e

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Detect hand bounding boxes in an input image frame.

        Args:
            image: NumPy image array (H, W, 3) in BGR or RGB format.
            conf_threshold: Confidence threshold for filtering detections (default from config).

        Returns:
            Dict containing boxes [[x1, y1, x2, y2]], confidences, count, latency_ms.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("HaGRIDHandDetector is not loaded.")

        if image is None or image.size == 0:
            raise ValueError("Input image frame is empty or invalid.")

        h, w = image.shape[:2]
        threshold = conf_threshold if conf_threshold is not None else self.config.detector_conf_threshold

        t0 = time.perf_counter()
        results = self.model(image, conf=threshold, verbose=False)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        boxes_list = []
        confidences = []

        if results and len(results) > 0:
            boxes_obj = results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()

                for box, conf in zip(xyxy, confs):
                    # Clip & validate coordinates to image dimensions
                    x1 = max(0, min(int(round(box[0])), w - 1))
                    y1 = max(0, min(int(round(box[1])), h - 1))
                    x2 = max(0, min(int(round(box[2])), w))
                    y2 = max(0, min(int(round(box[3])), h))

                    # Ensure non-zero width and height
                    if x2 > x1 and y2 > y1:
                        boxes_list.append([x1, y1, x2, y2])
                        confidences.append(float(conf))

        return {
            "boxes": boxes_list,
            "confidences": confidences,
            "count": len(boxes_list),
            "latency_ms": latency_ms,
        }
