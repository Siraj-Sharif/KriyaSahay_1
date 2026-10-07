"""HaGRID Model Wrapper Module for NeuroGrip Phase 1 Verification.

This module provides the HaGRIDClassifier wrapper for downloading, loading,
preprocessing images, and running PyTorch model inference using verified
official HaGRID checkpoints (ResNet18 / MobileNetV3_large).
"""

import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.models as tv_models

from neurogrip.hagrid.config import HaGRIDConfig
from neurogrip.hagrid.taxonomy import (
    HAGRID_CLASSES,
    HAGRID_INDEX_TO_CLASS,
    NO_COMMAND,
    map_hagrid_to_neurogrip,
)


class HaGRIDClassifier:
    """Official HaGRID model wrapper for PC-side gesture recognition."""

    def __init__(self, config: Optional[HaGRIDConfig] = None, auto_load: bool = True):
        """Initialize the HaGRID classifier.

        Args:
            config: HaGRIDConfig instance. Uses default settings if None.
            auto_load: If True, attempts to download (if needed) and load model weights on init.
        """
        self.config = config or HaGRIDConfig()
        self.device = self._select_device(self.config.device)
        self.model: Optional[torch.nn.Module] = None
        self.is_loaded: bool = False

        if auto_load:
            self.load_model()

    def _select_device(self, requested_device: str) -> torch.device:
        """Select execution device (CPU or CUDA)."""
        if requested_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("[HaGRIDClassifier] CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
        elif requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        else:
            return torch.device("cpu")

    def download_checkpoint_if_missing(self) -> Path:
        """Download official pretrained checkpoint if missing locally.

        Returns:
            Path to the verified checkpoint file.
        """
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            return checkpoint_path

        self.config.ensure_checkpoint_dir()
        url = self.config.official_checkpoint_url
        print(f"[HaGRIDClassifier] Downloading official checkpoint from: {url}")
        print(f"[HaGRIDClassifier] Saving to: {checkpoint_path}")

        req = urllib.request.Request(url, headers={"User-Agent": "NeuroGrip-Phase1/1.0"})
        with urllib.request.urlopen(req) as response, open(checkpoint_path, "wb") as out_file:
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
                    print(f"\r[HaGRIDClassifier] Progress: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({percent:.1f}%)", end="")
            print("\n[HaGRIDClassifier] Download complete.")

        return checkpoint_path

    def load_model(self) -> None:
        """Instantiate PyTorch model architecture and load official state dict."""
        checkpoint_path = self.download_checkpoint_if_missing()

        # Build PyTorch architecture matching official config (num_classes=34)
        if self.config.model_name == "ResNet18":
            self.model = tv_models.resnet18(num_classes=len(HAGRID_CLASSES))
        elif self.config.model_name == "ResNet152":
            self.model = tv_models.resnet152(num_classes=len(HAGRID_CLASSES))
        elif self.config.model_name == "MobileNetV3_large":
            self.model = tv_models.mobilenet_v3_large(num_classes=len(HAGRID_CLASSES))
        elif self.config.model_name == "MobileNetV3_small":
            self.model = tv_models.mobilenet_v3_small(num_classes=len(HAGRID_CLASSES))
        else:
            raise ValueError(f"Unsupported model architecture: {self.config.model_name}")

        # Load snapshot state dict
        snapshot = torch.load(checkpoint_path, map_location=self.device)
        state_dict = snapshot.get("MODEL_STATE", snapshot)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print(f"[HaGRIDClassifier] Successfully loaded {self.config.model_name} on device: {self.device}")

    def preprocess(self, image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        """Preprocess an input OpenCV frame into the exact tensor expected by HaGRID.

        Transforms strictly implemented from official HaGRID test config:
        1. LongestMaxSize(224)
        2. PadIfNeeded(min_height=224, min_width=224, value=[144,144,144], border_mode=0)
        3. Normalize(mean=[0.54, 0.499, 0.474], std=[0.234, 0.235, 0.231], max_pixel_value=255.0)

        Args:
            image: NumPy array (H, W, 3) image in BGR or RGB order.
            is_bgr: True if input image is in BGR color order (OpenCV default).

        Returns:
            PyTorch tensor of shape (1, 3, 224, 224) on configured device.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # Convert BGR -> RGB if needed
        if is_bgr:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()

        h, w, _ = rgb_image.shape
        target_size = self.config.img_size

        # 1. Resize longest side to target_size (224)
        scale = target_size / float(max(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. Pad to min_height x min_width (224x224) with background color [144, 144, 144]
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        pad_color = list(self.config.pad_value)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        # 3. Normalize pixel values [0, 255] with mean & std
        tensor_img = padded.astype(np.float32) / 255.0
        mean = np.array(self.config.img_mean, dtype=np.float32)
        std = np.array(self.config.img_std, dtype=np.float32)
        normalized = (tensor_img - mean) / std

        # Transpose HWC -> CHW and add batch dimension (1, 3, 224, 224)
        chw = np.transpose(normalized, (2, 0, 1))
        batch_tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        return batch_tensor

    def predict(
        self, image: np.ndarray, is_bgr: bool = True, top_k: int = 5
    ) -> Dict[str, Any]:
        """Perform full-frame inference on a single image.

        Args:
            image: NumPy array input frame.
            is_bgr: Whether frame is BGR (default True).
            top_k: Number of top predictions to include in result.

        Returns:
            Dict containing raw_class, mapped_command, confidence, top_k list, latency_ms, etc.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("HaGRIDClassifier model is not loaded.")

        input_tensor = self.preprocess(image, is_bgr=is_bgr)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_logits = self.model(input_tensor)
            probs = torch.softmax(output_logits, dim=1)[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        probs_np = probs.cpu().numpy()
        top_k_indices = np.argsort(probs_np)[::-1][:top_k]

        top_predictions = []
        for idx in top_k_indices:
            raw_cls = HAGRID_INDEX_TO_CLASS[int(idx)]
            mapped_cmd = map_hagrid_to_neurogrip(raw_cls)
            prob = float(probs_np[idx])
            top_predictions.append(
                {
                    "class": raw_cls,
                    "index": int(idx),
                    "mapped_command": mapped_cmd,
                    "probability": prob,
                }
            )

        top_class = top_predictions[0]["class"]
        top_prob = top_predictions[0]["probability"]
        mapped_command = map_hagrid_to_neurogrip(top_class)

        return {
            "raw_class": top_class,
            "raw_index": top_predictions[0]["index"],
            "mapped_command": mapped_command,
            "confidence": top_prob,
            "top_k": top_predictions,
            "latency_ms": latency_ms,
            "device": str(self.device),
        }
