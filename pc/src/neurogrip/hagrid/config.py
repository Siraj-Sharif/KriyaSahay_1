"""HaGRID Model & Detector Configuration Module for NeuroGrip."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


def get_default_checkpoint_dir() -> Path:
    """Return the absolute path to the official HaGRID model checkpoint directory."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent  # pc directory or root
    checkpoint_dir = base_dir / "models" / "hagrid"
    return checkpoint_dir


@dataclass
class HaGRIDConfig:
    """Configuration parameters for HaGRID classifier, detector, and benchmark pipelines."""

    # Primary Classifier (ResNet18)
    model_name: str = "ResNet18"
    checkpoint_filename: str = "ResNet18.pth"
    checkpoint_dir: Path = field(default_factory=get_default_checkpoint_dir)
    official_checkpoint_url: str = (
        "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth"
    )

    # Official Hand Detector (YOLOv10n)
    detector_filename: str = "YOLOv10n_hands.pt"
    official_detector_url: str = (
        "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt"
    )

    # Preprocessing parameters (strictly verified from official HaGRID repo configs)
    img_size: int = 224
    img_mean: Tuple[float, float, float] = (0.54, 0.499, 0.474)
    img_std: Tuple[float, float, float] = (0.234, 0.235, 0.231)
    pad_value: Tuple[int, int, int] = (144, 144, 144)

    # Architecture B (Hand-Crop) Parameters
    detector_conf_threshold: float = 0.40  # Default confidence threshold for hand detector
    crop_pad_ratio: float = 0.15           # Default 15% bounding box margin expansion

    # Hardware & runtime
    device: str = "auto"  # 'auto', 'cuda', or 'cpu'
    confidence_threshold: float = 0.5  # Provisional threshold for diagnostic display

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path to the primary classifier checkpoint file."""
        return self.checkpoint_dir / self.checkpoint_filename

    @property
    def detector_path(self) -> Path:
        """Absolute path to the primary hand detector checkpoint file."""
        return self.checkpoint_dir / self.detector_filename

    def ensure_checkpoint_dir(self) -> Path:
        """Ensure local model directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir
