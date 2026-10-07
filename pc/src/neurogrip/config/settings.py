"""
neurogrip/config/settings.py
─────────────────────────────
AppConfig dataclass hierarchy + YAML loader.
All application parameters flow through this module — nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Sub-config dataclasses
# ─────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "opencv"


@dataclass
class HandTrackingConfig:
    model_path: str = "models/mediapipe/hand_landmarker.task"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass
class FeaturesConfig:
    wrist_landmark: int = 0
    span_landmark: int = 9
    open_extension_threshold: float = 0.40
    open_distance_threshold: float = 0.40
    open_cosine_threshold: float = 0.50
    closed_extension_threshold: float = 0.25
    closed_cosine_threshold: float = 0.30
    thumb_lateral_threshold: float = 0.25


@dataclass
class RecognitionConfig:
    mode: str = "ml"                       # "ml" | "rule_based"
    model_path: str = "models/NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl"
    normal_min_confidence: float = 0.60


@dataclass
class GrabCalibrationConfig:
    """
    Thresholds for the rule-based GRAB detector.
    # CALIBRATION: adjust per physical robotic hand once range of motion is known.
    These are mid-range defaults — not engineering specifications.
    """
    min_extension_ratio: float = 0.20
    max_extension_ratio: float = 0.55
    min_cosine: float = 0.10
    max_cosine: float = 0.60


@dataclass
class StabilizationConfig:
    window_size: int = 20
    vote_threshold: float = 0.70
    min_frames_required: int = 5


@dataclass
class StopConfig:
    arm_key: str = "space"
    arm_timeout_ms: int = 5000
    min_frames: int = 3
    min_confidence: float = 0.85
    hold_ms: int = 1500


@dataclass
class SerialConfig:
    enabled: bool = True
    port: str = "COM3"
    baud_rate: int = 115200
    timeout_s: float = 1.0
    mock: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class VisualizationConfig:
    enabled: bool = True
    window_title: str = "NeuroGrip CV"
    show_landmarks: bool = True
    show_skeleton: bool = True
    show_fps: bool = True
    show_confidence: bool = True
    show_finger_states: bool = True
    show_raw_prediction: bool = True
    show_stabilization: bool = True


# ─────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    grab_calibration: GrabCalibrationConfig = field(default_factory=GrabCalibrationConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load config from a YAML file, falling back to defaults for missing keys."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file not found at '%s'. Using defaults.", path)
            return cls()

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            camera=_load_dc(CameraConfig, data.get("camera", {})),
            hand_tracking=_load_dc(HandTrackingConfig, data.get("hand_tracking", {})),
            features=_load_dc(FeaturesConfig, data.get("features", {})),
            recognition=_load_dc(RecognitionConfig, data.get("recognition", {})),
            grab_calibration=_load_dc(GrabCalibrationConfig, data.get("grab_calibration", {})),
            stabilization=_load_dc(StabilizationConfig, data.get("stabilization", {})),
            stop=_load_dc(StopConfig, data.get("stop", {})),
            serial=_load_dc(SerialConfig, data.get("serial", {})),
            logging=_load_dc(LoggingConfig, data.get("logging", {})),
            visualization=_load_dc(VisualizationConfig, data.get("visualization", {})),
        )

    @classmethod
    def default(cls) -> "AppConfig":
        """Return default configuration."""
        default_yaml = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
        return cls.from_yaml(default_yaml)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _load_dc(dc_cls, data: dict):
    """
    Instantiate a dataclass from a dict, ignoring unknown keys and filling
    missing fields with their declared defaults.
    """
    valid_fields = {f.name for f in dataclasses.fields(dc_cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return dc_cls(**filtered)


def configure_logging(config: LoggingConfig) -> None:
    """Apply logging configuration from AppConfig."""
    level = getattr(logging, config.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.file:
        handlers.append(logging.FileHandler(config.file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def resolve_model_path(model_path: str | Path) -> Path:
    """
    Resolve a model or asset path robustly regardless of current working directory (CWD).

    Resolution Order:
    1. Absolute path: return directly.
    2. Relative to CWD: (Path.cwd() / path).resolve()
    3. Relative to 'pc' package root directory: (pc_root / path).resolve()
    4. Relative to repository root directory: (repo_root / path).resolve()
    5. Fallback: CWD-resolved path.
    """
    p = Path(model_path)
    if p.is_absolute():
        return p

    # Candidate 1: CWD-relative resolution
    cwd_path = p.resolve()
    if cwd_path.exists():
        return cwd_path

    # Candidate 2: pc/ package root resolution (parents[3] of settings.py)
    pc_root = Path(__file__).resolve().parents[3]
    pc_path = (pc_root / p).resolve()
    if pc_path.exists():
        return pc_path

    # Candidate 3: Repo root resolution (parent of pc_root)
    repo_root = pc_root.parent
    repo_path = (repo_root / p).resolve()
    if repo_path.exists():
        return repo_path

    # Fallback to CWD resolution if file does not exist yet
    return cwd_path

