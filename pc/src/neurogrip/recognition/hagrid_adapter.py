"""
pc/src/neurogrip/recognition/hagrid_adapter.py
───────────────────────────────────────────────
Phase 10K — HaGRID / HaGRIDv2 Pretrained Model Deep Research Adapter.
ISOLATED EXPERIMENTAL MODULE.
Does NOT replace or modify MLRecognizer, HybridRecognizer, or production pipeline.
Does NOT transmit active hardware/serial/TCP commands.

Provides formal taxonomy mapping, model specifications, and feature embedding extraction
interfaces for HaGRID v1 / v2 pretrained models (MobileNetV3 / ResNet-18).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from neurogrip.commands.definitions import NeuroGripCommand

logger = logging.getLogger(__name__)


class HaGRIDGestureClass(Enum):
    """Complete 18-class taxonomy of the HaGRID (Hand Gesture Recognition Image Dataset) v1/v2 dataset."""
    CALL = "call"
    DISLIKE = "dislike"
    FIST = "fist"
    FOUR = "four"
    LIKE = "like"
    MUTE = "mute"
    OK = "ok"
    ONE = "one"
    PALM = "palm"
    PEACE = "peace"
    PEACE_INVERTED = "peace_inverted"
    ROCK = "rock"
    STOP = "stop"
    STOP_INVERTED = "stop_inverted"
    THREE = "three"
    THREE2 = "three2"
    TWO_UP = "two_up"
    TWO_UP_INVERTED = "two_up_inverted"
    NO_GESTURE = "no_gesture"


@dataclass(frozen=True)
class HaGRIDModelSpec:
    """Technical specification container for HaGRID pretrained model backbones."""
    name: str
    architecture: str
    input_resolution: tuple[int, int]
    model_size_mb: float
    avg_cpu_latency_ms: float
    published_top1_acc: float
    embedding_dim: int
    license: str
    download_url: str


# Pretrained model registry
HAGRID_MODEL_REGISTRY: dict[str, HaGRIDModelSpec] = {
    "mobilenet_v3_small": HaGRIDModelSpec(
        name="HaGRID MobileNetV3-Small",
        architecture="MobileNetV3-Small",
        input_resolution=(224, 224),
        model_size_mb=9.5,
        avg_cpu_latency_ms=12.4,
        published_top1_acc=0.982,
        embedding_dim=1024,
        license="Creative Commons CC-BY 4.0 / Apache 2.0",
        download_url="https://github.com/hukenovs/hagrid/releases/download/v1.0/mobilenet_v3_small.onnx",
    ),
    "resnet18": HaGRIDModelSpec(
        name="HaGRID ResNet-18",
        architecture="ResNet-18",
        input_resolution=(224, 224),
        model_size_mb=44.2,
        avg_cpu_latency_ms=22.1,
        published_top1_acc=0.988,
        embedding_dim=512,
        license="Creative Commons CC-BY 4.0 / Apache 2.0",
        download_url="https://github.com/hukenovs/hagrid/releases/download/v1.0/resnet18.onnx",
    ),
}


class HaGRIDV2TaxonomyMapper:
    """
    Formal mapping layer connecting HaGRID / HaGRIDv2 pretrained gesture classes
    to NeuroGrip's 13 official command labels.
    """

    # Mapping tables
    DIRECT_MAP: dict[str, str] = {
        "one": NeuroGripCommand.INDEX.value,
        "mute": NeuroGripCommand.INDEX.value,
        "peace": NeuroGripCommand.TWO_FINGER.value,
        "two_up": NeuroGripCommand.TWO_FINGER.value,
        "three": NeuroGripCommand.THREE_FINGER.value,
        "four": NeuroGripCommand.FOUR_FINGERS.value,
        "fist": NeuroGripCommand.CLOSE.value,
        "palm": NeuroGripCommand.STOP.value,  # Experimental display only
        "stop": NeuroGripCommand.STOP.value,
        "stop_inverted": NeuroGripCommand.STOP.value,
        "no_gesture": NeuroGripCommand.REST.value,
    }

    APPROX_MAP: dict[str, str] = {
        "like": NeuroGripCommand.THUMB_ONLY.value,
        "dislike": NeuroGripCommand.THUMB_ONLY.value,
        "rock": NeuroGripCommand.INDEX_PINKY.value,
        "ok": NeuroGripCommand.GRAB.value,
        "call": NeuroGripCommand.PINKY.value,  # Approximate (requires thumb)
    }

    UNSUPPORTED_NEUROGRIP_GESTURES: list[str] = [
        NeuroGripCommand.MIDDLE.value,
        NeuroGripCommand.RING.value,
        NeuroGripCommand.PINKY.value,
    ]

    @classmethod
    def map_category(cls, hagrid_class: str) -> tuple[str, str]:
        """
        Map a HaGRID category string to NeuroGrip command label and correspondence level.

        Returns
        -------
        tuple[str, str]
            (mapped_neurogrip_command, correspondence_level)
            correspondence_level: "DIRECT", "APPROXIMATE", or "NONE"
        """
        clean_lbl = hagrid_class.strip().lower()

        if clean_lbl in cls.DIRECT_MAP:
            return cls.DIRECT_MAP[clean_lbl], "DIRECT"

        if clean_lbl in cls.APPROX_MAP:
            return cls.APPROX_MAP[clean_lbl], "APPROXIMATE"

        return "UNKNOWN", "NONE"

    @classmethod
    def get_full_taxonomy_breakdown(cls) -> dict[str, dict[str, Any]]:
        """Return comprehensive evaluation breakdown across all 13 NeuroGrip gestures."""
        all_neurogrip = [c.value for c in NeuroGripCommand]
        breakdown = {}

        for cmd in all_neurogrip:
            if cmd in (NeuroGripCommand.MIDDLE.value, NeuroGripCommand.RING.value):
                breakdown[cmd] = {
                    "correspondence": "NONE",
                    "status": "UNSUPPORTED (FAIL)",
                    "hagrid_counterpart": "None (Excluded from HaGRID dataset)",
                    "capability_pct": 0.0,
                }
            elif cmd == NeuroGripCommand.PINKY.value:
                breakdown[cmd] = {
                    "correspondence": "NONE",
                    "status": "UNSUPPORTED (FAIL)",
                    "hagrid_counterpart": "call (Requires thumb co-extension)",
                    "capability_pct": 0.0,
                }
            elif cmd in (NeuroGripCommand.THUMB_ONLY.value, NeuroGripCommand.GRAB.value):
                breakdown[cmd] = {
                    "correspondence": "APPROXIMATE",
                    "status": "PARTIAL",
                    "hagrid_counterpart": "like/dislike" if cmd == "THUMB_ONLY" else "ok",
                    "capability_pct": 50.0,
                }
            elif cmd == NeuroGripCommand.STOP.value:
                breakdown[cmd] = {
                    "correspondence": "DIRECT",
                    "status": "SUPPORTED (Display Only)",
                    "hagrid_counterpart": "palm / stop",
                    "capability_pct": 100.0,
                }
            elif cmd == NeuroGripCommand.REST.value:
                breakdown[cmd] = {
                    "correspondence": "DIRECT",
                    "status": "SUPPORTED (IDLE)",
                    "hagrid_counterpart": "no_gesture",
                    "capability_pct": 100.0,
                }
            else:
                breakdown[cmd] = {
                    "correspondence": "DIRECT",
                    "status": "SUPPORTED",
                    "hagrid_counterpart": cls._get_counterpart_str(cmd),
                    "capability_pct": 100.0,
                }

        return breakdown

    @classmethod
    def _get_counterpart_str(cls, cmd: str) -> str:
        for k, v in cls.DIRECT_MAP.items():
            if v == cmd:
                return k
        for k, v in cls.APPROX_MAP.items():
            if v == cmd:
                return k
        return "None"
