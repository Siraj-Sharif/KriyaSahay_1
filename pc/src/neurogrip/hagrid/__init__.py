"""HaGRID Gesture Recognition Package for NeuroGrip.

Public names are re-exported lazily (PEP 562). Importing a light submodule such as
``neurogrip.hagrid.taxonomy`` therefore does NOT pull in the heavy optional runtime
dependencies (torch, torchvision, ultralytics, opencv). This keeps the command,
protocol and vocabulary layers importable with only the core requirements installed,
while ``from neurogrip.hagrid import HaGRIDClassifier`` continues to work unchanged.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing aid only, never executed at runtime
    from neurogrip.hagrid.benchmark import (
        crop_from_box,
        predict_architecture_a,
        predict_architecture_b,
    )
    from neurogrip.hagrid.config import HaGRIDConfig
    from neurogrip.hagrid.detector import HaGRIDHandDetector
    from neurogrip.hagrid.model import HaGRIDClassifier

__all__ = [
    "HaGRIDConfig",
    "HaGRIDClassifier",
    "HaGRIDHandDetector",
    "HAGRID_CLASSES",
    "HAGRID_INDEX_TO_CLASS",
    "HAGRID_CLASS_TO_INDEX",
    "NEUROGRIP_TAXONOMY",
    "NO_COMMAND",
    "CanonicalCommand",
    "canonical_from_string",
    "map_hagrid_to_neurogrip",
    "is_valid_neurogrip_command",
    "crop_from_box",
    "predict_architecture_a",
    "predict_architecture_b",
]

# name -> (module, attribute)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Light-weight taxonomy symbols (no heavy dependencies)
    "HAGRID_CLASSES": ("neurogrip.hagrid.taxonomy", "HAGRID_CLASSES"),
    "HAGRID_INDEX_TO_CLASS": ("neurogrip.hagrid.taxonomy", "HAGRID_INDEX_TO_CLASS"),
    "HAGRID_CLASS_TO_INDEX": ("neurogrip.hagrid.taxonomy", "HAGRID_CLASS_TO_INDEX"),
    "NEUROGRIP_TAXONOMY": ("neurogrip.hagrid.taxonomy", "NEUROGRIP_TAXONOMY"),
    "NO_COMMAND": ("neurogrip.hagrid.taxonomy", "NO_COMMAND"),
    "CanonicalCommand": ("neurogrip.hagrid.taxonomy", "CanonicalCommand"),
    "canonical_from_string": ("neurogrip.hagrid.taxonomy", "canonical_from_string"),
    "map_hagrid_to_neurogrip": ("neurogrip.hagrid.taxonomy", "map_hagrid_to_neurogrip"),
    "is_valid_neurogrip_command": ("neurogrip.hagrid.taxonomy", "is_valid_neurogrip_command"),
    # Configuration
    "HaGRIDConfig": ("neurogrip.hagrid.config", "HaGRIDConfig"),
    # Heavy model / detector / benchmark symbols (torch, ultralytics, opencv)
    "HaGRIDClassifier": ("neurogrip.hagrid.model", "HaGRIDClassifier"),
    "HaGRIDHandDetector": ("neurogrip.hagrid.detector", "HaGRIDHandDetector"),
    "crop_from_box": ("neurogrip.hagrid.benchmark", "crop_from_box"),
    "predict_architecture_a": ("neurogrip.hagrid.benchmark", "predict_architecture_a"),
    "predict_architecture_b": ("neurogrip.hagrid.benchmark", "predict_architecture_b"),
}


def __getattr__(name: str) -> Any:
    """Resolve public package attributes on first access (PEP 562)."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value  # cache so subsequent lookups skip this function
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
