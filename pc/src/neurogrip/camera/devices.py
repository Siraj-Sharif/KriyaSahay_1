"""
neurogrip/camera/devices.py
---------------------------
Best-effort camera device enumeration/labelling for the desktop UI.

OpenCV does not expose device names, so labels come from the platform enumeration when it
is available (pygrabber on Windows) and fall back to a plain index label otherwise. The
pipeline only ever reports indices it can actually open; this module only improves the
label shown next to them.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_windows_names: Optional[list[str]] = None
_probed_windows = False


def _windows_camera_names() -> list[str]:
    """Enumerate camera friendly names via DirectShow (Windows only, optional)."""
    global _windows_names, _probed_windows
    if _probed_windows:
        return _windows_names or []
    _probed_windows = True
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore

        _windows_names = list(FilterGraph().get_input_devices())
    except Exception:
        _windows_names = []
    return _windows_names


def describe_camera(index: int) -> Optional[str]:
    """Human-readable label for a camera index, or ``None`` when unknown."""
    names = _windows_camera_names()
    if 0 <= index < len(names):
        return names[index]
    return None
