"""
neurogrip/camera/base.py
────────────────────────
Abstract base class, camera state definitions, and frame container.
Decouples OpenCV camera capture from MediaPipe hand tracking and downstream CV modules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CameraState(Enum):
    """Camera capture lifecycle states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()
    MOCK = auto()


@dataclass(frozen=True)
class FrameContainer:
    """
    Immutable container wrapping a single captured camera frame.

    Attributes
    ----------
    frame : Optional[np.ndarray]
        OpenCV image array (BGR format) or None if read failed.
    timestamp_ms : int
        Monotonically increasing timestamp in milliseconds.
    frame_index : int
        0-indexed frame count.
    """
    frame: Optional[np.ndarray]
    timestamp_ms: int
    frame_index: int

    @property
    def is_valid(self) -> bool:
        """True if frame is present and non-empty."""
        return self.frame is not None and self.frame.size > 0


class CameraInterface(ABC):
    """
    Abstract interface for camera capture devices.
    """

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """True if the camera capture device is opened and ready to deliver frames."""
        pass

    @property
    @abstractmethod
    def state(self) -> CameraState:
        """Current operational state of the camera."""
        pass

    @abstractmethod
    def open(self) -> bool:
        """Initialize and open the camera device."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release the camera device and free underlying hardware resources."""
        pass

    @abstractmethod
    def read_frame(self) -> FrameContainer:
        """
        Capture and return the next video frame.

        Returns
        -------
        FrameContainer
            Container containing captured image array, timestamp, and frame index.
        """
        pass

    def __enter__(self) -> "CameraInterface":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
