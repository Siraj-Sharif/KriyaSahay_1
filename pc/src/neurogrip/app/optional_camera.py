"""Desktop startup adapter for an optional physical camera.

The application must remain available for serial/voice controls when a camera is not
connected. ``NeuroGripPipeline.initialize`` historically treats an unavailable camera as a
fatal startup error, so desktop mode wraps the existing camera for only the first open
attempt: one missing device does not prevent the pipeline/control channel from starting.
All successful opens, reads, selection and release are still delegated to the original
camera object. No frames are synthesized and no CV processing is performed here.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from neurogrip.camera.base import CameraInterface, CameraState, FrameContainer

logger = logging.getLogger(__name__)


class OptionalStartupCamera(CameraInterface):
    """Delegate capture to a real camera while tolerating one unavailable startup attempt."""

    def __init__(self, camera: CameraInterface) -> None:
        self._camera = camera
        self._allow_initial_unavailable = True
        self._empty_frame_index = 0

    @property
    def config(self) -> Any:
        """Expose the wrapped camera configuration used by pipeline camera controls."""
        return self._camera.config

    @property
    def is_opened(self) -> bool:
        return self._camera.is_opened

    @property
    def state(self) -> CameraState:
        return self._camera.state

    @property
    def actual_width(self) -> int:
        return int(getattr(self._camera, "actual_width", 0) or 0)

    @property
    def actual_height(self) -> int:
        return int(getattr(self._camera, "actual_height", 0) or 0)

    @property
    def actual_fps(self) -> float:
        return float(getattr(self._camera, "actual_fps", 0.0) or 0.0)

    def open(self) -> bool:
        opened = bool(self._camera.open())
        if opened:
            self._allow_initial_unavailable = False
            return True

        if self._allow_initial_unavailable:
            self._allow_initial_unavailable = False
            logger.warning(
                "[CAMERA] No camera opened during desktop startup; continuing with camera unavailable. "
                "Camera controls can retry after a device is selected or connected."
            )
            # This return value only tells the legacy application initializer that the
            # optional camera must not abort unrelated backend startup. is_opened/state
            # continue to report the real unavailable device.
            return True

        return False

    def close(self) -> None:
        self._camera.close()

    def read_frame(self) -> FrameContainer:
        if not self._camera.is_opened:
            self._empty_frame_index += 1
            return FrameContainer(
                frame=None,
                timestamp_ms=int(time.time() * 1000),
                frame_index=self._empty_frame_index - 1,
            )
        return self._camera.read_frame()
