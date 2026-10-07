"""
neurogrip/camera/mock_camera.py
───────────────────────────────
Mock Camera Capture Backend.
Delivers synthetic or pre-loaded frames for testing, simulation, and hardware-free execution.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from neurogrip.camera.base import CameraInterface, CameraState, FrameContainer
from neurogrip.config.settings import AppConfig, CameraConfig

logger = logging.getLogger(__name__)


class MockCamera(CameraInterface):
    """
    Mock camera device for unit testing and offline demo loops.

    Parameters
    ----------
    config : Optional[CameraConfig]
        Camera configuration settings.
    frames : Optional[list[np.ndarray]]
        Pre-supplied sequence of images to deliver sequentially.
    max_frames : Optional[int]
        Limit total delivered frames (simulates end of video stream).
    simulate_failure_at : Optional[int]
        Simulate frame capture failure at a specific 0-based frame index.
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        frames: Optional[list[np.ndarray]] = None,
        max_frames: Optional[int] = None,
        simulate_failure_at: Optional[int] = None,
    ) -> None:
        self.config = config or AppConfig.default().camera
        self._state: CameraState = CameraState.DISCONNECTED
        self._frame_count: int = 0
        self._custom_frames = frames
        self.max_frames = max_frames
        self.simulate_failure_at = simulate_failure_at
        self._start_time_ms: int = 0

    @property
    def is_opened(self) -> bool:
        return self._state == CameraState.MOCK

    @property
    def state(self) -> CameraState:
        return self._state

    def open(self) -> bool:
        self._state = CameraState.MOCK
        self._frame_count = 0
        self._start_time_ms = int(time.time() * 1000)
        logger.info("MockCamera opened (res: %dx%d @ %d FPS).", self.config.width, self.config.height, self.config.fps)
        return True

    def close(self) -> None:
        self._state = CameraState.DISCONNECTED
        logger.info("MockCamera closed.")

    def read_frame(self) -> FrameContainer:
        if not self.is_opened:
            logger.warning("MockCamera: read_frame called while camera is disconnected.")
            return FrameContainer(frame=None, timestamp_ms=0, frame_index=self._frame_count)

        # Check end of stream limit
        if self.max_frames is not None and self._frame_count >= self.max_frames:
            logger.debug("MockCamera: End of stream reached (max_frames=%d).", self.max_frames)
            return FrameContainer(frame=None, timestamp_ms=self._get_timestamp(), frame_index=self._frame_count)

        # Check simulated failure
        if self.simulate_failure_at is not None and self._frame_count == self.simulate_failure_at:
            logger.warning("MockCamera: Simulated frame read failure at index %d.", self._frame_count)
            idx = self._frame_count
            self._frame_count += 1
            return FrameContainer(frame=None, timestamp_ms=self._get_timestamp(), frame_index=idx)

        # Select frame to return
        if self._custom_frames:
            idx = self._frame_count % len(self._custom_frames)
            frame_data = self._custom_frames[idx].copy()
        else:
            # Generate synthetic test pattern (solid background with frame index indicator)
            frame_data = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            # Add subtle color variations based on frame index
            frame_data[:, :, 0] = (self._frame_count * 5) % 255

        idx = self._frame_count
        timestamp = self._get_timestamp()
        self._frame_count += 1

        return FrameContainer(frame=frame_data, timestamp_ms=timestamp, frame_index=idx)

    def _get_timestamp(self) -> int:
        fps = self.config.fps if self.config.fps > 0 else 30
        frame_interval_ms = int(1000.0 / fps)
        return self._start_time_ms + (self._frame_count * frame_interval_ms)
