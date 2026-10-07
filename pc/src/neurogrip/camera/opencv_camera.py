"""
neurogrip/camera/opencv_camera.py
─────────────────────────────────
OpenCV Camera Capture Backend.
Wraps cv2.VideoCapture to provide robust frame acquisition from local USB webcams.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from neurogrip.camera.base import CameraInterface, CameraState, FrameContainer
from neurogrip.config.settings import AppConfig, CameraConfig

logger = logging.getLogger(__name__)


class OpenCVCamera(CameraInterface):
    """
    OpenCV camera device wrapping cv2.VideoCapture.

    Parameters
    ----------
    config : Optional[CameraConfig]
        Camera configuration settings (index, width, height, fps).
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or AppConfig.default().camera
        self._cap: Optional[cv2.VideoCapture] = None
        self._state: CameraState = CameraState.DISCONNECTED
        self._frame_count: int = 0
        self._start_time_ms: int = 0

        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

    @property
    def is_opened(self) -> bool:
        return (
            self._state == CameraState.CONNECTED
            and self._cap is not None
            and self._cap.isOpened()
        )

    @property
    def state(self) -> CameraState:
        return self._state

    def open(self) -> bool:
        """
        Open the OpenCV video capture device and configure target resolution/FPS.

        Returns
        -------
        bool
            True if camera opened successfully, False otherwise.
        """
        if self.is_opened:
            return True

        self._state = CameraState.CONNECTING
        logger.info("Opening OpenCV camera index %d...", self.config.index)

        try:
            self._cap = cv2.VideoCapture(self.config.index)

            if not self._cap or not self._cap.isOpened():
                logger.warning("Camera index %d is unavailable or disabled in Windows.", self.config.index)
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None

                # Auto-scan candidate camera indices (0 to 9) for an available enabled device
                candidate_indices = [i for i in range(10) if i != self.config.index]
                for cand_idx in candidate_indices:
                    logger.info("Attempting auto-selection for camera index %d...", cand_idx)
                    try:
                        test_cap = cv2.VideoCapture(cand_idx)
                        if test_cap and test_cap.isOpened():
                            self._cap = test_cap
                            self.config.index = cand_idx
                            logger.info("Successfully auto-selected enabled camera device at index %d.", cand_idx)
                            break
                        else:
                            if test_cap is not None:
                                test_cap.release()
                    except Exception as scan_err:
                        logger.debug("Camera candidate index %d failed: %s", cand_idx, scan_err)

            if not self._cap or not self._cap.isOpened():
                logger.error("Failed to open any available or enabled camera device.")
                self._state = CameraState.ERROR
                self._cap = None
                return False

            # Configure target camera parameters
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
            self._cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))

            # Query actual hardware properties
            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))

            self._state = CameraState.CONNECTED
            self._frame_count = 0
            self._start_time_ms = int(time.time() * 1000)

            logger.info(
                "Camera %d opened successfully (Requested: %dx%d @ %d FPS | Actual: %dx%d @ %.1f FPS).",
                self.config.index,
                self.config.width,
                self.config.height,
                self.config.fps,
                self.actual_width,
                self.actual_height,
                self.actual_fps,
            )
            return True

        except Exception as e:
            logger.error("Exception while opening camera: %s", e)
            self._state = CameraState.ERROR
            self._cap = None
            return False

    def close(self) -> None:
        """Release underlying OpenCV VideoCapture device."""
        if self._cap is not None:
            try:
                if self._cap.isOpened():
                    self._cap.release()
            except Exception as e:
                logger.warning("Error releasing camera device: %s", e)
            finally:
                self._cap = None

        self._state = CameraState.DISCONNECTED
        logger.info("Camera %d released.", self.config.index)

    def read_frame(self) -> FrameContainer:
        """
        Capture the next frame from the camera device.

        Returns
        -------
        FrameContainer
            Container with OpenCV frame array, timestamp, and frame index.
        """
        if not self.is_opened or self._cap is None:
            logger.warning("read_frame called while camera is disconnected.")
            return FrameContainer(frame=None, timestamp_ms=0, frame_index=self._frame_count)

        try:
            ret, frame = self._cap.read()
            timestamp_ms = int(time.time() * 1000)

            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to read frame from camera %d.", self.config.index)
                idx = self._frame_count
                self._frame_count += 1
                return FrameContainer(frame=None, timestamp_ms=timestamp_ms, frame_index=idx)

            idx = self._frame_count
            self._frame_count += 1
            return FrameContainer(frame=frame, timestamp_ms=timestamp_ms, frame_index=idx)

        except Exception as e:
            logger.error("Exception reading frame from camera %d: %s", self.config.index, e)
            return FrameContainer(frame=None, timestamp_ms=int(time.time() * 1000), frame_index=self._frame_count)
