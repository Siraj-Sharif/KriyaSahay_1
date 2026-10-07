"""
neurogrip/visualization/overlay.py
───────────────────────────────────
High-level Overlay Helper and Optional GUI Window Manager.
Provides optional OpenCV window display wrappers for live webcam execution.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from neurogrip.config.settings import AppConfig, VisualizationConfig
from neurogrip.visualization.base import VisualizationState
from neurogrip.visualization.renderer import OverlayRenderer

logger = logging.getLogger(__name__)


class OverlayWindow:
    """
    Optional GUI window wrapper for live OpenCV rendering.
    Fully isolated from core headless rendering math.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        self.config = config or AppConfig.default().visualization
        self.renderer = OverlayRenderer(self.config)
        self.window_name = self.config.window_title
        self._is_window_created = False

    def show(self, frame: np.ndarray, state: VisualizationState, wait_key_ms: int = 1) -> int:
        """
        Render overlay onto frame and display in an OpenCV GUI window.

        Parameters
        ----------
        frame : np.ndarray
            Input frame array.
        state : VisualizationState
            Snapshot of current pipeline and gesture state.
        wait_key_ms : int
            OpenCV waitKey delay in milliseconds.

        Returns
        -------
        int
            ASCII keycode pressed by user, or -1 if no key pressed.
        """
        rendered = self.renderer.render(frame, state)

        if not self.config.enabled:
            return -1

        try:
            if not self._is_window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._is_window_created = True

            cv2.imshow(self.window_name, rendered)
            key = cv2.waitKey(wait_key_ms) & 0xFF
            return key
        except Exception as e:
            logger.warning("GUI window display unavailable (headless environment?): %s", e)
            return -1

    def close(self) -> None:
        """Close OpenCV GUI window."""
        if self._is_window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            finally:
                self._is_window_created = False
