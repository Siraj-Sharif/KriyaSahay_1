"""
neurogrip/visualization/base.py
───────────────────────────────
Visualization state structures and abstract renderer interface.
Observation/Sink layer — decoupled from MediaPipe, recognizers, and serial communication.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from neurogrip.hand_tracking.landmarks import HandLandmarks


@dataclass(frozen=True)
class VisualizationState:
    """
    Immutable data snapshot required by the visualization overlay renderer.

    Attributes
    ----------
    command : str
        Active/emitted command name (e.g. "INDEX_FINGER", "CLOSED_FIST", "STOP", "NO_COMMAND", "AMBIGUOUS", "NO_HAND").
    confidence : float
        Prediction/stability confidence score in range [0.0, 1.0].
    stabilizer_state : str
        State string of the temporal stabilizer (e.g. "STABLE", "STABILIZING", "STOP", "UNKNOWN_STATE").
    pipeline_state : str
        Execution state string of the pipeline (e.g. "TRACKING", "AMBIGUOUS", "NO_HAND", "STOP_ACTIVE").
    handedness : str
        Primary hand handedness string ("RIGHT", "LEFT", "UNKNOWN").
    landmarks : list[HandLandmarks]
        List of internal HandLandmarks detected in the frame.
    fps : float
        Real-time processing frame rate (FPS).
    is_stop_active : bool
        True if software-armed STOP state is active.
    is_ambiguous : bool
        True if 2+ hands are detected in frame.
    message : Optional[str]
        Optional custom status/diagnostic message to display on overlay.
    model_used : str
        Model architecture used ("HAGRID", "EXTRA_TREES", "RULE_BASED", "N/A").
    raw_prediction : str
        Original unmapped prediction string.
    hand_count: str
        Hand count string ("0", "1", "2+").
    latency_ms : float
        Frame latency in milliseconds.
    safety_status : str
        STOP safety arming status ("STOP ARMED", "STOP DISARMED", "STOP ACTIVE").
    routing_str : str
        Routing indicator string (e.g. "HAGRID -> GRIP", "EXTRA_TREES -> PINKY").
    """
    command: str = "UNKNOWN"
    confidence: float = 0.0
    stabilizer_state: str = "UNKNOWN_STATE"
    pipeline_state: str = "STARTING"
    handedness: str = "UNKNOWN"
    landmarks: list[HandLandmarks] = field(default_factory=list)
    fps: float = 0.0
    is_stop_active: bool = False
    is_ambiguous: bool = False
    message: Optional[str] = None
    model_used: str = "N/A"
    raw_prediction: str = "N/A"
    hand_count: str = "0"
    latency_ms: float = 0.0
    safety_status: str = "ARMED"
    routing_str: str = "N/A"
    transport_status: str = "DISCONNECTED"
    last_tx: str = "NONE"
    stop_rule_triggered: bool = False
    stop_rule_metrics: str = "N/A"
    hagrid_raw: str = "N/A"
    hagrid_conf: float = 0.0
    taxonomy_mapped: str = "NO_COMMAND"
    is_armed: bool = False
    is_stabilized: bool = False
    final_command: str = "NO_COMMAND"
    is_tx_permitted: bool = False



class VisualizationInterface(ABC):
    """
    Abstract interface for overlay renderers.
    """

    @abstractmethod
    def render(self, frame: np.ndarray, state: VisualizationState) -> np.ndarray:
        """
        Draw overlay elements onto a copy of the input image frame.

        Parameters
        ----------
        frame : np.ndarray
            Input OpenCV image array (BGR format).
        state : VisualizationState
            Snapshot of current pipeline and gesture state.

        Returns
        -------
        np.ndarray
            Rendered image array with drawn overlay elements.
        """
        pass
