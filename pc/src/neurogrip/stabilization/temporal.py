"""
neurogrip/stabilization/temporal.py
───────────────────────────────────
Temporal Stabilization & Safety Path Module.
Applies a majority-vote sliding window over recognition predictions to prevent frame flickering,
while allowing software-armed STOP commands to bypass the window immediately.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from typing import Optional

from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.config.settings import AppConfig, StabilizationConfig
from neurogrip.recognition.base import RecognitionResult

logger = logging.getLogger(__name__)


class StabilizerState(Enum):
    """Conceptual states of the temporal stabilizer state machine."""
    UNKNOWN_STATE = auto()  # Window empty or dominated by UNKNOWN / below threshold
    STABILIZING = auto()     # Valid predictions present, window building toward threshold
    STABLE = auto()          # Majority candidate reached or exceeded stability threshold
    STOP = auto()            # Software-armed STOP safety bypass triggered


@dataclass(frozen=True)
class StabilizerResult:
    """
    Output emitted by TemporalStabilizer after processing a frame prediction.

    Attributes
    ----------
    state : StabilizerState
        Current state of the stabilizer state machine.
    stable_command : Optional[str]
        The currently stable command string (e.g. "INDEX"), or None.
    emitted_command : Optional[str]
        Command to emit for transmission ONCE upon transition, or None if suppressed.
    confidence : float
        Proportion of window votes held by the winning candidate [0.0, 1.0].
    window_fill : int
        Current number of frames stored in the sliding window buffer.
    winning_count : int
        Number of window votes held by the winning candidate.
    """
    state: StabilizerState
    stable_command: Optional[str]
    emitted_command: Optional[str]
    confidence: float
    window_fill: int
    winning_count: int


class TemporalStabilizer:
    """
    Deterministic majority-vote temporal stabilizer.

    Parameters
    ----------
    config : Optional[AppConfig]
        Application configuration.
    window_size : Optional[int]
        Override sliding window size in frames (defaults to config value).
    vote_threshold : Optional[float]
        Override stability threshold fraction in range (0.0, 1.0] (defaults to config value).
    min_frames_required : Optional[int]
        Minimum window count before stability can be achieved (defaults to config value).
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        window_size: Optional[int] = None,
        vote_threshold: Optional[float] = None,
        min_frames_required: Optional[int] = None,
    ) -> None:
        app_cfg = config or AppConfig.default()
        stab_cfg: StabilizationConfig = app_cfg.stabilization

        self.window_size: int = window_size if window_size is not None else stab_cfg.window_size
        self.vote_threshold: float = vote_threshold if vote_threshold is not None else stab_cfg.vote_threshold
        self.min_frames_required: int = (
            min_frames_required if min_frames_required is not None else stab_cfg.min_frames_required
        )

        self._window: deque[str] = deque(maxlen=self.window_size)
        self._last_emitted_command: Optional[str] = None
        self._current_state: StabilizerState = StabilizerState.UNKNOWN_STATE
        self._current_stable_command: Optional[str] = None

    @property
    def current_state(self) -> StabilizerState:
        return self._current_state

    @property
    def last_emitted_command(self) -> Optional[str]:
        return self._last_emitted_command

    def reset(self) -> None:
        """Reset stabilizer window, state machine, and emission history."""
        self._window.clear()
        self._last_emitted_command = None
        self._current_state = StabilizerState.UNKNOWN_STATE
        self._current_stable_command = None
        logger.debug("TemporalStabilizer state reset.")

    def update(
        self,
        result: Optional[RecognitionResult],
        is_stop_armed: bool = False,
    ) -> StabilizerResult:
        """
        Update the sliding window with a new recognition prediction and evaluate stability.

        Parameters
        ----------
        result : Optional[RecognitionResult]
            Recognition result from the upstream recognizer.
        is_stop_armed : bool
            Flag indicating if STOP detection is armed in software.

        Returns
        -------
        StabilizerResult
            Current state machine state, stable command, emitted command, and window metrics.
        """
        # 1. Null / Invalid result handling
        if result is None or result.is_unknown:
            candidate_label = "UNKNOWN"
        else:
            candidate_label = str(result.label).upper()

        # 2. SAFETY PATH: STOP Immediate Bypass Rule
        # If upstream recognition predicts STOP
        if candidate_label == NeuroGripCommand.STOP.value:
            self._window.clear()  # Clear normal gesture window
            self._current_state = StabilizerState.STOP
            self._current_stable_command = NeuroGripCommand.STOP.value

            # Determine command emission (STOP is emitted once on transition)
            emitted = None
            if self._last_emitted_command != NeuroGripCommand.STOP.value:
                emitted = NeuroGripCommand.STOP.value
                self._last_emitted_command = NeuroGripCommand.STOP.value

            logger.info("STOP gesture detected and stabilized.")
            return StabilizerResult(
                state=StabilizerState.STOP,
                stable_command=NeuroGripCommand.STOP.value,
                emitted_command=emitted,
                confidence=1.0,
                window_fill=0,
                winning_count=1,
            )

        # 3. NORMAL GESTURE PATH: Push to sliding window
        self._window.append(candidate_label)
        n_window = len(self._window)

        # Count votes for each candidate in window
        counts = Counter(self._window)

        # Remove UNKNOWN from winning candidates (UNKNOWN counts toward total N, but cannot win)
        valid_counts = {cmd: count for cmd, count in counts.items() if cmd != "UNKNOWN"}

        if not valid_counts or n_window == 0:
            self._current_state = StabilizerState.UNKNOWN_STATE
            self._current_stable_command = None
            return StabilizerResult(
                state=StabilizerState.UNKNOWN_STATE,
                stable_command=None,
                emitted_command=None,
                confidence=0.0,
                window_fill=n_window,
                winning_count=0,
            )

        # Find candidate with highest vote count
        top_cmd, top_count = max(valid_counts.items(), key=lambda item: item[1])
        confidence = float(top_count / n_window)

        # Check if threshold and min_frames requirements are met
        is_threshold_met = (
            confidence >= self.vote_threshold
            and n_window >= self.min_frames_required
        )

        if is_threshold_met:
            self._current_state = StabilizerState.STABLE
            self._current_stable_command = top_cmd

            # De-duplication check: Emit command ONLY if it differs from last emitted command
            emitted = None
            if top_cmd != self._last_emitted_command:
                emitted = top_cmd
                self._last_emitted_command = top_cmd

            return StabilizerResult(
                state=StabilizerState.STABLE,
                stable_command=top_cmd,
                emitted_command=emitted,
                confidence=confidence,
                window_fill=n_window,
                winning_count=top_count,
            )
        else:
            # Below threshold -> STABILIZING state
            self._current_state = StabilizerState.STABILIZING
            self._current_stable_command = None

            return StabilizerResult(
                state=StabilizerState.STABILIZING,
                stable_command=None,
                emitted_command=None,
                confidence=confidence,
                window_fill=n_window,
                winning_count=top_count,
            )
