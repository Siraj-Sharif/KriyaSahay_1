"""
neurogrip/commands/validator.py
────────────────────────────────
Command Validation Layer.
Validates commands against canonical definitions, current system state, and de-duplication rules
before allowing them to pass to the serial communication interface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

from neurogrip.commands.definitions import (
    NeuroGripCommand,
    command_name,
    resolve_command,
)
from neurogrip.hagrid.taxonomy import CanonicalCommand

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Explicit pipeline execution states."""
    STARTING = auto()
    INITIALIZING = auto()
    NO_HAND = auto()
    AMBIGUOUS = auto()
    TRACKING = auto()
    STABLE_CMD = auto()
    STOP_ARMED = auto()
    STOP_ACTIVE = auto()
    TRACKING_LOST = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a command candidate."""
    is_valid: bool
    command: Optional[Union[NeuroGripCommand, CanonicalCommand]]
    reason: str
    was_deduplicated: bool = False


class CommandValidator:
    """
    Validates candidate commands against:
    1. Vocabulary rules: the candidate must belong to either the legacy dataset
       vocabulary (``NeuroGripCommand``) or the locked canonical hardware
       vocabulary (``CanonicalCommand``, derived from ``NEUROGRIP_TAXONOMY``).
       Resolving against both vocabularies guarantees that every gesture the
       recognizers can emit (including CALL and OK) is also transmittable.
    2. Pipeline system state (e.g. STOP_ACTIVE, NO_HAND, AMBIGUOUS).
    3. De-duplication rules (consecutive duplicate commands are suppressed, except STOP).
    """

    def __init__(self) -> None:
        self._last_emitted_command: Optional[Union[NeuroGripCommand, CanonicalCommand]] = None

    @property
    def last_emitted_command(self) -> Optional[Union[NeuroGripCommand, CanonicalCommand]]:
        """Return the last successfully validated and emitted command."""
        return self._last_emitted_command

    def reset(self) -> None:
        """Reset de-duplication state (e.g. when tracking is lost)."""
        self._last_emitted_command = None
        logger.debug("CommandValidator de-duplication state reset.")

    def validate(
        self,
        command_candidate: Optional[Union[str, NeuroGripCommand, CanonicalCommand]],
        pipeline_state: PipelineState = PipelineState.TRACKING,
    ) -> ValidationResult:
        """
        Validate a candidate command.

        Parameters
        ----------
        command_candidate : str | NeuroGripCommand | CanonicalCommand | None
            The command candidate to validate.
        pipeline_state : PipelineState
            The current state of the vision pipeline.

        Returns
        -------
        ValidationResult
            Validation status, parsed command object (if valid), and reason for decision.
        """
        # 1. Handle system states that forbid command emission entirely
        if pipeline_state in (PipelineState.NO_HAND, PipelineState.AMBIGUOUS, PipelineState.ERROR):
            reason = f"Command rejected: pipeline state is {pipeline_state.name}."
            logger.debug(reason)
            return ValidationResult(is_valid=False, command=None, reason=reason)

        # 2. Null or empty candidate check
        if command_candidate is None:
            reason = "Command rejected: candidate is None."
            return ValidationResult(is_valid=False, command=None, reason=reason)

        # 3. Resolve candidate string/enum to a vocabulary member
        #    (legacy dataset vocabulary first, then the canonical hardware vocabulary)
        cmd_enum = resolve_command(command_candidate)

        if cmd_enum is None:
            reason = f"Command rejected: '{command_candidate}' is not a valid NeuroGrip command."
            logger.warning(reason)
            return ValidationResult(is_valid=False, command=None, reason=reason)

        # 4. Handle REST NO-COMMAND / IDLE state (REST must not produce active command)
        if cmd_enum == NeuroGripCommand.REST:
            reason = "Command rejected: REST represents NO-COMMAND / IDLE state."
            logger.debug(reason)
            return ValidationResult(is_valid=False, command=NeuroGripCommand.REST, reason=reason)

        # 5. De-duplication check
        # RULE: Duplicate consecutive commands are suppressed EXCEPT STOP, which is never de-duplicated.
        # Comparison is performed on the resolved command name so it is vocabulary-agnostic
        # (STOP stays deduplication-exempt no matter which vocabulary resolved it).
        if command_name(cmd_enum) != "STOP" and command_name(cmd_enum) == command_name(self._last_emitted_command):
            reason = f"Command '{cmd_enum.name}' suppressed: duplicate of last emitted command."
            logger.debug(reason)
            return ValidationResult(
                is_valid=False,
                command=cmd_enum,
                reason=reason,
                was_deduplicated=True,
            )

        # 6. Command passed all validation rules
        self._last_emitted_command = cmd_enum
        reason = f"Command '{cmd_enum.name}' validated successfully."
        logger.info(reason)
        return ValidationResult(is_valid=True, command=cmd_enum, reason=reason)
