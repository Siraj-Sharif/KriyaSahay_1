"""
neurogrip/communication/transport.py
──────────────────────────────────────
Transport Boundary Abstraction Layer.
Decouples finalized NeuroGrip command emission from downstream hardware/network transports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional, Union

from neurogrip.commands.definitions import (
    CommandLike,
    NeuroGripCommand,
    command_name,
    resolve_command,
)
from neurogrip.communication.protocol import ProtocolEncoder
from neurogrip.hagrid.taxonomy import CanonicalCommand

logger = logging.getLogger(__name__)


class CommandTransport(ABC):
    """
    Abstract transport boundary for emitting finalized NeuroGrip commands.
    """

    PROTOCOL_PREFIX: str = ProtocolEncoder.PROTOCOL_PREFIX
    FIELD_SEPARATOR: str = ProtocolEncoder.FIELD_SEPARATOR
    FRAME_TERMINATOR: str = ProtocolEncoder.FRAME_TERMINATOR

    @abstractmethod
    def send_command(self, command: CommandLike) -> bool:
        """
        Emit a valid command candidate to downstream target.

        Returns
        -------
        bool
            True if emitted successfully, False otherwise.
        """
        pass

    @abstractmethod
    def send_idle(self) -> None:
        """
        Notify transport that the pipeline is in an IDLE / NO-COMMAND state.
        (Emits no active robotic command frame).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanly release transport resources."""
        pass

    @classmethod
    def format_protocol_frame(cls, command: CommandLike) -> Optional[str]:
        """
        Format a command candidate into standard protocol string: NG1|<COMMAND>
        """
        cmd_str = command_name(command)
        frame = ProtocolEncoder.encode_command_string(cmd_str if cmd_str else None)
        if frame is None:
            return None
        return frame.rstrip("\r\n")


class DisplayConsoleTransport(CommandTransport):
    """
    Stage 1 Production Local Display & Console Transport Handler.
    Formats valid active commands into protocol frames (e.g. NG1|INDEX_FINGER)
    and exposes them for local GUI display and terminal logging.
    Does NOT transmit over TCP network or serial ports.
    """

    def __init__(self) -> None:
        self._latest_command: Optional[Union[NeuroGripCommand, CanonicalCommand]] = None
        self._latest_frame: Optional[str] = None
        self._emission_count: int = 0

    @property
    def latest_command(self) -> Optional[Union[NeuroGripCommand, CanonicalCommand]]:
        return self._latest_command

    @property
    def latest_frame(self) -> Optional[str]:
        return self._latest_frame

    @property
    def emission_count(self) -> int:
        return self._emission_count

    def send_command(self, command: CommandLike) -> bool:
        cmd_str = command_name(command)
        frame_str = self.format_protocol_frame(cmd_str)
        if frame_str is None:
            self.send_idle()
            return False

        # Same resolution path as the CommandValidator, so the recorded command can
        # never diverge from what was actually validated and transmitted.
        cmd_enum = resolve_command(cmd_str)

        self._latest_command = cmd_enum
        self._latest_frame = frame_str
        self._emission_count += 1

        logger.info("[STAGE 1 COMMAND BOUNDARY] Protocol Frame: '%s' (Tx Count: %d)", frame_str, self._emission_count)
        return True

    def send_idle(self) -> None:
        self._latest_command = None
        self._latest_frame = None

    def close(self) -> None:
        self.send_idle()
        logger.info("DisplayConsoleTransport closed.")
