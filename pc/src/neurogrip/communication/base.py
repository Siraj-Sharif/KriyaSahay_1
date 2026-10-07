"""
neurogrip/communication/base.py
───────────────────────────────
Abstract base class and protocol constants for serial communication.
Decouples serial interface implementation from the rest of the NeuroGrip application.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
import logging
from typing import Optional

from neurogrip.commands.definitions import CommandLike, NeuroGripCommand, command_name

logger = logging.getLogger(__name__)


class SerialState(Enum):
    """Serial interface connection states."""
    DISCONNECTED = auto()
    CONNECTED = auto()
    MOCK = auto()
    ERROR = auto()


class SerialInterface(ABC):
    """
    Abstract interface for serial communication with the microcontroller (ESP32).
    LOCKED Protocol: NG1|<COMMAND>\n
    Max Frame Length: 32 bytes
    """

    PROTOCOL_PREFIX: str = "NG1"
    FIELD_SEPARATOR: str = "|"
    FRAME_TERMINATOR: str = "\n"
    MAX_FRAME_LENGTH: int = 32

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the serial interface is connected and ready to transmit."""
        pass

    @property
    @abstractmethod
    def state(self) -> SerialState:
        """Current connection state of the serial interface."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Open the serial connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the serial connection cleanly."""
        pass

    @abstractmethod
    def send_command(self, command: CommandLike) -> bool:
        """
        Validate and transmit a command over serial.

        Parameters
        ----------
        command : str | NeuroGripCommand
            Command to transmit (must be one of the 14 locked commands).

        Returns
        -------
        bool
            True if transmitted successfully, False otherwise.
        """
        pass

    def send(self, command: CommandLike) -> bool:
        """Alias for send_command."""
        return self.send_command(command)

    def close(self) -> None:
        """Alias for disconnect."""
        self.disconnect()

    @classmethod
    def format_frame(cls, command: CommandLike) -> Optional[str]:
        """
        Validate a command candidate and format it into a newline-terminated ASCII protocol frame.

        Parameters
        ----------
        command : str | NeuroGripCommand
            Command candidate.

        Returns
        -------
        Optional[str]
            Formatted protocol frame string (e.g. "NG1|INDEX_FINGER\n"), or None if command is invalid.
        """
        from neurogrip.communication.protocol import ProtocolEncoder

        cmd_str = command_name(command)
        return ProtocolEncoder.encode_command_string(cmd_str if cmd_str else None)

