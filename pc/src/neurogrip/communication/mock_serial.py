"""
neurogrip/communication/mock_serial.py
──────────────────────────────────────
Mock Serial Interface / Transport Implementation.
Records transmitted frames in memory for testing and demo execution without physical ESP32 hardware.
"""
from __future__ import annotations

import logging
from typing import Optional

from neurogrip.commands.definitions import CommandLike, NeuroGripCommand, command_name, resolve_command
from neurogrip.communication.base import SerialInterface, SerialState
from neurogrip.communication.protocol import ProtocolEncoder

logger = logging.getLogger(__name__)


class MockSerialInterface(SerialInterface):
    """
    Mock serial interface that records transmitted frames in memory.
    Useful for testing, headless pipeline execution, and hardware-free demos.
    """

    def __init__(self) -> None:
        self._state: SerialState = SerialState.DISCONNECTED
        self.sent_frames: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.sent_commands: list[NeuroGripCommand] = []

    @property
    def is_connected(self) -> bool:
        return self._state == SerialState.MOCK

    @property
    def state(self) -> SerialState:
        return self._state

    def connect(self) -> bool:
        self._state = SerialState.MOCK
        logger.info("MockSerialInterface connected.")
        return True

    def disconnect(self) -> None:
        self._state = SerialState.DISCONNECTED
        logger.info("MockSerialInterface disconnected.")

    def clear(self) -> None:
        """Clear recorded transmission history."""
        self.sent_frames.clear()
        self.sent_bytes.clear()
        self.sent_commands.clear()

    def send_command(self, command: CommandLike) -> bool:
        if not self.is_connected:
            logger.warning("MockSerialInterface: Cannot send command while disconnected.")
            return False

        frame_str = ProtocolEncoder.encode_command_string(command_name(command))
        if frame_str is None:
            logger.warning("MockSerialInterface: Rejected invalid/NO_COMMAND '%s'.", command)
            return False

        frame_bytes = frame_str.encode("ascii")
        cmd_enum = resolve_command(command)

        self.sent_frames.append(frame_str)
        self.sent_bytes.append(frame_bytes)
        if cmd_enum:
            self.sent_commands.append(cmd_enum)

        logger.info("[MOCK SERIAL TX] %s", frame_str.strip())
        return True


# Alias for explicit Transport nomenclature requirement
MockSerialTransport = MockSerialInterface
