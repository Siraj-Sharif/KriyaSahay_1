"""
neurogrip/communication/protocol.py
───────────────────────────────────
Dedicated Protocol Encoder Component.
Encodes canonical NeuroGrip commands into newline-terminated NG1 wire frames.

LOCKED WIRE PROTOCOL:
  NG1|<CMD>\n

Examples:
  NG1|CALL\n
  NG1|CLOSED_FIST\n
  NG1|FOUR_FINGERS\n
  NG1|GRIP\n
  NG1|GRIP\n
  NG1|THUMBS_UP\n
  NG1|PINKY\n
  NG1|MIDDLE_FINGER\n
  NG1|OK\n
  NG1|INDEX_FINGER\n
  NG1|INDEX_PINKY\n
  NG1|STOP\n
  NG1|TWO_FINGERS\n
  NG1|THREE_FINGERS\n

CRITICAL INVARIANTS:
  1. NO_COMMAND produces NO frame (returns None).
  2. UNKNOWN, INVALID, empty, or unmapped inputs produce NO frame (returns None).
  3. Wire frames are strictly deterministic UTF-8/ASCII bytes.
  4. Frame length MUST NOT exceed 32 bytes.
  5. This module MUST NOT import pyserial.
"""
from __future__ import annotations

import logging
from typing import Optional

from neurogrip.hagrid.taxonomy import NEUROGRIP_TAXONOMY, NO_COMMAND

logger = logging.getLogger(__name__)

# Protocol Constants
PROTOCOL_PREFIX: str = "NG1"
FIELD_SEPARATOR: str = "|"
FRAME_TERMINATOR: str = "\n"
MAX_FRAME_LENGTH: int = 32

# Locked 13 Canonical Commands
CANONICAL_COMMANDS: set[str] = set(NEUROGRIP_TAXONOMY)

# Wire Protocol Name Mapping for Canonical Commands & Internal Aliases
CANONICAL_WIRE_MAP: dict[str, str] = {
    "CALL": "CALL",
    "CLOSED_FIST": "CLOSED_FIST",
    "CLOSE": "CLOSED_FIST",
    "FIST": "CLOSED_FIST",
    "FOUR_FINGERS": "FOUR_FINGERS",
    "GRAB": "GRIP",
    "GRIP": "GRIP",
    "THUMBS_UP": "THUMBS_UP",
    "THUMB_ONLY": "THUMBS_UP",
    "LIKE": "THUMBS_UP",
    "PINKY": "PINKY",
    "LITTLE_FINGER": "PINKY",
    "MIDDLE_FINGER": "MIDDLE_FINGER",
    "MIDDLE": "MIDDLE_FINGER",
    "OK": "OK",
    "INDEX_FINGER": "INDEX_FINGER",
    "INDEX": "INDEX_FINGER",
    "POINT": "INDEX_FINGER",
    "INDEX_PINKY": "INDEX_PINKY",
    "ROCK": "INDEX_PINKY",
    "STOP": "STOP",
    "TWO_FINGERS": "TWO_FINGERS",
    "TWO_FINGER": "TWO_FINGERS",
    "THREE_FINGERS": "THREE_FINGERS",
    "THREE_FINGER": "THREE_FINGERS",
    "NEUTRAL": "NEUTRAL",
}


class ProtocolEncoder:
    """
    Hardware-independent protocol encoder for NeuroGrip commands.
    Does NOT import or depend on pyserial.
    """

    PROTOCOL_PREFIX = PROTOCOL_PREFIX
    FIELD_SEPARATOR = FIELD_SEPARATOR
    FRAME_TERMINATOR = FRAME_TERMINATOR
    MAX_FRAME_LENGTH = MAX_FRAME_LENGTH

    @classmethod
    def is_canonical_command(cls, command: str | None) -> bool:
        """
        Check whether command string corresponds to one of the 14 locked canonical commands.
        """
        if not command or not isinstance(command, str):
            return False
        cleaned = command.strip().upper()
        if cleaned in (NO_COMMAND, "UNKNOWN", "INVALID", "REST", ""):
            return False
        return cleaned in CANONICAL_WIRE_MAP

    @classmethod
    def encode_command_string(cls, command: str | None) -> Optional[str]:
        """
        Validate and format a canonical command into a newline-terminated ASCII protocol frame string.

        Parameters
        ----------
        command : str | None
            Command candidate string or Enum object.

        Returns
        -------
        Optional[str]
            Formatted frame string (e.g. "NG1|INDEX_FINGER\n"), or None if invalid/NO_COMMAND.
        """
        if command is None:
            return None

        # Extract enum string value if an Enum was passed
        if hasattr(command, "value"):
            raw_str = str(command.value)
        else:
            raw_str = str(command)

        if not isinstance(raw_str, str):
            return None

        cleaned = raw_str.strip().upper()

        if cleaned in (NO_COMMAND, "UNKNOWN", "INVALID", "REST", ""):
            return None

        wire_name = CANONICAL_WIRE_MAP.get(cleaned)
        if wire_name is None:
            logger.warning("ProtocolEncoder: Rejected non-canonical command '%s'", raw_str)
            return None

        frame_str = f"{cls.PROTOCOL_PREFIX}{cls.FIELD_SEPARATOR}{wire_name}{cls.FRAME_TERMINATOR}"
        frame_bytes = frame_str.encode("ascii")

        if len(frame_bytes) > cls.MAX_FRAME_LENGTH:
            logger.error(
                "ProtocolEncoder: Formatted frame exceeds max length (%d > %d bytes): '%s'",
                len(frame_bytes),
                cls.MAX_FRAME_LENGTH,
                frame_str.strip(),
            )
            return None

        return frame_str

    @classmethod
    def encode_command(cls, command: str | None) -> Optional[bytes]:
        """
        Validate and encode a canonical command into newline-terminated UTF-8/ASCII bytes.

        Parameters
        ----------
        command : str | None
            Command candidate string or Enum object.

        Returns
        -------
        Optional[bytes]
            Formatted frame bytes (e.g. b"NG1|INDEX_FINGER\n"), or None if invalid/NO_COMMAND.
        """
        frame_str = cls.encode_command_string(command)
        if frame_str is None:
            return None
        return frame_str.encode("ascii")


def encode_protocol_frame(command: str | None) -> Optional[bytes]:
    """Convenience module-level function to encode a canonical command into bytes."""
    return ProtocolEncoder.encode_command(command)
