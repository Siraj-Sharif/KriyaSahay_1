"""
neurogrip/commands/definitions.py
──────────────────────────────────
Command vocabulary definitions and the shared command-resolution helpers.

Two vocabularies exist in NeuroGrip and both are intentionally preserved:

1. ``NeuroGripCommand`` (this module)
   The legacy V1 / dataset + training label vocabulary (INDEX, CLOSE, GRAB, REST…).
   Dataset QA, dataset collectors and training scripts validate their CSV labels
   against this enum, so its membership is a data-contract and must not change.

2. ``neurogrip.hagrid.taxonomy.CanonicalCommand``
   The locked runtime/hardware gesture vocabulary (CALL, OK, GRIP, CLOSED_FIST…)
   actually emitted by the recognizers and transmitted to the ESP32.

``resolve_command()`` resolves a candidate against the legacy vocabulary first
(preserving all historical aliases and behaviour) and then against the canonical
hardware vocabulary. This is the single resolution path shared by the command
validator and by the transport boundary, so a command can never be accepted by one
layer and rejected by another.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Union

from neurogrip.hagrid.taxonomy import CanonicalCommand, canonical_from_string


class CommandCategory(str, Enum):
    FINGER_CONFIG = "finger_config"
    ACTION = "action"
    SYSTEM_STATE = "system_state"
    SAFETY = "safety"


@dataclass(frozen=True)
class CommandMetadata:
    """Metadata describing a single NeuroGrip command."""
    canonical_name: str
    category: CommandCategory
    is_safety_critical: bool
    description: str


@unique
class NeuroGripCommand(str, Enum):
    """
    Legacy V1 / dataset + training label vocabulary (13 members, locked).

    NOTE: this is the *dataset* vocabulary, not the runtime hardware vocabulary.
    The runtime hardware vocabulary is ``neurogrip.hagrid.taxonomy.CanonicalCommand``.
    Do not add runtime gesture names (e.g. CALL, OK) here — dataset QA and the
    dataset preparation scripts depend on this enum containing exactly these 13 labels.
    """
    INDEX = "INDEX"
    MIDDLE = "MIDDLE"
    RING = "RING"
    PINKY = "PINKY"
    THUMB_ONLY = "THUMB_ONLY"
    TWO_FINGER = "TWO_FINGER"
    THREE_FINGER = "THREE_FINGER"
    INDEX_PINKY = "INDEX_PINKY"
    FOUR_FINGERS = "FOUR_FINGERS"
    CLOSE = "CLOSE"
    GRAB = "GRAB"
    REST = "REST"
    STOP = "STOP"

    @property
    def metadata(self) -> CommandMetadata:
        return COMMAND_METADATA_MAP[self]

    @classmethod
    def from_string(cls, name: str) -> Optional["NeuroGripCommand"]:
        """Look up a command by string (case-insensitive) with alias support."""
        if not name or not isinstance(name, str):
            return None
        cleaned = name.strip().upper()
        if cleaned in ("NO_COMMAND", "UNKNOWN", "INVALID", "NONE", ""):
            return None

        alias_map = {
            "INDEX_FINGER": "INDEX",
            "POINT": "INDEX",
            "MIDDLE_FINGER": "MIDDLE",
            "TWO_FINGERS": "TWO_FINGER",
            "THREE_FINGERS": "THREE_FINGER",
            "CLOSED_FIST": "CLOSE",
            "FIST": "CLOSE",
            "GRIP": "GRAB",
            "THUMBS_UP": "THUMB_ONLY",
            "LIKE": "THUMB_ONLY",
            "ROCK": "INDEX_PINKY",
            "LITTLE_FINGER": "PINKY",
        }
        resolved = alias_map.get(cleaned, cleaned)
        try:
            return cls(resolved)
        except ValueError:
            return None


COMMAND_METADATA_MAP: dict[NeuroGripCommand, CommandMetadata] = {
    NeuroGripCommand.INDEX: CommandMetadata(
        canonical_name="INDEX",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Index finger extended, remaining fingers closed.",
    ),
    NeuroGripCommand.MIDDLE: CommandMetadata(
        canonical_name="MIDDLE",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Middle finger extended, remaining fingers closed.",
    ),
    NeuroGripCommand.RING: CommandMetadata(
        canonical_name="RING",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Ring finger extended, remaining fingers closed.",
    ),
    NeuroGripCommand.PINKY: CommandMetadata(
        canonical_name="PINKY",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Pinky extended, remaining fingers closed.",
    ),
    NeuroGripCommand.THUMB_ONLY: CommandMetadata(
        canonical_name="THUMB_ONLY",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Thumb extended sideways while remaining fingers closed.",
    ),
    NeuroGripCommand.TWO_FINGER: CommandMetadata(
        canonical_name="TWO_FINGER",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Index + middle extended, remaining fingers closed.",
    ),
    NeuroGripCommand.THREE_FINGER: CommandMetadata(
        canonical_name="THREE_FINGER",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Index + middle + ring extended, remaining fingers closed.",
    ),
    NeuroGripCommand.INDEX_PINKY: CommandMetadata(
        canonical_name="INDEX_PINKY",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Index + pinky extended, remaining fingers closed.",
    ),
    NeuroGripCommand.FOUR_FINGERS: CommandMetadata(
        canonical_name="FOUR_FINGERS",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="Index + middle + ring + pinky extended, thumb closed.",
    ),
    NeuroGripCommand.CLOSE: CommandMetadata(
        canonical_name="CLOSE",
        category=CommandCategory.FINGER_CONFIG,
        is_safety_critical=False,
        description="All five fingers fully closed into a fist.",
    ),
    NeuroGripCommand.GRAB: CommandMetadata(
        canonical_name="GRAB",
        category=CommandCategory.ACTION,
        is_safety_critical=False,
        description="Functional grasp action; static partial-curl claw configuration.",
    ),
    NeuroGripCommand.REST: CommandMetadata(
        canonical_name="REST",
        category=CommandCategory.SYSTEM_STATE,
        is_safety_critical=False,
        description="Relaxed/idle hand state.",
    ),
    NeuroGripCommand.STOP: CommandMetadata(
        canonical_name="STOP",
        category=CommandCategory.SAFETY,
        is_safety_critical=True,
        description="Dedicated safety stop command; compact upright flat-hand posture.",
    ),
}


# ─────────────────────────────────────────────────────────
# Shared command resolution helpers
# ─────────────────────────────────────────────────────────

#: Any value accepted as a command candidate by the resolution helpers.
CommandLike = Union[NeuroGripCommand, CanonicalCommand, str]


def command_name(command: Optional[CommandLike]) -> str:
    """
    Extract the canonical command *string* from an enum member or a plain string.

    Enum members of both vocabularies expose their wire name through ``.value``.
    Using ``str(member)`` would yield "ClassName.MEMBER" on Python 3.11, so every
    name extraction in the codebase must go through this helper.
    """
    if command is None:
        return ""
    if isinstance(command, Enum):
        return str(command.value)
    return str(command)


def resolve_command(command: Optional[CommandLike]) -> Optional[Union[NeuroGripCommand, CanonicalCommand]]:
    """
    Resolve a command candidate to a vocabulary member.

    Resolution order:
      1. Already-resolved members of either vocabulary are returned unchanged.
      2. The legacy dataset vocabulary (``NeuroGripCommand.from_string``), which
         preserves every historical alias (INDEX_FINGER->INDEX, GRIP->GRAB, …).
      3. The canonical hardware vocabulary (``CanonicalCommand``), which supplies
         the runtime gesture names missing from the legacy enum (CALL, OK).

    Returns ``None`` when the candidate is empty, of the wrong type, or is not part
    of either vocabulary (e.g. NO_COMMAND, UNKNOWN, INVALID, REST, arbitrary text).
    """
    if command is None:
        return None
    if isinstance(command, (NeuroGripCommand, CanonicalCommand)):
        return command
    if not isinstance(command, str):
        return None

    legacy = NeuroGripCommand.from_string(command)
    if legacy is not None:
        return legacy
    return canonical_from_string(command)
