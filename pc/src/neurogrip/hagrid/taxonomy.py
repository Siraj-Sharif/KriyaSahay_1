"""NeuroGrip Taxonomy and HaGRID Class Mapping Module.

This module defines:
1. The exact 34 official HaGRID class names and zero-indexed ordering.
2. The locked NeuroGrip gesture taxonomy (hardware command vocabulary).
3. The strict mapping rules converting raw HaGRID predictions to NeuroGrip commands.

Vocabulary note
---------------
There are two distinct command vocabularies in NeuroGrip and they MUST NOT be merged:

* ``NEUROGRIP_TAXONOMY`` / ``CanonicalCommand`` (this module)
    The locked runtime/hardware gesture vocabulary. These names are what the
    recognition models emit and what is transmitted to the ESP32 over serial.

* ``neurogrip.commands.definitions.NeuroGripCommand``
    The dataset/training label vocabulary (legacy V1 names such as ``INDEX``,
    ``CLOSE``, ``GRAB``). Dataset QA, collectors and training scripts validate
    their CSV labels against it. It is intentionally a *different* vocabulary.

``CanonicalCommand`` mirrors ``NEUROGRIP_TAXONOMY`` exactly and is derived from it
programmatically, so the two can never drift apart.
"""

from enum import Enum
from typing import Dict, List, Optional

# Official HaGRID 34-class taxonomy ordering (verified directly from constants.py in hukenovs/hagrid)
HAGRID_CLASSES: List[str] = [
    "grabbing",       # 0
    "grip",           # 1
    "holy",           # 2
    "point",          # 3
    "call",           # 4
    "three3",         # 5
    "timeout",        # 6
    "xsign",          # 7
    "hand_heart",     # 8
    "hand_heart2",    # 9
    "little_finger",  # 10
    "middle_finger",  # 11
    "take_picture",   # 12
    "dislike",        # 13
    "fist",           # 14
    "four",           # 15
    "like",           # 16
    "mute",           # 17
    "ok",             # 18
    "one",            # 19 (EXPLICITLY REMOVED -> NO_COMMAND)
    "palm",           # 20
    "peace",          # 21
    "peace_inverted", # 22
    "rock",           # 23
    "stop",           # 24
    "stop_inverted",  # 25
    "three",          # 26
    "three2",         # 27
    "two_up",         # 28
    "two_up_inverted",# 29
    "three_gun",      # 30
    "thumb_index",    # 31
    "thumb_index2",   # 32
    "no_gesture",     # 33
]

HAGRID_INDEX_TO_CLASS: Dict[int, str] = {i: cls for i, cls in enumerate(HAGRID_CLASSES)}
HAGRID_CLASS_TO_INDEX: Dict[str, int] = {cls: i for i, cls in enumerate(HAGRID_CLASSES)}

# Constants
NO_COMMAND: str = "NO_COMMAND"

# Locked NeuroGrip Gesture Taxonomy (the hardware command vocabulary)
NEUROGRIP_TAXONOMY: List[str] = [
    "CALL",
    "CLOSED_FIST",
    "FOUR_FINGERS",
    "GRIP",
    "THUMBS_UP",
    "PINKY",
    "MIDDLE_FINGER",
    "OK",
    "INDEX_FINGER",
    "INDEX_PINKY",
    "STOP",
    "TWO_FINGERS",
    "THREE_FINGERS",
]

# Canonical (hardware) command vocabulary, derived from NEUROGRIP_TAXONOMY so the
# enum and the list above can never disagree. Member value == member name.
CanonicalCommand = Enum(
    "CanonicalCommand",
    {name: name for name in NEUROGRIP_TAXONOMY},
    type=str,
    module=__name__,
    qualname="CanonicalCommand",
)
CanonicalCommand.__doc__ = (
    "Runtime/hardware gesture vocabulary derived from NEUROGRIP_TAXONOMY.\n\n"
    "Used by the command validator to resolve canonical gesture names (e.g. CALL, OK)\n"
    "that are absent from the legacy dataset vocabulary ``NeuroGripCommand``."
)


def canonical_from_string(name: Optional[str]) -> Optional[CanonicalCommand]:
    """Resolve a canonical taxonomy command by name (case-insensitive).

    Returns ``None`` for ``None``/``NO_COMMAND``/``UNKNOWN``/``INVALID``/empty or any
    string that is not part of the locked taxonomy.
    """
    if not name or not isinstance(name, str):
        return None
    cleaned = name.strip().upper()
    if cleaned in (NO_COMMAND, "UNKNOWN", "INVALID", "REST", ""):
        return None
    try:
        return CanonicalCommand(cleaned)
    except ValueError:
        return None

# Explicit mapping dictionary from HaGRID class name -> NeuroGrip command string
_HAGRID_TO_NEUROGRIP_MAP: Dict[str, str] = {
    "call": "CALL",
    "fist": "CLOSED_FIST",
    "four": "FOUR_FINGERS",
    "grabbing": NO_COMMAND,
    "grip": "GRIP",
    "like": "THUMBS_UP",
    "little_finger": "PINKY",
    "middle_finger": "MIDDLE_FINGER",
    "ok": "OK",
    "point": "INDEX_FINGER",
    "rock": "INDEX_PINKY",
    "stop": "STOP",
    "stop_inverted": "STOP",
    "two_up": "TWO_FINGERS",
    "two_up_inverted": "TWO_FINGERS",
    "three": "THREE_FINGERS",
    "three2": "THREE_FINGERS",
    "three3": "THREE_FINGERS",
    "three_gun": "THREE_FINGERS",
    # HaGRID 'one' is explicitly REMOVED and mapped to NO_COMMAND
    "one": NO_COMMAND,
}


def map_hagrid_to_neurogrip(class_name: str) -> str:
    """Map an official HaGRID class name to the locked NeuroGrip taxonomy command.

    Args:
        class_name: The raw prediction class string from HaGRID.

    Returns:
        The corresponding NeuroGrip taxonomy command string, or 'NO_COMMAND' if invalid/unmapped.
    """
    if not isinstance(class_name, str):
        return NO_COMMAND
    
    clean_name = class_name.strip().lower()
    return _HAGRID_TO_NEUROGRIP_MAP.get(clean_name, NO_COMMAND)


def is_valid_neurogrip_command(command: str) -> bool:
    """Check if a command string is part of the locked 13-gesture NeuroGrip taxonomy."""
    return command in NEUROGRIP_TAXONOMY
