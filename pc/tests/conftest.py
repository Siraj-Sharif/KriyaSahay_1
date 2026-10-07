"""
tests/conftest.py
─────────────────
Shared pytest fixtures and synthetic hand landmark generators.
Provides analytically constructed 21-point HandLandmarks for unit testing without webcam hardware.
"""
import pytest

from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark


def make_hand_landmarks(
    tip_offsets: dict[str, tuple[float, float]],
    handedness: Handedness = Handedness.RIGHT,
    wrist_x: float = 0.5,
    wrist_y: float = 0.8,
    scale: float = 1.0,
) -> HandLandmarks:
    """
    Construct a synthetic 21-landmark HandLandmarks object with specified finger extensions.
    For LEFT handedness, the generated landmark x coordinates are mirrored relative to wrist_x.

    Base geometry (wrist at (wrist_x, wrist_y)):
    0: Wrist (wrist_x, wrist_y)
    1-4: Thumb (1=CMC, 2=MCP, 3=IP, 4=TIP)
    5-8: Index (5=MCP, 6=PIP, 7=DIP, 8=TIP)
    9-12: Middle (9=MCP, 10=PIP, 11=DIP, 12=TIP)
    13-16: Ring (13=MCP, 14=PIP, 15=DIP, 16=TIP)
    17-20: Pinky (17=MCP, 18=PIP, 19=DIP, 20=TIP)
    """
    # Fixed MCP base positions relative to wrist for RIGHT hand (in scale=1.0 space)
    # Right hand: thumb is on the left (-x side of wrist when palm faces camera)
    mcp_bases = {
        "thumb": (-0.08, -0.10),
        "index": (-0.05, -0.20),
        "middle": (0.00, -0.22),
        "ring": (0.05, -0.20),
        "pinky": (0.09, -0.18),
    }

    lms: list[NormalizedLandmark] = [NormalizedLandmark(x=wrist_x, y=wrist_y, z=0.0)]

    # Thumb (1..4)
    tx, ty = tip_offsets.get("thumb", (0.15, -0.15))
    lms.append(NormalizedLandmark(x=wrist_x + mcp_bases["thumb"][0] * scale * 0.5, y=wrist_y + mcp_bases["thumb"][1] * scale * 0.5, z=0.0))
    lms.append(NormalizedLandmark(x=wrist_x + mcp_bases["thumb"][0] * scale, y=wrist_y + mcp_bases["thumb"][1] * scale, z=0.0))
    lms.append(NormalizedLandmark(x=wrist_x + (mcp_bases["thumb"][0] + tx * 0.5) * scale, y=wrist_y + (mcp_bases["thumb"][1] + ty * 0.5) * scale, z=0.0))
    lms.append(NormalizedLandmark(x=wrist_x + (mcp_bases["thumb"][0] + tx) * scale, y=wrist_y + (mcp_bases["thumb"][1] + ty) * scale, z=-0.02))

    # Non-thumb fingers helper
    fingers = [("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)]
    for fname, start_idx in fingers:
        bx, by = mcp_bases[fname]
        tx, ty = tip_offsets.get(fname, (0.0, -0.15))

        # MCP
        lms.append(NormalizedLandmark(x=wrist_x + bx * scale, y=wrist_y + by * scale, z=0.0))
        # PIP
        lms.append(NormalizedLandmark(x=wrist_x + (bx + tx * 0.33) * scale, y=wrist_y + (by + ty * 0.33) * scale, z=0.0))
        # DIP
        lms.append(NormalizedLandmark(x=wrist_x + (bx + tx * 0.66) * scale, y=wrist_y + (by + ty * 0.66) * scale, z=0.0))
        # TIP
        lms.append(NormalizedLandmark(x=wrist_x + (bx + tx) * scale, y=wrist_y + (by + ty) * scale, z=0.01))

    # If handedness is LEFT, geometrically mirror all x coordinates around wrist_x
    if handedness == Handedness.LEFT:
        lms = [
            NormalizedLandmark(x=2.0 * wrist_x - lm.x, y=lm.y, z=lm.z)
            for lm in lms
        ]

    assert len(lms) == 21
    return HandLandmarks(landmarks=lms, handedness=handedness, score=0.99)


@pytest.fixture
def open_hand_landmarks():
    """All 5 fingers fully extended."""
    return make_hand_landmarks(
        tip_offsets={
            "thumb": (0.20, -0.15),
            "index": (0.0, -0.25),
            "middle": (0.0, -0.27),
            "ring": (0.0, -0.25),
            "pinky": (0.0, -0.22),
        },
        handedness=Handedness.RIGHT,
    )


@pytest.fixture
def closed_hand_landmarks():
    """All 5 fingers fully closed (fist)."""
    return make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, 0.05),
            "middle": (0.0, 0.05),
            "ring": (0.0, 0.05),
            "pinky": (0.0, 0.05),
        },
        handedness=Handedness.RIGHT,
    )


@pytest.fixture
def index_only_landmarks():
    """Index finger extended, thumb/middle/ring/pinky closed."""
    return make_hand_landmarks(
        tip_offsets={
            "thumb": (0.02, 0.05),
            "index": (0.0, -0.25),
            "middle": (0.0, 0.05),
            "ring": (0.0, 0.05),
            "pinky": (0.0, 0.05),
        },
        handedness=Handedness.RIGHT,
    )


@pytest.fixture
def thumb_only_landmarks():
    """Thumb extended laterally to the right, remaining fingers closed."""
    return make_hand_landmarks(
        tip_offsets={
            "thumb": (0.25, -0.05),
            "index": (0.0, 0.05),
            "middle": (0.0, 0.05),
            "ring": (0.0, 0.05),
            "pinky": (0.0, 0.05),
        },
        handedness=Handedness.RIGHT,
    )
