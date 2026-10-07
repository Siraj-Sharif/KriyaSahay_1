"""
tests/test_visualization.py
────────────────────────────
Unit tests for Visualization Overlay and Renderer.
Does NOT require a physical camera, GUI window, or live display.
"""
import numpy as np
import pytest

from neurogrip.config.settings import VisualizationConfig
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.visualization.base import VisualizationState
from neurogrip.visualization.renderer import OverlayRenderer
from tests.conftest import make_hand_landmarks


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """Create a blank 720x1280 BGR image array."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def renderer():
    return OverlayRenderer()


# ─────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────

def test_visualization_config_defaults():
    """Verify VisualizationConfig default settings."""
    cfg = VisualizationConfig()
    assert cfg.enabled is True
    assert cfg.window_title == "NeuroGrip CV"
    assert cfg.show_landmarks is True
    assert cfg.show_skeleton is True
    assert cfg.show_fps is True
    assert cfg.show_confidence is True


def test_renderer_accepts_valid_frame(renderer, blank_frame):
    """Verify renderer processes valid frames without throwing exceptions."""
    state = VisualizationState(command="INDEX", confidence=0.95)
    rendered = renderer.render(blank_frame, state)

    assert rendered is not None
    assert isinstance(rendered, np.ndarray)


def test_renderer_preserves_dimensions_and_dtype(renderer, blank_frame):
    """Verify renderer output matches input dimensions (720x1280x3) and dtype (uint8)."""
    state = VisualizationState(command="CLOSE")
    rendered = renderer.render(blank_frame, state)

    assert rendered.shape == blank_frame.shape
    assert rendered.dtype == blank_frame.dtype


def test_renderer_does_not_mutate_input_frame(renderer, blank_frame):
    """Verify renderer returns a new copy without modifying the original input array."""
    original_copy = blank_frame.copy()
    state = VisualizationState(command="STOP", is_stop_active=True)

    rendered = renderer.render(blank_frame, state)

    assert np.array_equal(blank_frame, original_copy)
    assert not np.array_equal(rendered, blank_frame)  # Rendered overlay alters pixels


def test_renderer_landmark_rendering_single_hand(renderer, blank_frame, open_hand_landmarks):
    """Verify renderer correctly draws skeleton & landmarks for a single detected hand."""
    state = VisualizationState(
        command="CLOSE",
        confidence=0.98,
        handedness="RIGHT",
        landmarks=[open_hand_landmarks],
    )

    rendered = renderer.render(blank_frame, state)
    assert rendered is not None
    assert np.sum(rendered) > 0  # Pixels were drawn


def test_renderer_no_hand(renderer, blank_frame):
    """Verify rendering when no hands are detected in frame."""
    state = VisualizationState(command="NO_HAND", landmarks=[])
    rendered = renderer.render(blank_frame, state)

    assert rendered is not None
    assert np.sum(rendered) > 0  # HUD text drawn


def test_renderer_ambiguous_multi_hand(renderer, blank_frame, open_hand_landmarks):
    """Verify rendering for ambiguous multi-hand state (2+ hands)."""
    state = VisualizationState(
        command="AMBIGUOUS",
        is_ambiguous=True,
        landmarks=[open_hand_landmarks, open_hand_landmarks],
    )

    rendered = renderer.render(blank_frame, state)
    assert rendered is not None


def test_renderer_unknown_rendering(renderer, blank_frame):
    """Verify rendering when gesture state is UNKNOWN."""
    state = VisualizationState(command="UNKNOWN", confidence=0.10, stabilizer_state="UNKNOWN_STATE")
    rendered = renderer.render(blank_frame, state)

    assert rendered is not None


def test_renderer_stable_command_rendering(renderer, blank_frame, open_hand_landmarks):
    """Verify rendering for a STABLE gesture command (e.g. INDEX)."""
    state = VisualizationState(
        command="INDEX",
        confidence=0.92,
        stabilizer_state="STABLE",
        pipeline_state="TRACKING",
        handedness="RIGHT",
        landmarks=[open_hand_landmarks],
        fps=30.0,
    )

    rendered = renderer.render(blank_frame, state)
    assert rendered is not None


def test_renderer_stop_rendering(renderer, blank_frame, open_hand_landmarks):
    """Verify rendering when software-armed STOP safety alert is active."""
    state = VisualizationState(
        command="STOP",
        confidence=1.0,
        stabilizer_state="STOP",
        pipeline_state="STOP_ACTIVE",
        is_stop_active=True,
        landmarks=[open_hand_landmarks],
    )

    rendered = renderer.render(blank_frame, state)
    assert rendered is not None


def test_renderer_disabled_config(blank_frame):
    """Verify renderer passes through original frame unmodified when disabled in config."""
    cfg = VisualizationConfig(enabled=False)
    renderer = OverlayRenderer(config=cfg)
    state = VisualizationState(command="INDEX")

    rendered = renderer.render(blank_frame, state)
    assert np.array_equal(rendered, blank_frame)


def test_renderer_invalid_empty_frame(renderer):
    """Verify renderer handles empty array safely without raising errors."""
    state = VisualizationState(command="INDEX")
    res = renderer.render(np.array([]), state)
    assert res.size == 0
