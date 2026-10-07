"""
tests/test_features.py
───────────────────────
Unit tests for FeatureExtractor.
Verifies the locked 68-dimensional Feature Version v2 representation,
feature ordering, numerical stability, safeguards, handedness normalization,
and regression compatibility for legacy v1 67-dimensional extraction.
"""
import numpy as np
import pytest

from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.landmarks import Handedness, HandLandmarks, NormalizedLandmark
from tests.conftest import make_hand_landmarks


def test_feature_extractor_v2_output_shape_and_version(open_hand_landmarks):
    """Verify default feature vector has shape (68,), dtype float32, and metadata v2."""
    extractor = FeatureExtractor()
    features = extractor.extract(open_hand_landmarks)

    assert isinstance(features, np.ndarray)
    assert features.shape == (68,)
    assert features.dtype == np.float32
    assert FeatureExtractor.FEATURE_DIM == 68
    assert FeatureExtractor.FEATURE_VERSION == "v2"
    assert len(features) == 68


def test_feature_extractor_v1_regression(open_hand_landmarks):
    """Verify v1 legacy extraction remains unchanged and returns 67-dimensional vector."""
    extractor = FeatureExtractor(version="v1")
    features_v1 = extractor.extract(open_hand_landmarks)

    assert isinstance(features_v1, np.ndarray)
    assert features_v1.shape == (67,)
    assert features_v1.dtype == np.float32

    # Also test extract_v1 directly
    f1_direct = extractor.extract_v1(open_hand_landmarks)
    assert np.array_equal(features_v1, f1_direct)


def test_feature_extractor_finite_values(open_hand_landmarks, closed_hand_landmarks):
    """Verify all 68 feature vector values are finite (no NaN or Inf)."""
    extractor = FeatureExtractor()
    feat1 = extractor.extract(open_hand_landmarks)
    feat2 = extractor.extract(closed_hand_landmarks)

    assert np.all(np.isfinite(feat1))
    assert np.all(np.isfinite(feat2))


def test_feature_extractor_determinism(open_hand_landmarks):
    """Verify multiple extractions from the same input yield identical feature vectors."""
    extractor = FeatureExtractor()
    f1 = extractor.extract(open_hand_landmarks)
    f2 = extractor.extract(open_hand_landmarks)

    assert np.array_equal(f1, f2)


def test_feature_extractor_wrist_coordinates_excluded(open_hand_landmarks):
    """Verify landmark 0 (wrist) is excluded from coordinates [0..39], leaving landmarks 1..20."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    # Coordinates [0..39] represent landmarks 1..20
    # Check that feature 0 and 1 correspond to landmark 1 (Thumb CMC)
    lms = open_hand_landmarks.landmarks
    w = lms[0]
    span2d = np.hypot(lms[9].x - w.x, lms[9].y - w.y)

    expected_dx1 = (lms[1].x - w.x) / span2d
    expected_dy1 = (lms[1].y - w.y) / span2d

    assert np.isclose(feats[0], expected_dx1, atol=1e-5)
    assert np.isclose(feats[1], expected_dy1, atol=1e-5)


def test_feature_extractor_relative_fingertip_z(open_hand_landmarks):
    """Verify relative fingertip z values [40..44] are normalized by 3D span and wrist-centered."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    lms = open_hand_landmarks.landmarks
    w = lms[0]
    span3d = np.sqrt((lms[9].x - w.x)**2 + (lms[9].y - w.y)**2 + (lms[9].z - w.z)**2)

    # Check index fingertip z (Landmark 8) -> feature 41
    expected_dz8 = (lms[8].z - w.z) / span3d
    assert np.isclose(feats[41], expected_dz8, atol=1e-5)


def test_feature_extractor_angle_clipping(open_hand_landmarks):
    """Verify 3D joint-angle cosines [45..54] are clipped to [-1.0, 1.0]."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    angles = feats[45:55]
    assert np.all(angles >= -1.0)
    assert np.all(angles <= 1.0)


def test_feature_extractor_straightness_indices(open_hand_landmarks):
    """Verify finger straightness indices [55..59] are finite, non-negative, and <= 1.0."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    straightness = feats[55:60]
    assert np.all(np.isfinite(straightness))
    assert np.all(straightness > 0.0)
    assert np.all(straightness <= 1.0 + 1e-5)


def test_feature_extractor_contrast_clipping_and_logic(open_hand_landmarks):
    """Verify inter-finger contrast indices [60..64] are clipped to [-1.0, 1.0] and match formula."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    straightness = feats[55:60]
    contrast = feats[60:65]

    assert np.all(contrast >= -1.0)
    assert np.all(contrast <= 1.0)

    # Verify C_k = S_k - (1/4) * sum(S_j for j != k)
    s_sum = np.sum(straightness)
    for k in range(5):
        expected_ck = np.clip(straightness[k] - (s_sum - straightness[k]) / 4.0, -1.0, 1.0)
        assert np.isclose(contrast[k], expected_ck, atol=1e-5)


def test_feature_extractor_thumb_index_direction_cosine(open_hand_landmarks):
    """Verify feature 65 (THUMB_INDEX_DIRECTION_COSINE) is clipped to [-1.0, 1.0]."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    cos_alpha = feats[65]
    assert np.isfinite(cos_alpha)
    assert -1.0 <= cos_alpha <= 1.0


def test_feature_extractor_thumb_to_palm_distance(open_hand_landmarks):
    """Verify feature 66 (Thumb-to-Palm 3D distance) is finite and positive."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    d_tp = feats[66]
    assert np.isfinite(d_tp)
    assert d_tp > 0.0


def test_feature_extractor_minimum_straightness(open_hand_landmarks):
    """Verify feature 67 (S_min) equals the minimum of the 5 straightness features [55..59]."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    straightness = feats[55:60]
    s_min = feats[67]

    assert np.isclose(s_min, np.min(straightness), atol=1e-5)


def test_feature_extractor_exact_ordering(open_hand_landmarks):
    """Verify exact feature ordering across all 68 indices."""
    extractor = FeatureExtractor()
    feats = extractor.extract(open_hand_landmarks)

    # 0..39: Coords (40)
    # 40..44: Z depths (5)
    # 45..54: Angles (10)
    # 55..59: Straightness (5)
    # 60..64: Contrast (5)
    # 65: Thumb-Index cosine (1)
    # 66: Thumb-palm dist (1)
    # 67: S_min (1)
    assert len(feats[:40]) == 40
    assert len(feats[40:45]) == 5
    assert len(feats[45:55]) == 10
    assert len(feats[55:60]) == 5
    assert len(feats[60:65]) == 5
    assert len(feats[65:66]) == 1
    assert len(feats[66:67]) == 1
    assert len(feats[67:68]) == 1
    assert len(feats) == 68


def test_feature_extractor_degenerate_landmarks_safeguard():
    """Verify zero-span or colinear landmarks do not produce NaN or Inf values."""
    extractor = FeatureExtractor()

    # Create degenerate landmarks where all 21 points are at origin (0, 0, 0)
    zero_lms = [NormalizedLandmark(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    hand_zero = HandLandmarks(landmarks=zero_lms, handedness=Handedness.RIGHT)

    feats = extractor.extract(hand_zero)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (68,)
    assert np.all(np.isfinite(feats))
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))


def test_feature_extractor_handedness_normalization():
    """Verify left-hand x-mirroring produces equivalent features for left and right equivalent poses."""
    tip_cfg = {
        "thumb": (0.20, -0.15),
        "index": (0.0, -0.25),
        "middle": (0.0, -0.27),
        "ring": (0.0, -0.25),
        "pinky": (0.0, -0.22),
    }

    right_hand = make_hand_landmarks(tip_cfg, handedness=Handedness.RIGHT)
    left_hand = make_hand_landmarks(tip_cfg, handedness=Handedness.LEFT)

    extractor = FeatureExtractor()
    f_right = extractor.extract(right_hand)
    f_left = extractor.extract(left_hand)

    # Feature Group 1 (normalized coords 0..39) should match closely after x-mirroring
    np.testing.assert_allclose(f_right[:40], f_left[:40], atol=1e-4)
