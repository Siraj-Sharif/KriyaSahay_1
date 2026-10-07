"""
tests/test_hagrid_adapter.py
───────────────────────────────
Isolated Unit Tests for Phase 10K HaGRID / HaGRIDv2 Deep Research Adapter & Taxonomy Mapper.
"""
from neurogrip.recognition.hagrid_adapter import (
    HAGRID_MODEL_REGISTRY,
    HaGRIDGestureClass,
    HaGRIDV2TaxonomyMapper,
)


def test_hagrid_model_registry_entries():
    """Verify registry specifications for HaGRID pretrained backbones."""
    assert "mobilenet_v3_small" in HAGRID_MODEL_REGISTRY
    assert "resnet18" in HAGRID_MODEL_REGISTRY

    mb_spec = HAGRID_MODEL_REGISTRY["mobilenet_v3_small"]
    assert mb_spec.architecture == "MobileNetV3-Small"
    assert mb_spec.embedding_dim == 1024
    assert mb_spec.model_size_mb < 15.0

    res_spec = HAGRID_MODEL_REGISTRY["resnet18"]
    assert res_spec.architecture == "ResNet-18"
    assert res_spec.embedding_dim == 512


def test_hagrid_taxonomy_mapper_direct():
    """Test direct mapping from HaGRID categories to NeuroGrip commands."""
    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("one")
    assert cmd == "INDEX"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("peace")
    assert cmd == "TWO_FINGER"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("three")
    assert cmd == "THREE_FINGER"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("four")
    assert cmd == "FOUR_FINGERS"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("fist")
    assert cmd == "CLOSE"
    assert corr == "DIRECT"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("no_gesture")
    assert cmd == "REST"
    assert corr == "DIRECT"


def test_hagrid_taxonomy_mapper_approximate():
    """Test approximate mapping from HaGRID categories."""
    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("like")
    assert cmd == "THUMB_ONLY"
    assert corr == "APPROXIMATE"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("rock")
    assert cmd == "INDEX_PINKY"
    assert corr == "APPROXIMATE"

    cmd, corr = HaGRIDV2TaxonomyMapper.map_category("ok")
    assert cmd == "GRAB"
    assert corr == "APPROXIMATE"


def test_hagrid_taxonomy_mapper_unsupported():
    """Test unsupported gestures (MIDDLE, RING, PINKY)."""
    breakdown = HaGRIDV2TaxonomyMapper.get_full_taxonomy_breakdown()

    assert breakdown["MIDDLE"]["status"] == "UNSUPPORTED (FAIL)"
    assert breakdown["MIDDLE"]["correspondence"] == "NONE"

    assert breakdown["RING"]["status"] == "UNSUPPORTED (FAIL)"
    assert breakdown["RING"]["correspondence"] == "NONE"

    assert breakdown["PINKY"]["status"] == "UNSUPPORTED (FAIL)"
    assert breakdown["PINKY"]["correspondence"] == "NONE"
