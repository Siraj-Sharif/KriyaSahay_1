"""
tests/test_phase2_1_gaps.py
───────────────────────────
Phase 2.1 regression tests — the functional gaps closed after the Phase 2 review.

One section per Phase 2.1 requirement:

  §3  ExtraTrees / HaGRID label → canonical command mapping cannot land on the wrong gesture
      (bundle class order, no off-by-one index, no legacy-name drift). The locked taxonomy is
      asserted, never modified.
  §4  STOP arm/disarm round-trips through the persistent control channel; the reported arming
      state is the backend's own (`is_armed`), never a UI-local copy.
  §5  STOP stays available on the authoritative path — one frame per STOP event, never
      de-duplicated by the validator, transmitted even while the software arm flag is off.
  §6  Camera device failures are reported truthfully; a disabled camera genuinely stops
      consuming/producing frames (no fake preview).
  §7  A disconnected serial link refuses commands, transmits nothing and reports the real error.
  §8  Voice transcripts dispatch exactly one command — or none for unsupported/ambiguous input.
  §9  CV, manual and voice commands all end in the same single serial owner.
  §10/§11  The state payload carries the required fields and real failures stay visible.

The pipeline/`FakeElectronServer` fixtures are shared with `test_control_channel.py` so both
suites exercise the identical bridge contract.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pytest
from unittest.mock import MagicMock

from tests.test_control_channel import FakeElectronServer, StubRecognizer, _one_hand

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.bridge.control import Action
from neurogrip.bridge.tcp_bridge import TCPIPBridge
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.config.settings import AppConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.features.finger_state import FingerState, FingerStateVector
from neurogrip.hagrid.taxonomy import (
    HAGRID_CLASSES,
    HAGRID_CLASS_TO_INDEX,
    HAGRID_INDEX_TO_CLASS,
    NEUROGRIP_TAXONOMY,
    NO_COMMAND,
    map_hagrid_to_neurogrip,
)
from neurogrip.hand_tracking.landmarks import HandLandmarks, Handedness, NormalizedLandmark
from neurogrip.recognition.hybrid import (
    EXTRATREES_TO_NEUROGRIP_MAP,
    HybridRecognizer,
    map_extratrees_to_neurogrip,
)
from neurogrip.recognition.ml_recognizer import MLRecognizer
from neurogrip.recognition.rule_based import RuleBasedRecognizer

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIM = FeatureExtractor.FEATURE_DIM


# ─────────────────────────────────────────────────────────
# Fixtures (mock camera + mock serial + fake Electron)
# ─────────────────────────────────────────────────────────

@pytest.fixture()
def electron() -> Iterator[FakeElectronServer]:
    server = FakeElectronServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture()
def pipeline(electron: FakeElectronServer) -> Iterator[NeuroGripPipeline]:
    config = AppConfig.default()
    config.camera.backend = "mock"
    config.serial.enabled = True
    config.serial.mock = True
    config.visualization.enabled = False
    config.stabilization.min_frames_required = 1
    config.stabilization.vote_threshold = 0.0
    config.stop.arm_timeout_ms = 0  # the tests drive arming explicitly

    bridge = TCPIPBridge(host="127.0.0.1", port=electron.port)
    pipe = NeuroGripPipeline(
        config=config,
        serial=MockSerialInterface(),
        recognizer=StubRecognizer("NO_COMMAND"),
        bridge=bridge,
    )
    assert pipe.initialize() is True

    pipe.detector = MagicMock()
    pipe.detector.is_initialized = True
    pipe.detector.detect.return_value = _one_hand()
    try:
        yield pipe
    finally:
        pipe.shutdown()


def _pump(pipe: NeuroGripPipeline, frames: int = 1) -> None:
    """Run `frames` pipeline iterations, leaving room for the bridge's 10 Hz state throttle."""
    for _ in range(frames):
        pipe.process_frame()
        time.sleep(0.12)


def _hand() -> HandLandmarks:
    return HandLandmarks(
        landmarks=[NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)],
        handedness=Handedness.RIGHT,
        score=0.99,
    )


# ─────────────────────────────────────────────────────────
# §3  ExtraTrees / HaGRID mapping review
# ─────────────────────────────────────────────────────────

# Deliberately NOT alphabetical: it proves the prediction indexes the bundle's own
# `classes` list instead of a hardcoded/sorted label order.
_BUNDLE_CLASSES = ["CLOSE", "INDEX", "REST", "TWO_FINGER", "GRAB", "STOP"]

_EXPECTED_CANONICAL = {
    "CLOSE": "CLOSED_FIST",
    "INDEX": "INDEX_FINGER",
    "REST": NO_COMMAND,
    "TWO_FINGER": "TWO_FINGERS",
    "GRAB": "GRIP",
    "STOP": "STOP",
}


class _ArgmaxClassifier:
    """`predict_proba` puts all mass on one column, pinning index → label deterministically."""

    def __init__(self, index: int, n_classes: int) -> None:
        self.index = int(index)
        self.classes_ = np.array([f"c{i}" for i in range(n_classes)])

    def predict_proba(self, X):  # noqa: N803 - sklearn signature
        probs = np.zeros((len(X), len(self.classes_)), dtype=float)
        probs[:, self.index] = 1.0
        return probs


def _write_bundle(tmp_path: Path, classes: list[str], classifier: Any) -> Path:
    path = tmp_path / "bundle.pkl"
    joblib.dump(
        {
            "classifier": classifier,
            "preprocessor": None,
            "classes": list(classes),
            "feature_version": FeatureExtractor.FEATURE_VERSION,
            "feature_dim": FEATURE_DIM,
        },
        path,
    )
    return path


def _loaded_recognizer(tmp_path: Path, classes: list[str], index: int) -> MLRecognizer:
    """A real joblib bundle on disk + a classifier whose argmax is a known column."""
    classifier = _ArgmaxClassifier(index, len(classes))
    bundle_path = _write_bundle(tmp_path, classes, classifier)
    config = AppConfig.default()
    # Point the constructor at a missing file on purpose: the explicit load below is what the
    # test exercises, and this keeps the "bundle not found" path covered as a side effect.
    config.recognition.model_path = str(tmp_path / "absent_on_purpose.pkl")
    recognizer = MLRecognizer(config=config)
    assert recognizer.load_model(bundle_path) is True
    return recognizer


@pytest.mark.parametrize("index", list(range(len(_BUNDLE_CLASSES))))
def test_extratrees_prediction_uses_the_bundle_class_order(tmp_path: Path, index: int):
    """
    The ML layer must resolve `argmax` through the bundle's own `classes` list.

    A hardcoded / re-sorted / off-by-one label table would return a different gesture here;
    this pins the numeric → string step for every class in the bundle.
    """
    recognizer = _loaded_recognizer(tmp_path, _BUNDLE_CLASSES, index)

    result = recognizer.predict(
        np.zeros(FEATURE_DIM, dtype=np.float32), is_stop_armed=True
    )

    assert result.label == _BUNDLE_CLASSES[index], (
        f"argmax index {index} must resolve to {_BUNDLE_CLASSES[index]!r}, got {result.label!r}"
    )
    assert result.confidence == pytest.approx(1.0)
    assert set(result.all_scores) == set(_BUNDLE_CLASSES)
    # The probability vector is read in the bundle's own class order.
    assert result.all_scores[_BUNDLE_CLASSES[index]] == pytest.approx(1.0)


@pytest.mark.parametrize("index", list(range(len(_BUNDLE_CLASSES))))
def test_extratrees_label_then_canonical_mapping_is_exact(tmp_path: Path, index: int):
    """`map_extratrees_to_neurogrip` must send each bundle label to its intended gesture."""
    recognizer = _loaded_recognizer(tmp_path, _BUNDLE_CLASSES, index)
    result = recognizer.predict(np.zeros(FEATURE_DIM, dtype=np.float32), is_stop_armed=True)

    canonical = map_extratrees_to_neurogrip(result.label)

    assert canonical == _EXPECTED_CANONICAL[result.label]


def test_extratrees_mapping_is_total_over_the_frozen_vocabularies():
    """
    Every legacy dataset label maps to either a locked taxonomy command or NO_COMMAND.

    Nothing may map outside the taxonomy (a typo would otherwise reach the encoder), and the
    two historically ambiguous names must stay pinned: REST is idle-only, RING has no
    canonical hardware gesture.
    """
    for command in NeuroGripCommand:
        mapped = map_extratrees_to_neurogrip(command.value)
        assert mapped == NO_COMMAND or mapped in NEUROGRIP_TAXONOMY, (
            f"{command.value} mapped to {mapped!r}, which is not part of the locked taxonomy"
        )

    assert map_extratrees_to_neurogrip("REST") == NO_COMMAND
    assert map_extratrees_to_neurogrip("RING") == NO_COMMAND
    assert map_extratrees_to_neurogrip("GRAB") == "GRIP"
    assert map_extratrees_to_neurogrip("CLOSE") == "CLOSED_FIST"
    assert map_extratrees_to_neurogrip("THUMB_ONLY") == "THUMBS_UP"
    assert map_extratrees_to_neurogrip("STOP") == "STOP"
    # Unknown / malformed input can never invent a command.
    assert map_extratrees_to_neurogrip("GRABBING") == NO_COMMAND
    assert map_extratrees_to_neurogrip("") == NO_COMMAND
    assert map_extratrees_to_neurogrip(None) == NO_COMMAND  # type: ignore[arg-type]
    # The map and the pure function cannot drift apart.
    assert set(EXTRATREES_TO_NEUROGRIP_MAP) >= {c.value for c in NeuroGripCommand}


def test_hagrid_index_order_and_mapping_are_frozen():
    """
    The 34-class HaGRID table is index-addressed: verify index ↔ class, and that every mapped
    value is a member of the locked 13-command taxonomy (no drift, no `GRABBING`).
    """
    assert len(HAGRID_CLASSES) == 34
    for index, name in enumerate(HAGRID_CLASSES):
        assert HAGRID_INDEX_TO_CLASS[index] == name
        assert HAGRID_CLASS_TO_INDEX[name] == index
        mapped = map_hagrid_to_neurogrip(name)
        assert mapped == NO_COMMAND or mapped in NEUROGRIP_TAXONOMY, f"{name} -> {mapped!r}"

    assert map_hagrid_to_neurogrip("grabbing") == NO_COMMAND
    assert map_hagrid_to_neurogrip("one") == NO_COMMAND
    assert map_hagrid_to_neurogrip("stop") == "STOP"
    assert map_hagrid_to_neurogrip("call") == "CALL"
    assert map_hagrid_to_neurogrip("ok") == "OK"
    assert map_hagrid_to_neurogrip("not_a_hagrid_class") == NO_COMMAND

    # The frozen taxonomy itself: 13 members, no GRABBING.
    assert len(NEUROGRIP_TAXONOMY) == 13
    assert "GRABBING" not in NEUROGRIP_TAXONOMY
    assert set(EXTRATREES_TO_NEUROGRIP_MAP.values()) <= set(NEUROGRIP_TAXONOMY) | {NO_COMMAND}


class _StubHaGRID:
    """Canned HaGRID classifier; records whether the CPU CNN was consulted at all."""

    def __init__(self, raw_class: str = "call") -> None:
        self.is_loaded = True
        self.calls = 0
        self.raw_class = raw_class

    def predict(self, frame, is_bgr: bool = True, top_k: int = 5):  # noqa: ARG002
        self.calls += 1
        return {
            "raw_class": self.raw_class,
            "mapped_command": map_hagrid_to_neurogrip(self.raw_class),
            "confidence": 0.92,
            "top_k": [
                {"raw_class": self.raw_class, "mapped_command": map_hagrid_to_neurogrip(self.raw_class), "probability": 0.92}
            ],
        }


def _hybrid(tmp_path: Path, bundle_index: int, hagrid: _StubHaGRID) -> HybridRecognizer:
    recognizer = _loaded_recognizer(tmp_path, _BUNDLE_CLASSES, bundle_index)
    config = AppConfig.default()
    config.stop.arm_timeout_ms = 0
    return HybridRecognizer(
        config=config,
        hagrid_classifier=hagrid,
        ml_recognizer=recognizer,
        rule_recognizer=RuleBasedRecognizer(config=config),
    )


def _gate(hybrid: HybridRecognizer, **fingers: FingerState) -> str | None:
    vector = FingerStateVector(
        thumb=fingers.get("thumb", FingerState.CLOSED),
        index=fingers.get("index", FingerState.CLOSED),
        middle=fingers.get("middle", FingerState.CLOSED),
        ring=fingers.get("ring", FingerState.CLOSED),
        pinky=fingers.get("pinky", FingerState.CLOSED),
    )
    hybrid.rule_recognizer.finger_detector.detect = lambda _hand: vector  # type: ignore[method-assign]
    return hybrid._check_landmark_routing_gate(_hand())


def test_landmark_routing_gate_selects_the_specialised_model_only(tmp_path: Path):
    """Posture → model routing is deterministic, and CALL can never be mis-routed as PINKY."""
    hybrid = _hybrid(tmp_path, 1, _StubHaGRID())  # ExtraTrees bundle column 1 == "INDEX"

    assert _gate(hybrid, index=FingerState.OPEN) == "INDEX_FINGER"
    assert _gate(hybrid, index=FingerState.OPEN, middle=FingerState.OPEN) == "TWO_FINGERS"
    # Thumb CLOSED + pinky only -> the ExtraTrees pinky posture.
    assert _gate(hybrid, pinky=FingerState.OPEN) == "PINKY"
    # Thumb OPEN (the CALL posture) must NOT be routed to the ExtraTrees pinky path.
    assert _gate(hybrid, thumb=FingerState.OPEN, pinky=FingerState.OPEN) is None


def test_gated_posture_routes_to_extratrees_and_skips_hagrid(tmp_path: Path):
    """An index-only posture must be answered by the ExtraTrees column, never by the CNN."""
    hagrid = _StubHaGRID()
    hybrid = _hybrid(tmp_path, 1, hagrid)
    _gate(hybrid, index=FingerState.OPEN)

    result = hybrid.predict(
        np.zeros(FEATURE_DIM, dtype=np.float32),
        hand_landmarks=_hand(),
        is_stop_armed=True,
        raw_frame=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert result.model_used == "EXTRA_TREES"
    assert result.label == "INDEX_FINGER"
    assert hagrid.calls == 0, "gated postures must not run the CPU CNN"
    assert "INDEX" in (result.reason or "")


def test_ungated_posture_falls_through_to_hagrid(tmp_path: Path):
    """A thumb-open pinky posture is the CALL posture; it must be answered by HaGRID as CALL."""
    hagrid = _StubHaGRID("call")
    hybrid = _hybrid(tmp_path, 1, hagrid)
    _gate(hybrid, thumb=FingerState.OPEN, pinky=FingerState.OPEN)

    result = hybrid.predict(
        np.zeros(FEATURE_DIM, dtype=np.float32),
        hand_landmarks=_hand(),
        is_stop_armed=True,
        raw_frame=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert hagrid.calls == 1
    assert result.model_used == "HAGRID"
    assert result.label == "CALL"
    assert result.raw_prediction == "call"


# ─────────────────────────────────────────────────────────
# §4  STOP arm / disarm control channel
# ─────────────────────────────────────────────────────────

def test_stop_arm_disarm_round_trips_through_the_control_channel(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """`stop.set_armed` changes backend state, and the state stream reports the backend's own flag."""
    armed = electron.request(Action.STOP_SET_ARMED, {"armed": True})
    assert armed["ok"] is True
    assert armed["data"]["armed"] is True
    assert pipeline.is_stop_armed is True

    _pump(pipeline)
    state = electron.wait_for_state(lambda s: s.get("is_armed") is True)
    assert state["safety_status"] == "STOP ARMED"

    disarmed = electron.request(Action.STOP_SET_ARMED, {"armed": False})
    assert disarmed["ok"] is True and disarmed["data"]["armed"] is False
    assert pipeline.is_stop_armed is False

    _pump(pipeline)
    state = electron.wait_for_state(lambda s: s.get("safety_status") == "STOP DISARMED")
    assert state["is_armed"] is False


def test_stop_arm_rejects_non_boolean(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.STOP_SET_ARMED, {"armed": "yes"})
    assert result["ok"] is False
    assert "boolean" in (result["error"] or "")
    assert pipeline.is_stop_armed is False, "a rejected request must not change backend state"


# ─────────────────────────────────────────────────────────
# §5  STOP stays available on the authoritative path
# ─────────────────────────────────────────────────────────

def test_disarmed_stop_is_still_transmitted_exactly_once(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """
    Frozen STOP semantics: STOP is a normal, always-available command.

    The software arm flag is reported (and auto-expires) but never gates STOP — pinning this
    means a future change cannot silently make the safety gesture unavailable.
    """
    pipeline.set_stop_armed(False)
    pipeline.recognizer.label = "STOP"
    pipeline.detector.detect.return_value = _one_hand()

    _pump(pipeline, 3)

    frames = list(pipeline.serial.sent_bytes)
    assert frames.count(b"NG1|STOP\n") == 1, "STOP must reach the wire exactly once while disarmed"

    state = electron.wait_for_state(lambda s: s.get("final_command") == "STOP")
    assert state["is_armed"] is False  # the arming state is reported honestly
    assert state["last_tx"] == "NG1|STOP"  # while STOP still reached the hardware


def test_manual_stop_is_never_deduplicated(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """An operator must be able to send STOP repeatedly (validator de-duplication exempts it)."""
    first = electron.request(Action.COMMAND_SEND, {"command": "STOP", "source": "manual"})
    second = electron.request(Action.COMMAND_SEND, {"command": "STOP", "source": "manual"})

    assert first["ok"] is True and second["ok"] is True
    assert list(pipeline.serial.sent_bytes) == [b"NG1|STOP\n", b"NG1|STOP\n"]
    assert pipeline.serial_info["frames_sent"] == 2


def test_cv_transmission_waits_for_the_pipeline_stabilizer(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """
    Only the pipeline's stabilized CV result may reach serial.

    This is the regression guard for the renderer-side duplicate/bypass fixed in Phase 2.1:
    a detection that is still stabilizing must not appear on the wire.
    """
    pipeline.stabilizer.min_frames_required = 3  # default production value behaviour
    pipeline.recognizer.label = "CALL"
    pipeline.detector.detect.return_value = _one_hand()

    pipeline.process_frame()
    pipeline.process_frame()
    assert pipeline.serial.sent_bytes == [], "an unstable detection must not transmit"

    pipeline.process_frame()
    assert pipeline.serial.sent_bytes == [b"NG1|CALL\n"]
    assert pipeline.serial_info["frames_sent"] == 1


# ─────────────────────────────────────────────────────────
# §6  Camera readiness
# ─────────────────────────────────────────────────────────

class _NeverOpenCapture:
    """cv2.VideoCapture stand-in that can never be opened."""

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return False

    def release(self) -> None:
        pass


def test_unavailable_camera_is_reported_and_never_faked(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer, monkeypatch: pytest.MonkeyPatch
):
    """
    Real backend path (OpenCV) with a device that cannot open.

    The pipeline must (a) refuse, (b) mark capture disabled instead of pretending it is live,
    and (c) publish the real error so the UI can show it.
    """
    monkeypatch.setattr(
        "neurogrip.camera.opencv_camera.cv2.VideoCapture", lambda *a, **k: _NeverOpenCapture()
    )
    pipeline.camera = OpenCVCamera(config=pipeline.config.camera)
    pipeline.config.camera.backend = "opencv"

    result = electron.request(Action.CAMERA_SET_ENABLED, {"enabled": True})

    assert result["ok"] is False
    assert "camera" in (result["error"] or "").lower()
    assert result["data"]["camera"]["enabled"] is False
    assert result["data"]["camera"]["error"] == "camera unavailable"
    assert pipeline.camera_info["opened"] is False

    _pump(pipeline)
    state = electron.wait_for_state(lambda s: s.get("pipeline_state") == "CAMERA_OFF")
    assert state["camera_info"]["enabled"] is False
    assert state["camera_info"]["error"] == "camera unavailable"
    assert "camera unavailable" in state["errors"]


def test_disabled_camera_stops_consuming_and_publishing_frames(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """Stopping capture must genuinely stop frame reads — no synthetic frames, no preview."""
    pipeline.tcp_bridge.send_frame = MagicMock()
    pipeline.camera.read_frame = MagicMock()

    result = electron.request(Action.CAMERA_SET_ENABLED, {"enabled": False})

    assert result["ok"] is True
    assert result["data"]["camera"]["enabled"] is False
    assert pipeline.camera.is_opened is False

    container, rendered, state = pipeline.process_frame()

    assert state.pipeline_state == "CAMERA_OFF"
    assert state.fps == 0.0
    assert container.frame is None
    assert rendered is None
    assert pipeline.camera.read_frame.call_count == 0, "a stopped camera must not be read"
    assert pipeline.tcp_bridge.send_frame.call_count == 0, "no fake preview may be published"


# ─────────────────────────────────────────────────────────
# §7  Serial readiness
# ─────────────────────────────────────────────────────────

def test_command_after_disconnect_is_rejected_and_transmits_nothing(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """Disconnect → command must fail visibly and put nothing on the link."""
    first = electron.request(Action.COMMAND_SEND, {"command": "GRIP", "source": "manual"})
    assert first["ok"] is True and first["data"]["sent"] is True
    before = list(pipeline.serial.sent_bytes)

    assert electron.request(Action.SERIAL_DISCONNECT)["ok"] is True
    assert pipeline.serial.is_connected is False

    rejected = electron.request(Action.COMMAND_SEND, {"command": "GRIP", "source": "manual"})

    assert rejected["ok"] is False
    assert rejected["data"]["sent"] is False
    assert "serial" in (rejected["error"] or "").lower()
    assert list(pipeline.serial.sent_bytes) == before, "nothing may reach a closed link"
    assert pipeline.serial_info["connected"] is False
    assert "serial link unavailable" in pipeline.runtime_info()["errors"]


def test_serial_swap_keeps_exactly_one_owner(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """Connecting a new port must release the previous interface, never hold two."""
    original = pipeline.serial

    result = electron.request(
        Action.SERIAL_CONNECT, {"port": "COM_TEST", "baud": 57600, "mode": "mock"}
    )

    assert result["ok"] is True
    assert pipeline.serial is not original, "the pipeline builds its own interface"
    assert original.is_connected is False, "the previous owner must be released"
    assert pipeline.serial_info == {
        "mode": "mock",
        "port": "COM_TEST",
        "baud": 57600,
        "connected": True,
        "state": "MOCK",
        "frames_sent": 0,
        "last_tx": "NONE",
    }
    assert result["data"]["serial"]["port"] == "COM_TEST"

    # A fresh link must not display a frame the previous port transmitted.
    assert pipeline.serial_info["last_tx"] == "NONE"
    assert pipeline.serial_info["frames_sent"] == 0

    electron.request(Action.SERIAL_DISCONNECT)
    assert pipeline.serial_info["connected"] is False


# ─────────────────────────────────────────────────────────
# §8  Voice readiness
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("call", "CALL"),
        ("okay", "OK"),
        ("open", "STOP"),
        ("stop", "STOP"),
        ("close", "CLOSED_FIST"),
        ("point", "INDEX_FINGER"),
    ],
)
def test_voice_phrase_transmits_exactly_one_frame(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer, phrase: str, expected: str
):
    """A recognised utterance produces exactly one authoritative dispatch."""
    before = len(pipeline.serial.sent_bytes)

    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": phrase})

    assert result["ok"] is True
    assert len(pipeline.serial.sent_bytes) == before + 1, "exactly one frame per utterance"
    assert pipeline.serial.sent_bytes[-1] == f"NG1|{expected}\n".encode("ascii")
    assert result["data"]["sent"] is True
    assert result["data"]["command"] == expected
    assert result["data"]["frame"] == f"NG1|{expected}"
    assert result["data"]["voice"]["command"] == expected


def test_voice_unsupported_phrase_transmits_nothing(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """`pinch` is not in the voice vocabulary: it must never invent a command."""
    before = list(pipeline.serial.sent_bytes)

    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "pinch"})

    assert result["ok"] is True
    assert result["data"]["intent"] == "unknown"
    assert result["data"].get("sent") is not True
    assert list(pipeline.serial.sent_bytes) == before


def test_voice_command_is_refused_while_serial_is_down(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """A voice command with no link must fail visibly (never a silent success)."""
    electron.request(Action.SERIAL_DISCONNECT)
    before = list(pipeline.serial.sent_bytes)

    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "call"})

    assert result["ok"] is False
    assert result["data"]["sent"] is False
    assert "serial" in (result["error"] or "").lower()
    assert list(pipeline.serial.sent_bytes) == before
    assert "serial link unavailable" in pipeline.runtime_info()["errors"]


def test_repeated_voice_command_is_an_explicit_retry(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """Saying the same command twice must send twice (explicit sources bypass detector dedup)."""
    first = electron.request(Action.VOICE_TRANSCRIPT, {"text": "close"})
    second = electron.request(Action.VOICE_TRANSCRIPT, {"text": "close"})

    assert first["ok"] is True and second["ok"] is True
    assert list(pipeline.serial.sent_bytes) == [b"NG1|CLOSED_FIST\n", b"NG1|CLOSED_FIST\n"]
    assert pipeline.runtime_info()["manual_tx_count"] == 2


# ─────────────────────────────────────────────────────────
# §9  One command source path
# ─────────────────────────────────────────────────────────

def test_cv_manual_and_voice_reach_the_same_serial_owner(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """All three sources must end in the pipeline's single serial interface, in order."""
    owner = pipeline.serial
    pipeline.recognizer.label = "OK"
    pipeline.detector.detect.return_value = _one_hand()

    _pump(pipeline, 2)
    manual = electron.request(Action.COMMAND_SEND, {"command": "CALL", "source": "manual"})
    voice = electron.request(Action.VOICE_TRANSCRIPT, {"text": "thumbs up"})

    assert manual["ok"] is True and voice["ok"] is True
    assert pipeline.serial is owner, "no source may swap or open its own serial interface"
    assert list(pipeline.serial.sent_bytes) == [b"NG1|OK\n", b"NG1|CALL\n", b"NG1|THUMBS_UP\n"]
    assert pipeline.serial_info["frames_sent"] == 3
    assert pipeline.runtime_info()["manual_tx_count"] == 2


def test_no_duplicate_hardware_path_exists_in_the_sources():
    """
    Static guard for §9: the renderer/Electron layers must not own a hardware path, and
    pyserial may only be opened by the pipeline's single serial owner.
    """
    electron_sources = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "electron").glob("*.cjs"))
    )
    # NOTE: the string "python -c" legitimately appears in a comment in `main.cjs`
    # documenting the removed one-shot helpers, so only real process/port APIs are checked.
    for forbidden in ("child_process", "spawn(", "execFile(", "execSync", "serialport"):
        assert forbidden not in electron_sources, (
            f"{forbidden!r} in electron/*.cjs would be a second hardware path"
        )

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "pc" / "src" / "neurogrip").rglob("*.py")
        if "serial.Serial(" in path.read_text(encoding="utf-8") and path.name != "serial_port.py"
    ]
    assert offenders == [], f"pyserial opened outside the single owner: {offenders}"


# ─────────────────────────────────────────────────────────
# §10 / §11  State completeness and failure visibility
# ─────────────────────────────────────────────────────────

def test_state_payload_carries_every_required_field(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """Every field the Phase 2.1 state audit requires must arrive in the UI payload."""
    _pump(pipeline)
    state = electron.wait_for_state(lambda s: "serial_info" in s and "camera_info" in s)

    required = {
        "command",
        "confidence",
        "fps",
        "latency_ms",
        "hand_count",
        "pipeline_state",
        "stabilizer_state",
        "is_armed",
        "is_tx_permitted",
        "final_command",
        "stop_rule_triggered",
        "stop_rule_metrics",
        "hagrid_raw",
        "hagrid_conf",
        "taxonomy_mapped",
        "last_tx",
        "camera_info",
        "serial_info",
        "voice_info",
        "errors",
        "landmarks_count",
        "manual_tx_count",
    }
    missing = sorted(required - set(state))
    assert missing == [], f"state payload is missing {missing}"

    assert state["camera_info"]["index"] == pipeline.config.camera.index
    assert state["serial_info"]["mode"] == "mock"
    assert state["voice_info"]["state"] in {"idle", "listening", "processing", "speaking"}
    assert isinstance(state["errors"], list)


def test_backend_failure_is_visible_in_the_state_stream(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """A refused transmission must surface as a real error string, not a silent success."""
    electron.request(Action.SERIAL_DISCONNECT)
    rejected = electron.request(Action.COMMAND_SEND, {"command": "CALL", "source": "manual"})
    assert rejected["ok"] is False

    _pump(pipeline)
    state = electron.wait_for_state(lambda s: bool(s.get("errors")))

    assert "serial link unavailable" in state["errors"]
    assert state["is_tx_permitted"] is False
    assert state["final_command"] in {"NO_COMMAND", "CALL"}
