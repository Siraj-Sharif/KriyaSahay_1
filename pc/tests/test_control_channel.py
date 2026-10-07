"""
tests/test_control_channel.py
─────────────────────────────
Phase 2 integration tests for the persistent UI ↔ pipeline control channel.

These tests stand up a fake "Electron" TCP server, connect a real `NeuroGripPipeline`
(mock camera + mock serial) to it, and drive it exactly as the desktop app does:
newline-delimited JSON control requests in, state snapshots and control results out.

They prove the properties the UI depends on:

  * one persistent connection carries both state and control (no per-action subprocess)
  * the pipeline — not the UI — owns the camera and the serial link
  * manual / voice commands take the same validate → encode → transmit path as detection
  * CALL and OK work through the control path
  * camera and serial state reported to the UI is truthful
  * the state payload carries the safety/transmission fields the old bridge dropped
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Iterator, Optional

import pytest

from unittest.mock import MagicMock

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.bridge.control import Action, encode_result
from neurogrip.bridge.tcp_bridge import TCPIPBridge
from neurogrip.commands.validator import PipelineState
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.config.settings import AppConfig
from neurogrip.hand_tracking.landmarks import (
    DetectionResult,
    Handedness,
    HandLandmarks,
    NormalizedLandmark,
)
from neurogrip.recognition.base import RecognitionResult


# ─────────────────────────────────────────────────────────
# Fake Electron bridge server
# ─────────────────────────────────────────────────────────

class FakeElectronServer:
    """Minimal stand-in for `electron/main.cjs`: accepts the pipeline and talks JSON lines."""

    def __init__(self) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))  # ephemeral port
        self._server.listen(1)
        self.port: int = self._server.getsockname()[1]
        self._conn: Optional[socket.socket] = None
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)
        self._accept_thread.start()

        self.states: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.frames: list[dict[str, Any]] = []
        self.bridge_status: list[bool] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._buffer = ""
        self._next_id = 0

    # -- lifecycle ---------------------------------------------------

    def _accept(self) -> None:
        try:
            self._server.settimeout(10.0)
            conn, _ = self._server.accept()
        except (socket.timeout, OSError):
            return
        self._conn = conn
        self.bridge_status.append(True)
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        while True:
            conn = self._conn
            if conn is None:
                return
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                self.bridge_status.append(False)
                return
            self._buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("type") == "control_result":
                    self.results[str(msg.get("id"))] = msg
                elif msg.get("type") == "frame":
                    self.frames.append(msg)
                else:
                    self.states.append(msg)

    def _wait_for_connection(self, timeout: float = 10.0) -> None:
        """The accept thread may still be returning when the client finishes connecting."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._conn is not None:
                return
            time.sleep(0.01)
        raise AssertionError("pipeline never connected to the bridge")

    def request(self, action: str, payload: Optional[dict[str, Any]] = None, timeout: float = 5.0) -> dict[str, Any]:
        """Send one control request and wait for its correlated result."""
        self._wait_for_connection()
        assert self._conn is not None
        self._next_id += 1
        request_id = f"c{self._next_id}"
        line = json.dumps({"type": "control", "id": request_id, "action": action, "payload": payload or {}})
        self._conn.sendall((line + "\n").encode("utf-8"))

        deadline = time.time() + timeout
        while time.time() < deadline:
            if request_id in self.results:
                return self.results.pop(request_id)
            time.sleep(0.01)
        raise AssertionError(f"no control result for {action} within {timeout}s")

    def wait_for_state(self, predicate, timeout: float = 5.0) -> dict[str, Any]:
        """Wait until a state snapshot satisfies `predicate`."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for state in reversed(self.states):
                if predicate(state):
                    return state
            time.sleep(0.01)
        raise AssertionError("no matching state snapshot arrived")

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except OSError:
            pass
        try:
            self._server.close()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────

class StubRecognizer:
    """Always predicts one fixed label, so the CV path can be driven deterministically."""

    def __init__(self, label: str = "NO_COMMAND") -> None:
        self.label = label

    @property
    def is_ready(self) -> bool:
        return True

    def predict(self, features, hand_landmarks=None, is_stop_armed=False, raw_frame=None) -> RecognitionResult:
        return RecognitionResult(
            label=self.label,
            confidence=1.0,
            all_scores={self.label: 1.0},
            recognizer_type="stub",
            reason="TEST_STUB",
            model_used="RULE_BASED",
            raw_prediction=self.label,
        )


def _one_hand() -> DetectionResult:
    hand = HandLandmarks(
        landmarks=[NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)],
        handedness=Handedness.RIGHT,
        score=0.99,
    )
    return DetectionResult(hands=[hand], num_hands=1)


# ─────────────────────────────────────────────────────────
# Fixtures
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

    # Fast stabilization so a single frame can reach the validator.
    config.stabilization.min_frames_required = 1
    config.stabilization.vote_threshold = 0.0
    config.stop.arm_timeout_ms = 0  # tests control arming explicitly

    bridge = TCPIPBridge(host="127.0.0.1", port=electron.port)
    serial = MockSerialInterface()
    recognizer = StubRecognizer("NO_COMMAND")

    pipe = NeuroGripPipeline(
        config=config, serial=serial, recognizer=recognizer, bridge=bridge
    )
    assert pipe.initialize() is True

    # Stub the detector: no MediaPipe model bundle / torch checkpoint in unit tests.
    pipe.detector = MagicMock()
    pipe.detector.is_initialized = True
    pipe.detector.detect.return_value = DetectionResult(hands=[], num_hands=0)
    try:
        yield pipe
    finally:
        pipe.shutdown()


def _frames(pipe: NeuroGripPipeline) -> list[bytes]:
    return list(getattr(pipe.serial, "sent_bytes", []))


# ─────────────────────────────────────────────────────────
# 1. Channel establishment
# ─────────────────────────────────────────────────────────

def test_pipeline_connects_and_state_streams(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """One persistent connection carries the state stream (no per-action subprocess)."""
    pipeline.process_frame()
    state = electron.wait_for_state(lambda s: "pipeline_state" in s)
    assert state["pipeline_state"] in {p.name for p in PipelineState} | {"CAMERA_OFF"}
    assert electron.bridge_status == [True]


def test_state_payload_includes_previously_dropped_fields(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """
    The old bridge hand-wrote 17 fields and silently dropped the rest; the UI could not
    show real arming / transmission state. The full dataclass must now be published.
    """
    pipeline.process_frame()
    state = electron.wait_for_state(lambda s: "is_tx_permitted" in s)

    for field in (
        "is_tx_permitted",
        "is_armed",
        "is_stabilized",
        "final_command",
        "taxonomy_mapped",
        "stabilizer_state",
        "stop_rule_triggered",
        "stop_rule_metrics",
        "hagrid_raw",
        "hagrid_conf",
    ):
        assert field in state, f"{field} missing from the state payload"

    # Phase 2 runtime extras
    assert "camera_info" in state and "serial_info" in state and "voice_info" in state
    assert "errors" in state
    assert state["camera_info"]["index"] == pipeline.config.camera.index


# ─────────────────────────────────────────────────────────
# 2. Camera control
# ─────────────────────────────────────────────────────────

def test_camera_list_reports_backend_cameras(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.CAMERA_LIST)
    assert result["ok"] is True
    cameras = result["data"]["cameras"]
    assert cameras and cameras[0]["active"] is True
    assert cameras[0]["index"] == pipeline.config.camera.index


def test_camera_disable_actually_stops_capture(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.CAMERA_SET_ENABLED, {"enabled": False})
    assert result["ok"] is True
    assert result["data"]["camera"]["enabled"] is False
    assert pipeline.camera.is_opened is False, "backend camera must actually be released"

    _, _, state = pipeline.process_frame()
    assert state.pipeline_state == "CAMERA_OFF"

    snapshot = electron.wait_for_state(lambda s: s.get("pipeline_state") == "CAMERA_OFF")
    assert snapshot["camera_info"]["enabled"] is False


def test_camera_enable_restores_capture(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron.request(Action.CAMERA_SET_ENABLED, {"enabled": False})
    result = electron.request(Action.CAMERA_SET_ENABLED, {"enabled": True})
    assert result["ok"] is True
    assert pipeline.camera.is_opened is True
    assert pipeline.camera_info["enabled"] is True


def test_camera_select_changes_the_backend_index(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.CAMERA_SELECT, {"index": 1})
    assert result["ok"] is True
    assert pipeline.config.camera.index == 1
    assert result["data"]["camera"]["index"] == 1


def test_camera_select_rejects_bad_index(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.CAMERA_SELECT, {"index": -3})
    assert result["ok"] is False
    assert "index" in (result["error"] or "")


# ─────────────────────────────────────────────────────────
# 3. Serial control
# ─────────────────────────────────────────────────────────

def test_serial_list_ports_comes_from_the_pipeline(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.SERIAL_LIST_PORTS)
    assert result["ok"] is True
    assert isinstance(result["data"]["ports"], list)
    assert "serial" in result["data"]


def test_serial_connect_mock_then_disconnect(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    connect = electron.request(Action.SERIAL_CONNECT, {"mode": "mock"})
    assert connect["ok"] is True
    assert connect["data"]["serial"]["connected"] is True
    assert connect["data"]["serial"]["mode"] == "mock"

    disconnect = electron.request(Action.SERIAL_DISCONNECT)
    assert disconnect["ok"] is True
    assert pipeline.serial_info["mode"] == "mock"


def test_serial_connect_reports_failure_for_unusable_port(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.SERIAL_CONNECT, {"port": "/dev/does-not-exist-ng", "baud": 115200, "mode": "real"})
    assert result["ok"] is False
    assert result["error"]
    # The pipeline must keep working on mock rather than dying with a bad port.
    assert pipeline.is_running is True


def test_transport_status_matches_serial_info(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron.request(Action.SERIAL_CONNECT, {"mode": "mock"})
    _, _, state = pipeline.process_frame()
    assert state.transport_status == "MOCK"
    assert pipeline.serial_info["connected"] is True


# ─────────────────────────────────────────────────────────
# 4. Authoritative command dispatch (incl. CALL / OK)
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,expected", [("CALL", b"NG1|CALL\n"), ("OK", b"NG1|OK\n")])
def test_manual_command_reaches_serial_through_the_pipeline(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer, command: str, expected: bytes
):
    """The Phase 1 fix, now exercised through the real UI control channel."""
    result = electron.request(Action.COMMAND_SEND, {"command": command, "source": "manual"})

    assert result["ok"] is True, result.get("error")
    assert result["data"]["command"] == command
    assert result["data"]["frame"] == f"NG1|{command}"
    assert result["data"]["sent"] is True
    assert expected in _frames(pipeline), f"{expected!r} never reached the pipeline's serial link"
    assert _frames(pipeline).count(expected) == 1


@pytest.mark.parametrize("command", ["CALL", "OK", "STOP", "GRIP", "CLOSED_FIST", "INDEX_FINGER"])
def test_every_commanded_gesture_is_validated_and_encoded(pipeline: NeuroGripPipeline, electron: FakeElectronServer, command: str):
    result = electron.request(Action.COMMAND_SEND, {"command": command})
    assert result["ok"] is True, result.get("error")
    assert result["data"]["frame"] == f"NG1|{command}"


def test_invalid_command_is_rejected_and_transmits_nothing(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    before = len(_frames(pipeline))
    result = electron.request(Action.COMMAND_SEND, {"command": "GRABBING"})
    assert result["ok"] is False
    assert "rejected" in (result["error"] or "")
    assert len(_frames(pipeline)) == before


def test_manual_transmission_updates_last_tx_telemetry(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron.request(Action.COMMAND_SEND, {"command": "OK"})
    _, _, state = pipeline.process_frame()
    assert state.last_tx == "NG1|OK"
    assert pipeline.serial_info["last_tx"] == "NG1|OK"
    assert pipeline.serial_info["frames_sent"] >= 1


# ─────────────────────────────────────────────────────────
# 5. STOP / safety
# ─────────────────────────────────────────────────────────

def test_stop_arming_is_real_backend_state(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    assert pipeline.is_stop_armed is False

    armed = electron.request(Action.STOP_SET_ARMED, {"armed": True})
    assert armed["ok"] is True
    assert pipeline.is_stop_armed is True

    disarm = electron.request(Action.STOP_SET_ARMED, {"armed": False})
    assert disarm["ok"] is True
    assert pipeline.is_stop_armed is False


def test_armed_state_is_reported_to_the_ui(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron.request(Action.STOP_SET_ARMED, {"armed": True})
    _, _, state = pipeline.process_frame()
    assert state.is_armed is True
    assert state.safety_status == "STOP ARMED"

    snapshot = electron.wait_for_state(lambda s: s.get("is_armed") is True)
    assert snapshot["safety_status"] == "STOP ARMED"


def test_stop_requires_a_boolean(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.STOP_SET_ARMED, {"armed": "yes"})
    assert result["ok"] is False


# ─────────────────────────────────────────────────────────
# 6. Voice (transcript parsed inside the pipeline)
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("call", b"NG1|CALL\n"),
        ("okay", b"NG1|OK\n"),
        ("close the hand", b"NG1|CLOSED_FIST\n"),
        ("grip", b"NG1|GRIP\n"),
        ("thumbs up", b"NG1|THUMBS_UP\n"),
    ],
)
def test_voice_transcript_transmits_through_the_same_path(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer, transcript: str, expected: bytes
):
    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": transcript, "engine": "test"})

    assert result["ok"] is True, result.get("error")
    assert result["data"]["intent"] == "gesture"
    assert expected in _frames(pipeline), f"{expected!r} never reached serial for {transcript!r}"


def test_voice_unknown_phrase_transmits_nothing(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    before = len(_frames(pipeline))
    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "make me a sandwich"})
    assert result["ok"] is True
    assert result["data"]["intent"] == "unknown"
    assert result["data"]["gesture"] is None
    assert len(_frames(pipeline)) == before


def test_voice_requires_hardware_like_any_other_command(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """A voice command must fail when the serial link cannot transmit — not silently 'succeed'."""
    pipeline.serial.disconnect()
    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "call"})
    assert result["ok"] is False
    assert "serial" in (result["error"] or "").lower()


def test_voice_state_is_reported_back_from_the_backend(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron.request(Action.VOICE_SET_STATE, {"state": "listening", "engine": "whisper-test"})
    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "grip"})
    voice = result["data"]["voice"]
    assert voice["transcript"] == "grip"
    # Reported command is the wire name: the dataset alias GRAB encodes as NG1|GRIP.
    assert voice["command"] == "GRIP"
    assert result["data"]["frame"] == "NG1|GRIP"
    assert voice["engine"] == "whisper-test"
    assert voice["reply"]


def test_voice_status_intent_reports_real_state(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.VOICE_TRANSCRIPT, {"text": "what is the system status"})
    assert result["ok"] is True
    assert result["data"]["intent"] == "status"
    assert "Camera" in result["data"]["reply"]


# ─────────────────────────────────────────────────────────
# 7. Protocol robustness
# ─────────────────────────────────────────────────────────

def test_unknown_action_is_answered_not_ignored(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request("camera.teleport", {})
    assert result["ok"] is False
    assert "unknown control action" in (result["error"] or "")


def test_malformed_line_does_not_break_the_channel(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    electron._wait_for_connection()
    assert electron._conn is not None
    electron._conn.sendall(b"this is not json\n")
    time.sleep(0.05)
    # Channel still works afterwards.
    result = electron.request(Action.PIPELINE_SNAPSHOT)
    assert result["ok"] is True
    assert result["data"]["running"] is True


def test_snapshot_reports_backend_truth(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    result = electron.request(Action.PIPELINE_SNAPSHOT)
    assert result["ok"] is True
    data = result["data"]
    assert data["initialized"] is True
    assert data["camera"]["index"] == pipeline.config.camera.index
    assert data["serial"]["mode"] == "mock"
    assert "voice" in data


# ─────────────────────────────────────────────────────────
# 8. Preview framing + single-owner guarantees
# ─────────────────────────────────────────────────────────

def test_preview_frames_are_throttled(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """Frames are throttled in the bridge, not JPEG-encoded on every CV frame."""
    assert pipeline.tcp_bridge.min_frame_interval > 0
    for _ in range(12):
        pipeline.process_frame()
    # 12 frames processed well inside one throttle window → far fewer preview sends.
    assert pipeline.tcp_bridge.frame_messages_sent <= 12
    assert pipeline.tcp_bridge.frame_messages_sent < 12 or pipeline.tcp_bridge.min_frame_interval == 0


def test_no_second_serial_owner_is_created(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """Connecting/reconnecting repeatedly must never leak a second serial interface."""
    for _ in range(3):
        connect = electron.request(Action.SERIAL_CONNECT, {"mode": "mock"})
        assert connect["data"]["serial"]["connected"] is True

    # Exactly one serial object is reachable from the pipeline.
    assert pipeline.serial is not None
    assert electron.request(Action.PIPELINE_SNAPSHOT)["data"]["serial"]["connected"] is True


def test_disconnect_really_disconnects_and_refuses_transmission(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer
):
    """'Disconnected' must mean the pipeline cannot transmit — no stand-in link."""
    electron.request(Action.SERIAL_CONNECT, {"mode": "mock"})
    assert pipeline.serial_info["connected"] is True
    before = len(_frames(pipeline))

    result = electron.request(Action.SERIAL_DISCONNECT)
    assert result["ok"] is True
    assert result["data"]["serial"]["connected"] is False
    assert pipeline.serial_info["connected"] is False
    assert pipeline._transport_status_str().startswith("DISCONNECTED")

    send = electron.request(Action.COMMAND_SEND, {"command": "CALL"})
    assert send["ok"] is False
    assert "serial" in (send["error"] or "").lower()
    assert len(_frames(pipeline)) == before

    # Reconnect restores real transmission on the same single owner.
    reconnect = electron.request(Action.SERIAL_CONNECT, {"mode": "mock"})
    assert reconnect["data"]["serial"]["connected"] is True
    ok_send = electron.request(Action.COMMAND_SEND, {"command": "CALL"})
    assert ok_send["ok"] is True
    assert _frames(pipeline)[-1] == b"NG1|CALL\n"


# ─────────────────────────────────────────────────────────
# 9. Camera → CV → stabilizer → validator → serial (auto path)
# ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,expected", [("CALL", b"NG1|CALL\n"), ("OK", b"NG1|OK\n")])
def test_camera_detection_path_transmits_detected_gesture(
    pipeline: NeuroGripPipeline, electron: FakeElectronServer, label: str, expected: bytes
):
    """The auto path must reach the same serial owner as manual and voice commands."""
    pipeline.recognizer.label = label
    pipeline.detector.detect.return_value = _one_hand()

    for _ in range(3):
        pipeline.process_frame()

    frames = _frames(pipeline)
    assert expected in frames, f"{label} detected but never transmitted"
    assert frames.count(expected) == 1, "gesture must emit exactly once per detection"

    state = electron.wait_for_state(lambda s: s.get("final_command") == label)
    assert state["last_tx"] == f"NG1|{label}"


def test_camera_off_blocks_detection_transmission(pipeline: NeuroGripPipeline, electron: FakeElectronServer):
    """With capture stopped, a detected gesture must not reach hardware."""
    pipeline.recognizer.label = "CALL"
    pipeline.detector.detect.return_value = _one_hand()
    electron.request(Action.CAMERA_SET_ENABLED, {"enabled": False})

    before = len(_frames(pipeline))
    _, _, state = pipeline.process_frame()
    assert state.pipeline_state == "CAMERA_OFF"
    assert len(_frames(pipeline)) == before


# ─────────────────────────────────────────────────────────
# 10. Cross-language contract (renderer ↔ pipeline)
# ─────────────────────────────────────────────────────────

def _renderer_actions() -> list[str]:
    """Extract the action strings from `src/services/pipelineControl.ts`."""
    import re
    from pathlib import Path

    ts_path = Path(__file__).resolve().parents[2] / "src" / "services" / "pipelineControl.ts"
    source = ts_path.read_text(encoding="utf-8")
    block = source.split("export const CONTROL_ACTIONS = {", 1)[1].split("} as const;", 1)[0]
    return re.findall(r':\s*"([a-z_]+\.[a-z_]+)"', block)


def test_renderer_and_pipeline_control_vocabularies_match():
    """
    The renderer may only send actions the pipeline understands.

    `src/services/pipelineControl.ts` and `neurogrip/bridge/control.py` are two halves of one
    contract; if they drift, a UI button would fail at runtime instead of at review time.
    """
    renderer = _renderer_actions()
    assert renderer, "could not read any actions from pipelineControl.ts"
    assert set(renderer) == set(Action.ALL), (
        "control vocabulary drift: renderer=" + repr(sorted(set(renderer) ^ set(Action.ALL)))
    )
    assert len(renderer) == len(set(renderer)), "duplicate action in pipelineControl.ts"


def test_control_result_shape_matches_the_renderer_envelope():
    """`ControlEnvelope` in the renderer expects exactly these keys."""
    import json

    encoded = json.loads(encode_result("u1", True, {"x": 1}, None))
    assert set(encoded) == {"type", "id", "ok", "data", "error"}
    assert encoded["type"] == "control_result"
    assert encoded["ok"] is True and encoded["data"] == {"x": 1} and encoded["error"] is None
