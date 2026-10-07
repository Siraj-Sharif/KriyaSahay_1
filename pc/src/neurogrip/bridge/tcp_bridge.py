"""
neurogrip/bridge/tcp_bridge.py
------------------------------
Bidirectional JSON-lines bridge between the running NeuroGrip pipeline and the Electron UI.

Transport: the pipeline connects as a TCP **client** to Electron's server on
127.0.0.1:8765 and keeps that single connection open for the whole session.

  Python → Electron   state snapshots (10 Hz) + JPEG preview frames + control results
  Electron → Python   control requests (camera / serial / command / STOP / voice)

Before Phase 2 this channel was one-way, which is why every UI control had to be faked or
implemented with a throwaway ``python -c`` process. The reader thread below closes that
gap so the running pipeline stays the single owner of the camera and the serial link.

Framing is unchanged: one JSON object per line, UTF-8, ``\\n`` terminated.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import socket
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

from neurogrip.bridge.control import (
    ControlDispatcher,
    encode_result,
    parse_request,
)
from neurogrip.visualization.base import VisualizationState

# Fields of VisualizationState that must not be JSON-serialised verbatim.
_NON_SERIALISABLE = {"landmarks"}


class TCPIPBridge:
    """
    Bidirectional TCP bridge between the NeuroGrip pipeline and the Electron UI.

    Fault-tolerant by design: if the UI is not running, or the connection drops, the
    pipeline keeps processing frames and simply reports no state.

    Parameters
    ----------
    host, port : str, int
        Electron's TCP server address.
    min_send_interval : float
        Minimum seconds between state snapshots (default 0.1 → 10 Hz).
    min_frame_interval : float
        Minimum seconds between JPEG preview frames (default 0.1 → 10 Hz). Prevents the
        preview from JPEG-encoding every 30 fps frame on the pipeline thread.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        min_send_interval: float = 0.1,
        min_frame_interval: float = 0.1,
    ) -> None:
        self.host = host
        # NEUROGRIP_BRIDGE_PORT lets tests (and a second instance) use another port.
        self.port = int(port if port is not None else os.environ.get("NEUROGRIP_BRIDGE_PORT", "8765"))
        self.socket: Optional[socket.socket] = None
        self.last_send_time = 0.0
        self.min_send_interval = min_send_interval

        self._last_frame_time = 0.0
        self.min_frame_interval = min_frame_interval

        self._dispatcher = ControlDispatcher()
        self._write_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer = ""
        self._connected_once = False

        # Observability for tests / diagnostics
        self.state_messages_sent: int = 0
        self.frame_messages_sent: int = 0
        self.control_requests_handled: int = 0
        self.last_send_error: Optional[str] = None

        self._connect()

    # ──────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ──────────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Attempt to connect to the Electron TCP server. Silently fails if unavailable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((self.host, self.port))
            sock.settimeout(None)  # reader thread blocks; writes are guarded by a lock
            self.socket = sock
            self._buffer = ""
            self._connected_once = True
            print(f"[TCP Bridge] Connected to {self.host}:{self.port}")
            self._start_reader()
        except (ConnectionRefusedError, socket.timeout, OSError):
            self.socket = None

    def _drop_socket(self, reason: str) -> None:
        self.last_send_error = reason
        self._stop_reader()
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
        self.socket = None

    @property
    def is_connected(self) -> bool:
        return self.socket is not None

    # ──────────────────────────────────────────────────────────────────
    # Control channel (Electron → Python)
    # ──────────────────────────────────────────────────────────────────

    def register_control_handler(self, action: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        """Register a handler for one control action."""
        self._dispatcher.register(action, handler)

    @property
    def dispatcher(self) -> ControlDispatcher:
        return self._dispatcher

    def _start_reader(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="neurogrip-bridge-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        self._stop_event.set()
        self._reader_thread = None

    def _read_loop(self) -> None:
        """Read control requests off the socket and answer them on the same connection."""
        while not self._stop_event.is_set():
            sock = self.socket
            if sock is None:
                return
            try:
                chunk = sock.recv(65536)
            except (OSError, socket.timeout):
                if self._stop_event.is_set():
                    return
                time.sleep(0.05)
                continue
            except Exception:
                return

            if not chunk:
                # Peer closed the connection.
                self._drop_socket("peer closed connection")
                return

            self._buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if line:
                    self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        request = parse_request(line)
        if request is None:
            # Not a control message: this direction only carries control requests.
            return

        request_id = request.get("id")
        ok_flag, data, error = self._dispatcher.handle(request)
        self.control_requests_handled += 1
        self._send_raw(encode_result(request_id, ok_flag, data, error))

    def _send_raw(self, payload: str) -> bool:
        """Write one already-encoded line. Returns False if the link is down."""
        sock = self.socket
        if sock is None:
            return False
        data = (payload + "\n").encode("utf-8")
        try:
            with self._write_lock:
                sock.sendall(data)
            return True
        except (OSError, BrokenPipeError, ConnectionResetError) as exc:
            self._drop_socket(f"send error: {exc}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # State stream (Python → Electron)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise_landmarks(hands: Any) -> list[list[list[float]]]:
        """Compact ``[[[x, y, z], ...], ...]`` copy of the detected hands (never None)."""
        out: list[list[list[float]]] = []
        for hand in hands or []:
            points = getattr(hand, "landmarks", None) or []
            row: list[list[float]] = []
            for point in points:
                try:
                    row.append([round(float(point.x), 4), round(float(point.y), 4), round(float(point.z), 4)])
                except (TypeError, ValueError, AttributeError):
                    continue
            out.append(row)
        return out

    @staticmethod
    def serialise_state(state: VisualizationState, extras: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Serialise the complete VisualizationState plus runtime extras.

        Uses the dataclass itself rather than a hand-written field list: the previous
        hand-written dict silently dropped ten fields (is_armed, is_tx_permitted,
        final_command, taxonomy_mapped, is_stabilized, stabilizer_state, stop_rule_*,
        hagrid_*) so the UI could not show real safety or transmission state.
        """
        payload: dict[str, Any] = {}
        for field in dataclasses.fields(state):
            if field.name in _NON_SERIALISABLE:
                continue
            payload[field.name] = getattr(state, field.name)
        # The landmark overlay in the UI is only real if it gets the actual points. Send a
        # compact first-hand array (4-decimal normalised coords ≈ 1.5 kB) at the state rate.
        payload["landmarks"] = TCPIPBridge._serialise_landmarks(state.landmarks)
        payload["landmarks_count"] = len(state.landmarks)
        payload["message"] = state.message or ""
        if extras:
            payload.update(extras)
        return payload

    def send(self, state: VisualizationState, extras: Optional[dict[str, Any]] = None) -> None:
        """Send a state snapshot if connected and the throttle interval has elapsed."""
        now = time.time()
        if now - self.last_send_time < self.min_send_interval:
            return

        if self.socket is None:
            self._connect()
            if self.socket is None:
                return

        payload = self.serialise_state(state, extras)
        if self._send_raw(json.dumps(payload)):
            self.last_send_time = now
            self.state_messages_sent += 1

    def send_frame(self, frame: np.ndarray, force: bool = False) -> bool:
        """Send a JPEG-encoded camera preview frame (throttled to `min_frame_interval`)."""
        now = time.time()
        if not force and now - self._last_frame_time < self.min_frame_interval:
            return False

        if self.socket is None:
            self._connect()
            if self.socket is None:
                return False

        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return False
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            if self._send_raw(json.dumps({"type": "frame", "data": b64, "ts": now})):
                self._last_frame_time = now
                self.frame_messages_sent += 1
                return True
        except Exception as exc:  # pragma: no cover - defensive
            self.last_send_error = str(exc)
        return False

    def close(self) -> None:
        """Stop the reader and close the socket."""
        self._stop_reader()
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None
