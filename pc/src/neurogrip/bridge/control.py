"""
neurogrip/bridge/control.py
---------------------------
Downstream control protocol for the Electron UI.

The TCP bridge already carries Python → Electron state at 10 Hz. This module adds the
missing direction: **Electron → Python** control requests, multiplexed over that same
socket. One persistent connection, one pipeline, one serial owner — no per-action
``python -c`` subprocesses.

Wire format (newline-delimited JSON, same framing as the state stream):

    request   Electron → Python
    {"type": "control", "id": "c7", "action": "serial.connect",
     "payload": {"port": "COM9", "baud": 115200}}

    response  Python → Electron
    {"type": "control_result", "id": "c7", "ok": true,
     "data": {...}, "error": null}

Actions are the single vocabulary shared with the renderer; keep
``src/services/pipelineControl.ts`` in sync when adding one.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Action vocabulary (shared contract with the renderer)
# ─────────────────────────────────────────────────────────

class Action:
    """Control actions accepted by the running pipeline."""

    # Camera
    CAMERA_LIST = "camera.list"
    CAMERA_SELECT = "camera.select"
    CAMERA_SET_ENABLED = "camera.set_enabled"

    # Serial / hardware
    SERIAL_LIST_PORTS = "serial.list_ports"
    SERIAL_CONNECT = "serial.connect"
    SERIAL_DISCONNECT = "serial.disconnect"

    # Commands
    COMMAND_SEND = "command.send"

    # STOP / safety
    STOP_SET_ARMED = "stop.set_armed"

    # Voice
    VOICE_SET_STATE = "voice.set_state"
    VOICE_TRANSCRIPT = "voice.transcript"

    # Introspection
    PIPELINE_SNAPSHOT = "pipeline.snapshot"

    ALL = (
        CAMERA_LIST, CAMERA_SELECT, CAMERA_SET_ENABLED,
        SERIAL_LIST_PORTS, SERIAL_CONNECT, SERIAL_DISCONNECT,
        COMMAND_SEND, STOP_SET_ARMED,
        VOICE_SET_STATE, VOICE_TRANSCRIPT,
        PIPELINE_SNAPSHOT,
    )


class ControlResult(Protocol):
    """The (ok, data, error) triple every handler returns."""

    def __call__(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any], Optional[str]]:
        ...


def ok(data: Optional[dict[str, Any]] = None) -> tuple[bool, dict[str, Any], None]:
    """Successful control result."""
    return True, (data or {}), None


def fail(error: str, data: Optional[dict[str, Any]] = None) -> tuple[bool, dict[str, Any], str]:
    """Failed control result with a human-readable reason."""
    return False, (data or {}), error


def parse_request(raw: str) -> Optional[dict[str, Any]]:
    """
    Parse one control line.

    Returns the request dict, or ``None`` if the line is not a control request.
    """
    import json

    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict) or msg.get("type") != "control":
        return None
    action = msg.get("action")
    if not isinstance(action, str) or action not in Action.ALL:
        return {"id": msg.get("id"), "action": action, "payload": {}, "_invalid": True}
    payload = msg.get("payload")
    return {
        "id": msg.get("id"),
        "action": action,
        "payload": payload if isinstance(payload, dict) else {},
        "_invalid": False,
    }


def encode_result(request_id: Any, ok_flag: bool, data: dict[str, Any], error: Optional[str]) -> str:
    """Encode a control response as one JSON line (terminator NOT included)."""
    import json

    return json.dumps(
        {
            "type": "control_result",
            "id": request_id,
            "ok": bool(ok_flag),
            "data": data or {},
            "error": error,
        }
    )


class ControlDispatcher:
    """
    Routes control actions to handlers and normalises every outcome.

    Handlers are registered per action and always produce an ``(ok, data, error)`` triple,
    so a broken action can never take the pipeline down with it.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ControlResult] = {}
        self.handled_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None

    def register(self, action: str, handler: ControlResult) -> None:
        self._handlers[action] = handler

    def handle(self, request: dict[str, Any]) -> tuple[bool, dict[str, Any], Optional[str]]:
        """Dispatch one parsed request. Never raises."""
        if request.get("_invalid"):
            self.error_count += 1
            self.last_error = f"unknown control action: {request.get('action')!r}"
            return fail(self.last_error)

        action = request.get("action")
        handler = self._handlers.get(action)
        if handler is None:
            self.error_count += 1
            self.last_error = f"no handler registered for action {action!r}"
            return fail(self.last_error)

        try:
            result = handler(request.get("payload") or {})
            ok_flag, data, error = result
            self.handled_count += 1
            if not ok_flag:
                self.error_count += 1
                self.last_error = error
            return ok_flag, data, error
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Control action %s raised", action)
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return fail(self.last_error)

    @property
    def registered_actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))
