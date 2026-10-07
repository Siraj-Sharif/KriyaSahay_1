"""
neurogrip/app/control.py
------------------------
Binds bridge control actions to operations on the running pipeline.

This is the bridge-side half of the Phase 2 control channel: every handler below calls a
method on the *live* `NeuroGripPipeline`, so the UI can only ever change state that the
backend genuinely owns. Nothing here opens its own camera, port or process.

Registered once from `NeuroGripPipeline.initialize()`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neurogrip.bridge.control import Action, fail, ok

if TYPE_CHECKING:  # pragma: no cover
    from neurogrip.app.pipeline import NeuroGripPipeline

logger = logging.getLogger(__name__)


def register_pipeline_controls(pipeline: "NeuroGripPipeline") -> None:
    """Register every control action on the pipeline's bridge."""
    bridge = pipeline.tcp_bridge

    # ── Camera ────────────────────────────────────────────────────────

    def _camera_list(payload: dict[str, Any]):
        # Match the OpenCV camera abstraction's existing 0–9 fallback scan while keeping
        # discovery inside the process that owns the active capture device.
        cameras = pipeline.list_cameras(probe_limit=10)
        return ok({"cameras": cameras, "active": pipeline.camera_info["index"]})

    def _camera_select(payload: dict[str, Any]):
        index = payload.get("index")
        if index is None:
            return fail("camera.select requires an 'index'")
        success, error = pipeline.set_camera_index(index)
        if not success:
            return fail(error or "camera select failed", {"camera": pipeline.camera_info})
        return ok({"camera": pipeline.camera_info})

    def _camera_set_enabled(payload: dict[str, Any]):
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            return fail("camera.set_enabled requires a boolean 'enabled'")
        success, error = pipeline.set_camera_enabled(enabled)
        if not success:
            return fail(error or "camera enable failed", {"camera": pipeline.camera_info})
        return ok({"camera": pipeline.camera_info})

    # ── Serial ────────────────────────────────────────────────────────

    def _serial_list_ports(payload: dict[str, Any]):
        ports = pipeline.list_available_serial_ports()
        return ok({"ports": ports, "serial": pipeline.serial_info})

    def _serial_connect(payload: dict[str, Any]):
        port = payload.get("port")
        baud = payload.get("baud")
        mode = payload.get("mode") or "real"
        success, error = pipeline.configure_serial(port=port, baud=baud, mode=mode)
        if not success:
            return fail(error or "serial connect failed", {"serial": pipeline.serial_info})
        return ok({"serial": pipeline.serial_info})

    def _serial_disconnect(payload: dict[str, Any]):
        pipeline.disconnect_serial()
        return ok({"serial": pipeline.serial_info})

    # ── Commands ──────────────────────────────────────────────────────

    def _command_send(payload: dict[str, Any]):
        command = payload.get("command")
        source = str(payload.get("source") or "manual")
        if not command:
            return fail("command.send requires a 'command'")
        success, data, error = pipeline.dispatch_command(command, source=source)
        if not success:
            return fail(error or "command rejected", data)
        return ok(data)

    # ── STOP / safety ─────────────────────────────────────────────────

    def _stop_set_armed(payload: dict[str, Any]):
        armed = payload.get("armed")
        if not isinstance(armed, bool):
            return fail("stop.set_armed requires a boolean 'armed'")
        state = pipeline.set_stop_armed(armed)
        return ok({"armed": state, "serial": pipeline.serial_info})

    # ── Voice ─────────────────────────────────────────────────────────

    def _voice_set_state(payload: dict[str, Any]):
        pipeline.set_voice_state(str(payload.get("state") or "idle"), payload.get("engine"))
        return ok({"voice": pipeline.voice_info})

    def _voice_transcript(payload: dict[str, Any]):
        text = payload.get("text")
        if not isinstance(text, str):
            return fail("voice.transcript requires string 'text'")
        success, data, error = pipeline.handle_voice_transcript(text, engine=payload.get("engine"))
        data = dict(data or {})
        data["voice"] = pipeline.voice_info
        if not success:
            return fail(error or "voice command failed", data)
        return ok(data)

    # ── Introspection / lifecycle ────────────────────────────────────

    def _snapshot(payload: dict[str, Any]):
        return ok(
            {
                "camera": pipeline.camera_info,
                "serial": pipeline.serial_info,
                "voice": pipeline.voice_info,
                "running": pipeline.is_running,
                "initialized": pipeline.is_initialized,
                "cv_ready": bool(getattr(pipeline.detector, "is_initialized", False)),
                "stop_armed": pipeline.is_stop_armed,
                "state": pipeline.state.name,
                "runtime": pipeline.runtime_info(),
            }
        )

    def _app_shutdown(payload: dict[str, Any]):
        # The control reader acknowledges on this same socket; the pipeline loop then exits
        # through its existing finally/shutdown path, releasing camera and serial handles.
        pipeline.stop()
        return ok({"state": "SHUTTING_DOWN"})

    bridge.register_control_handler(Action.CAMERA_LIST, _camera_list)
    bridge.register_control_handler(Action.CAMERA_SELECT, _camera_select)
    bridge.register_control_handler(Action.CAMERA_SET_ENABLED, _camera_set_enabled)
    bridge.register_control_handler(Action.SERIAL_LIST_PORTS, _serial_list_ports)
    bridge.register_control_handler(Action.SERIAL_CONNECT, _serial_connect)
    bridge.register_control_handler(Action.SERIAL_DISCONNECT, _serial_disconnect)
    bridge.register_control_handler(Action.COMMAND_SEND, _command_send)
    bridge.register_control_handler(Action.STOP_SET_ARMED, _stop_set_armed)
    bridge.register_control_handler(Action.VOICE_SET_STATE, _voice_set_state)
    bridge.register_control_handler(Action.VOICE_TRANSCRIPT, _voice_transcript)
    bridge.register_control_handler(Action.PIPELINE_SNAPSHOT, _snapshot)
    bridge.register_control_handler(Action.APP_SHUTDOWN, _app_shutdown)

    logger.info("Registered %d pipeline control actions: %s", len(bridge.dispatcher.registered_actions), ", ".join(bridge.dispatcher.registered_actions))
