"""
neurogrip/app/pipeline.py
─────────────────────────
NeuroGrip PC-Side End-to-End Pipeline Orchestrator.
Connects Camera, Hand Tracking, Feature Extraction, Recognition, Temporal Stabilization,
Command Validation, Serial Communication, and Visualization Overlay into one controlled execution loop.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from neurogrip.camera.base import CameraInterface, FrameContainer
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.camera.opencv_camera import OpenCVCamera
from neurogrip.commands.definitions import NeuroGripCommand, command_name
from neurogrip.commands.validator import CommandValidator, PipelineState, ValidationResult
from neurogrip.communication.base import SerialInterface
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.communication.serial_port import RealSerialInterface, list_serial_ports
from neurogrip.communication.transport import CommandTransport, DisplayConsoleTransport
from neurogrip.config.settings import AppConfig
from neurogrip.features.extractor import FeatureExtractor
from neurogrip.hand_tracking.detector import HandDetector, HandDetectorError
from neurogrip.hand_tracking.landmarks import DetectionResult
from neurogrip.recognition.base import GestureRecognizer, RecognitionResult
from neurogrip.recognition.hybrid import HybridRecognizer
from neurogrip.recognition.ml_recognizer import MLRecognizer
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from neurogrip.stabilization.temporal import StabilizerResult, StabilizerState, TemporalStabilizer
from neurogrip.visualization.base import VisualizationState
from neurogrip.visualization.renderer import OverlayRenderer
from neurogrip.bridge import TCPIPBridge
from neurogrip.communication.protocol import ProtocolEncoder
from neurogrip.voice.intents import parse_intent

logger = logging.getLogger(__name__)


def _encoded_frame_for(command) -> str:
    """
    Return the exact NG1 protocol frame string that the encoder writes for a command
    (newline stripped), so telemetry reports real transmitted bytes.

    Falls back to the command's own value if the encoder rejects it.
    """
    frame = ProtocolEncoder.encode_command_string(command)
    if frame is None:
        return f"NG1|{getattr(command, 'value', command)}"
    return frame.rstrip("\r\n")


class NeuroGripPipeline:
    """
    End-to-End PC-side Pipeline Orchestrator.

    Component Flow:
    Camera -> Hand Detector -> Feature Extractor -> Gesture Recognizer ->
    Temporal Stabilizer -> Command Validator -> Serial Interface -> [Visualization Overlay]
    """

    #: Command sources that represent an explicit user request (button press, voice,
    #: manual override). These outrank the detector's duplicate suppression.
    _EXPLICIT_COMMAND_SOURCES = frozenset({"manual", "voice", "override"})

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        camera: Optional[CameraInterface] = None,
        serial: Optional[SerialInterface] = None,
        recognizer: Optional[GestureRecognizer] = None,
        detector: Optional[HandDetector] = None,
        transport: Optional[CommandTransport] = None,
        bridge: Optional[TCPIPBridge] = None,
    ) -> None:
        self.config = config or AppConfig.default()

        # 1. Camera Abstraction
        if camera is not None:
            self.camera = camera
        else:
            if self.config.camera.backend == "mock":
                self.camera = MockCamera(config=self.config.camera)
            else:
                self.camera = OpenCVCamera(config=self.config.camera)

        # 2. Hand Tracking Detector
        self.detector = detector or HandDetector(config=self.config.hand_tracking)

        # 3. Feature Extractor
        self.feature_extractor = FeatureExtractor(config=self.config.features)

        # 4. Gesture Recognizer (Production Hybrid: ML Extra Trees + Rule-Based STOP Safety)
        if recognizer is not None:
            self.recognizer = recognizer
        else:
            self.recognizer = HybridRecognizer(config=self.config)
            logger.info("Pipeline initialized with Production HybridRecognizer (ML + Rule-based STOP safety).")

        # 5. Temporal Stabilizer
        self.stabilizer = TemporalStabilizer(config=self.config)

        # 6. Command Validator
        self.validator = CommandValidator()

        # 7. Transport Boundary Interface (Stage 1 Default: DisplayConsoleTransport)
        self.transport = transport or DisplayConsoleTransport()

        # 8. Serial Communication (Backward Compatibility)
        if serial is not None:
            self.serial = serial
        else:
            if self.config.serial.mock:
                self.serial = MockSerialInterface()
            else:
                self.serial = RealSerialInterface(config=self.config.serial)

        # 9. Visualization Renderer
        self.renderer = OverlayRenderer(config=self.config.visualization)

        # 10. Real-time TCP Bridge for Electron/React frontend (bidirectional control + state)
        self.tcp_bridge = bridge if bridge is not None else TCPIPBridge()

        # Orchestrator state tracking
        self._pipeline_state: PipelineState = PipelineState.STARTING
        self._is_stop_armed: bool = False
        self._stop_arm_timestamp_ms: float = 0.0
        self._is_running: bool = False
        self._is_initialized: bool = False

        # Diagnostic state tracking
        self.last_detection_result: Optional[DetectionResult] = None
        self.last_features: Optional[np.ndarray] = None
        self.last_rec_result: Optional[RecognitionResult] = None
        self.last_stab_result: Optional[StabilizerResult] = None
        self.last_val_result: Optional[ValidationResult] = None

        # FPS metrics
        self._fps_history: list[float] = []
        self._last_frame_time: float = 0.0
        self.current_fps: float = 0.0

        # ── Phase 2: runtime control state ──────────────────────────────
        # All of it is owned here so the UI can never hold state the backend does not.
        self._control_lock = threading.RLock()
        self._camera_enabled: bool = True
        self._camera_error: Optional[str] = None
        self._last_error: Optional[str] = None
        self._frames_sent: int = 0
        self._manual_tx_count: int = 0
        self._voice_state: str = "idle"
        self._voice_transcript: str = ""
        self._voice_command: Optional[str] = None
        self._voice_reply: str = ""
        self._voice_engine: str = "N/A"

    @property
    def state(self) -> PipelineState:
        return self._pipeline_state

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def is_stop_armed(self) -> bool:
        return self._is_stop_armed

    def arm_stop(self) -> None:
        """Arm the software STOP safety detection path."""
        self._is_stop_armed = True
        self._stop_arm_timestamp_ms = time.time() * 1000.0
        logger.warning("SOFTWARE STOP ARMED!")

    def disarm_stop(self) -> None:
        """Disarm the software STOP safety detection path."""
        self._is_stop_armed = False
        logger.info("Software STOP disarmed.")

    def toggle_stop_arm(self) -> bool:
        """Toggle software STOP arming state."""
        if self._is_stop_armed:
            self.disarm_stop()
        else:
            self.arm_stop()
        return self._is_stop_armed

    def initialize(self) -> bool:
        """
        Initialize all pipeline components and open hardware interfaces.

        Returns
        -------
        bool
            True if all components initialized successfully, False otherwise.
        """
        if self._is_initialized:
            return True

        logger.info("Initializing NeuroGrip Pipeline...")
        self._pipeline_state = PipelineState.INITIALIZING

        try:
            # 1. Open Camera Interface
            if not self.camera.open():
                logger.error("Pipeline initialization failed: Camera open failed.")
                self._pipeline_state = PipelineState.ERROR
                return False

            # 2. Initialize Hand Detector (MediaPipe model)
            try:
                self.detector.initialize()
            except HandDetectorError as e:
                logger.warning("HandDetector initialization warning/error: %s", e)

            # 3. Connect Serial Interface
            if self.config.serial.enabled:
                if not self.serial.connect():
                    logger.warning("Serial interface failed to connect. Pipeline proceeding in disconnected mode.")

            # 4. Bind the UI control channel to this live pipeline instance.
            #    Deferred import avoids a circular import at module load.
            from neurogrip.app.control import register_pipeline_controls

            register_pipeline_controls(self)

            self._is_initialized = True
            self._is_running = True
            self._pipeline_state = PipelineState.TRACKING
            self._camera_error = None if self.camera.is_opened else "camera not opened"
            logger.info("NeuroGrip Pipeline initialized and ready.")
            return True

        except Exception as e:
            logger.error("Unhandled error during pipeline initialization: %s", e)
            self._pipeline_state = PipelineState.ERROR
            self.shutdown()
            return False

    def process_frame(self) -> tuple[FrameContainer, Optional[np.ndarray], VisualizationState]:
        """
        Process a single video frame through the end-to-end CV pipeline.

        Returns
        -------
        tuple[FrameContainer, Optional[np.ndarray], VisualizationState]
            (Raw frame container, Rendered overlay frame, VisualizationState snapshot)
        """
        if not self._is_initialized or not self._is_running:
            self.initialize()

        # Update FPS calculation
        now = time.time()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_history.append(inst_fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                self.current_fps = float(np.mean(self._fps_history))
        self._last_frame_time = now

        # Auto-disarm STOP if arming timeout expires
        if self._is_stop_armed and self.config.stop.arm_timeout_ms > 0:
            elapsed_ms = (now * 1000.0) - self._stop_arm_timestamp_ms
            if elapsed_ms > self.config.stop.arm_timeout_ms:
                logger.info("STOP arming timeout expired (%d ms). Auto-disarming.", self.config.stop.arm_timeout_ms)
                self.disarm_stop()

        # 1. Read frame from Camera
        #
        # When the UI has switched the camera off, the pipeline keeps running (so manual
        # commands, voice and serial still work) but genuinely stops consuming frames.
        if not self._camera_enabled:
            viz_state = VisualizationState(
                command="NO_COMMAND",
                confidence=0.0,
                stabilizer_state=self.stabilizer.current_state.name,
                pipeline_state="CAMERA_OFF",
                handedness="UNKNOWN",
                landmarks=[],
                fps=0.0,
                is_stop_active=self._is_stop_armed,
                is_ambiguous=False,
                message="CAMERA OFF — capture stopped by UI",
                hand_count="0",
                safety_status="STOP ARMED" if self._is_stop_armed else "STOP DISARMED",
                transport_status=self._transport_status_str(),
                last_tx=getattr(self, "_last_tx_frame", "NONE"),
                is_armed=self._is_stop_armed,
                final_command="NO_COMMAND",
            )
            self.current_fps = 0.0
            self.tcp_bridge.send(viz_state, extras=self.runtime_info())
            return FrameContainer(frame=None, timestamp_ms=int(time.time() * 1000), frame_index=0), None, viz_state

        frame_container = self.camera.read_frame()

        # Handle invalid camera frame (e.g. read failure or end of stream).
        # This path MUST still publish state: previously it returned early without
        # touching the bridge, so the UI froze on the last good frame and never learned
        # that the camera had failed.
        if not frame_container.is_valid or frame_container.frame is None:
            if self._camera_error is None:
                self._camera_error = "camera frame invalid"
                self._last_error = self._camera_error
            viz_state = VisualizationState(
                command="NO_HAND",
                confidence=0.0,
                stabilizer_state=self.stabilizer.current_state.name,
                pipeline_state=PipelineState.ERROR.name,
                handedness="UNKNOWN",
                landmarks=[],
                fps=self.current_fps,
                is_stop_active=self._is_stop_armed,
                is_ambiguous=False,
                message="CAMERA FRAME INVALID",
                hand_count="0",
                safety_status="STOP ARMED" if self._is_stop_armed else "STOP DISARMED",
                transport_status=self._transport_status_str(),
                last_tx=getattr(self, "_last_tx_frame", "NONE"),
                is_armed=self._is_stop_armed,
                final_command="NO_COMMAND",
            )
            self.tcp_bridge.send(viz_state, extras=self.runtime_info())
            return frame_container, None, viz_state

        raw_frame = frame_container.frame
        t_frame_start = time.perf_counter()

        # 2. Detect Hand Landmarks (MediaPipe Tasks)
        detection_res: DetectionResult = self.detector.detect(raw_frame, frame_container.timestamp_ms)
        self.last_detection_result = detection_res
        self.last_features = None
        self.last_rec_result = None
        self.last_stab_result = None
        self.last_val_result = None

        # One labelling rule for every branch: the UI must be able to show the pipeline's real
        # arming state, and the 1-hand branch previously hardcoded "NORMAL" while the other
        # branches reported ARMED/DISARMED (so the safety row changed text with hand presence).
        safety_status_str = "STOP ARMED" if self._is_stop_armed else "STOP DISARMED"

        # 3. Evaluate Hand Detection States (0 hands, 1 hand, 2+ hands)
        if detection_res.is_empty:
            # 0 Hands detected
            self._pipeline_state = PipelineState.NO_HAND
            self.stabilizer.reset()
            self.validator.validate(None, pipeline_state=PipelineState.NO_HAND)
            self.transport.send_idle()

            t_frame_end = time.perf_counter()
            frame_latency_ms = (t_frame_end - t_frame_start) * 1000.0

            viz_state = VisualizationState(
                command="NO_COMMAND",
                confidence=0.0,
                stabilizer_state=self.stabilizer.current_state.name,
                pipeline_state=PipelineState.NO_HAND.name,
                handedness="UNKNOWN",
                landmarks=[],
                fps=self.current_fps,
                is_stop_active=False,
                is_ambiguous=False,
                message="IDLE / 0 HANDS DETECTED",
                model_used="N/A",
                raw_prediction="NONE",
                hand_count="0",
                latency_ms=frame_latency_ms,
                safety_status="STOP ARMED" if self._is_stop_armed else "STOP DISARMED",
                routing_str="NONE -> NO_COMMAND",
                transport_status=self._transport_status_str(),
                last_tx=getattr(self, "_last_tx_frame", "NONE"),
                is_armed=self._is_stop_armed,
                final_command="NO_COMMAND",
            )

        elif detection_res.is_ambiguous:
            # 2+ Hands detected -> AMBIGUOUS / Safe State
            self._pipeline_state = PipelineState.AMBIGUOUS
            self.stabilizer.reset()
            self.validator.validate(None, pipeline_state=PipelineState.AMBIGUOUS)
            self.transport.send_idle()

            t_frame_end = time.perf_counter()
            frame_latency_ms = (t_frame_end - t_frame_start) * 1000.0

            viz_state = VisualizationState(
                command="NO_COMMAND",
                confidence=0.0,
                stabilizer_state=StabilizerState.UNKNOWN_STATE.name,
                pipeline_state=PipelineState.AMBIGUOUS.name,
                handedness="UNKNOWN",
                landmarks=detection_res.hands,
                fps=self.current_fps,
                is_stop_active=False,
                is_ambiguous=True,
                message="AMBIGUOUS STATE: 2+ HANDS DETECTED",
                model_used="N/A",
                raw_prediction="MULTIPLE_HANDS",
                hand_count="2+",
                latency_ms=frame_latency_ms,
                safety_status="STOP ARMED" if self._is_stop_armed else "STOP DISARMED",
                routing_str="AMBIGUOUS -> NO_COMMAND",
                transport_status=self._transport_status_str(),
                last_tx=getattr(self, "_last_tx_frame", "NONE"),
                is_armed=self._is_stop_armed,
                final_command="NO_COMMAND",
            )

        else:
            # Exactly 1 Hand detected -> Primary Processing Path
            primary_hand = detection_res.primary_hand
            assert primary_hand is not None

            # 4. Feature Extraction (68-dim vector)
            features = self.feature_extractor.extract(primary_hand)
            self.last_features = features

            # 5. Production Hybrid Gesture Recognition
            rec_result: RecognitionResult = self.recognizer.predict(
                features,
                hand_landmarks=primary_hand,
                is_stop_armed=True,
                raw_frame=raw_frame,
            )
            self.last_rec_result = rec_result

            # 6. Temporal Stabilization on FINAL CANONICAL COMMAND
            stab_res: StabilizerResult = self.stabilizer.update(
                rec_result, is_stop_armed=True
            )
            self.last_stab_result = stab_res

            # Determine pipeline state from stabilizer
            display_cmd = stab_res.stable_command or rec_result.label
            if stab_res.state in (StabilizerState.STABLE, StabilizerState.STOP):
                self._pipeline_state = PipelineState.STABLE_CMD
            else:
                self._pipeline_state = PipelineState.TRACKING

            # Determine transport status string
            t_status_str = self._transport_status_str()

            # 7. Command Validation & Transport Emission
            is_tx_ok = False
            if stab_res.emitted_command and stab_res.emitted_command != "NO_COMMAND":
                val_res: ValidationResult = self.validator.validate(
                    stab_res.emitted_command, pipeline_state=self._pipeline_state
                )
                self.last_val_result = val_res

                if val_res.is_valid and val_res.command and val_res.command.value != "NO_COMMAND":
                    self.transport.send_command(val_res.command)
                    is_tx_ok = True
                    if self.config.serial.enabled and hasattr(self, "serial") and self.serial is not None:
                        try:
                            tx_ok = self.serial.send_command(val_res.command)
                            if tx_ok:
                                self._frames_sent += 1
                                # Report the frame that was ACTUALLY encoded and written to the
                                # wire (e.g. "NG1|GRIP"), not the internal resolved vocabulary
                                # value (e.g. "GRAB"), so UI telemetry cannot disagree with the
                                # bytes the ESP32 received.
                                self._last_tx_frame = _encoded_frame_for(val_res.command)
                        except Exception as e:
                            logger.error("Unhandled serial transmission error: %s", e)
                else:
                    self.transport.send_idle()
            else:
                self.transport.send_idle()

            # Diagnostic metrics
            stop_rule_trig = (getattr(rec_result, "model_used", None) == "RULE_BASED" and getattr(rec_result, "label", None) == "STOP")
            spread_val = 0.0
            if hasattr(self.recognizer, "rule_recognizer"):
                raw_spread = self.recognizer.rule_recognizer.get_thumb_spread_metric(primary_hand)
                if isinstance(raw_spread, (int, float)):
                    spread_val = float(raw_spread)
            stop_metrics_str = f"spread={spread_val:.3f}"
            hagrid_raw_val = str(getattr(rec_result, "raw_prediction", "NONE"))
            raw_conf = getattr(rec_result, "confidence", 0.0)
            hagrid_conf_val = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
            tax_mapped_val = str(getattr(rec_result, "label", "UNKNOWN"))
            is_armed_val = self._is_stop_armed
            is_stab_val = (stab_res.state in (StabilizerState.STOP, StabilizerState.STABLE))

            if tax_mapped_val == "STOP" or stab_res.state == StabilizerState.STOP or hagrid_raw_val.lower() in ("stop", "stop_inverted", "palm"):
                logger.info(
                    "[STOP DIAGNOSTIC] RULE=%s (%s) | HAGRID_RAW=%s (conf=%.2f) | TAXONOMY=%s | ARMED=%s | STABILIZED=%s | FINAL=%s | TX=%s",
                    stop_rule_trig, stop_metrics_str, hagrid_raw_val, hagrid_conf_val, tax_mapped_val, is_armed_val, is_stab_val, display_cmd, is_tx_ok
                )

            # Protocol frame display string for Stage 1 boundary
            if hasattr(self.transport, "latest_frame") and self.transport.latest_frame:
                msg_str = f"PROTOCOL: {self.transport.latest_frame}"
            else:
                msg_str = "IDLE / NO COMMAND"

            t_frame_end = time.perf_counter()
            frame_latency_ms = (t_frame_end - t_frame_start) * 1000.0

            model_used_str = getattr(rec_result, "model_used", "HYBRID")
            raw_pred_str = getattr(rec_result, "raw_prediction", "UNKNOWN")
            routing_str_val = f"{model_used_str} -> {display_cmd}"
            last_tx_val = getattr(self, "_last_tx_frame", "NONE")

            viz_state = VisualizationState(
                command=display_cmd,
                confidence=stab_res.confidence if stab_res.stable_command else rec_result.confidence,
                stabilizer_state=stab_res.state.name,
                pipeline_state=self._pipeline_state.name,
                handedness=primary_hand.handedness.value,
                landmarks=[primary_hand],
                fps=self.current_fps,
                is_stop_active=(stab_res.state == StabilizerState.STOP),
                is_ambiguous=False,
                message=msg_str,
                model_used=model_used_str,
                raw_prediction=raw_pred_str,
                hand_count="1",
                latency_ms=frame_latency_ms,
                safety_status=safety_status_str,
                routing_str=routing_str_val,
                transport_status=t_status_str,
                last_tx=last_tx_val,
                stop_rule_triggered=stop_rule_trig,
                stop_rule_metrics=stop_metrics_str,
                hagrid_raw=hagrid_raw_val,
                hagrid_conf=hagrid_conf_val,
                taxonomy_mapped=tax_mapped_val,
                is_armed=is_armed_val,
                is_stabilized=is_stab_val,
                final_command=display_cmd,
                is_tx_permitted=is_tx_ok,
            )


        # 8. Render Visualization Overlay
        rendered_frame = self.renderer.render(raw_frame, viz_state)

        # 9. Send real camera frame from existing Python camera to Electron.
        #    Throttled inside the bridge so the preview never JPEG-encodes every CV frame.
        if self._camera_enabled and raw_frame is not None and raw_frame.size > 0:
            try:
                self.tcp_bridge.send_frame(raw_frame)
            except Exception:
                pass  # Non-blocking: frame preview never blocks CV

        # 10. Send the complete state snapshot to the TCP bridge (throttled to 10 Hz)
        self.tcp_bridge.send(viz_state, extras=self.runtime_info())

        return frame_container, rendered_frame, viz_state

    # ══════════════════════════════════════════════════════════════════
    # Phase 2 — Runtime control surface
    #
    # Every method here is called from the bridge's control reader thread and mutates
    # pipeline state under `self._control_lock`. The pipeline remains the single owner
    # of the camera and of the serial link; the UI only ever expresses intent.
    # ══════════════════════════════════════════════════════════════════

    def set_stop_armed(self, armed: bool) -> bool:
        """Set the software STOP arming state explicitly (no toggle ambiguity)."""
        with self._control_lock:
            if armed:
                self.arm_stop()
            else:
                self.disarm_stop()
            return self._is_stop_armed

    # ── Camera ────────────────────────────────────────────────────────

    def set_camera_index(self, index: int) -> tuple[bool, str]:
        """
        Switch the live camera to a different device index.

        Re-opens the real capture device — the UI selection is only real if the backend
        device actually changes.
        """
        try:
            index = int(index)
        except (TypeError, ValueError):
            return False, f"invalid camera index: {index!r}"
        if index < 0:
            return False, "camera index must be >= 0"

        with self._control_lock:
            if index == int(self.config.camera.index) and self.camera.is_opened:
                return True, ""

            # Release the old device before touching the new one: two captures must never
            # hold the same camera, and a failed switch must not leave a stale handle open.
            self.camera.close()
            self.config.camera.index = index
            if not self._camera_enabled:
                logger.info("Camera index set to %d (capture currently stopped).", index)
                return True, ""

            opened = self.camera.open()
            if not opened:
                self._camera_enabled = False
                self._camera_error = f"camera {index} unavailable"
                self._last_error = self._camera_error
                logger.error("Camera %d could not be opened.", index)
                return False, self._camera_error

            self._camera_enabled = True
            self._camera_error = None
            self._last_error = None
            logger.info("Camera switched to index %d.", self.config.camera.index)
            return True, ""

    def set_camera_enabled(self, enabled: bool) -> tuple[bool, str]:
        """Start/stop actual camera capture. The pipeline process keeps running."""
        with self._control_lock:
            if enabled:
                if not self.camera.is_opened:
                    if not self.camera.open():
                        self._camera_enabled = False
                        self._camera_error = "camera unavailable"
                        self._last_error = self._camera_error
                        return False, self._camera_error
                self._camera_enabled = True
                self._camera_error = None
                self._last_error = None
                logger.info("Camera capture enabled.")
                return True, ""

            self._camera_enabled = False
            self.camera.close()
            # Nothing can be detected while the camera is off — drop stale motion state.
            self.stabilizer.reset()
            self.transport.send_idle()
            logger.info("Camera capture disabled.")
            return True, ""

    def list_cameras(self, probe_limit: int = 5) -> list[dict[str, Any]]:
        """
        Enumerate camera indices the backend can actually open.

        Probing opens each index briefly; only indices that open are reported as
        available, so the UI never offers a device the pipeline cannot use.
        """
        cameras: list[dict[str, Any]] = []
        current = int(self.config.camera.index)

        if self.config.camera.backend == "mock":
            return [
                {"index": current, "label": f"Mock camera {current}", "available": True, "active": True},
            ]

        try:
            import cv2
        except Exception:  # pragma: no cover - cv2 is a hard dependency in practice
            return [{"index": current, "label": f"Camera {current}", "available": self.camera.is_opened, "active": True}]

        # Always offer the configured index first, then probe the rest.
        for index in [current] + [i for i in range(probe_limit) if i != current]:
            available = False
            if index == current and self.camera.is_opened:
                available = True
            else:
                try:
                    cap = cv2.VideoCapture(index)
                    available = bool(cap and cap.isOpened())
                    if cap is not None:
                        cap.release()
                except Exception:
                    available = False
            if available or index == current:
                label = self._camera_label(index)
                cameras.append(
                    {
                        "index": index,
                        "label": label,
                        "available": available,
                        "active": index == current,
                    }
                )
        return cameras

    @staticmethod
    def _camera_label(index: int) -> str:
        """Best-effort human label for a camera index."""
        try:
            from neurogrip.camera.devices import describe_camera

            described = describe_camera(index)
            if described:
                return described
        except Exception:
            pass
        return f"Camera {index}"

    def _transport_status_str(self) -> str:
        """
        One place that decides how the serial link is described to the UI.

        Replaces the duplicated ``CONNECTED (COMx)`` / ``MOCK`` / ``DISABLED`` inline logic
        so the state stream can never disagree with `serial_info`.
        """
        if not self.config.serial.enabled:
            return "DISABLED"
        if self.serial is not None and self.serial.is_connected:
            if self.serial.state.name == "MOCK":
                return "MOCK"
            return f"CONNECTED ({self.config.serial.port})"
        return f"DISCONNECTED ({self.config.serial.port})"

    @property
    def camera_info(self) -> dict[str, Any]:
        """Truthful camera state for the UI."""
        actual_w = int(getattr(self.camera, "actual_width", 0) or 0)
        actual_h = int(getattr(self.camera, "actual_height", 0) or 0)
        resolution = f"{actual_w} × {actual_h}" if actual_w and actual_h else f"{self.config.camera.width} × {self.config.camera.height}"
        return {
            "index": int(self.config.camera.index),
            "enabled": bool(self._camera_enabled),
            "opened": bool(self.camera.is_opened),
            "state": self.camera.state.name,
            "backend": self.config.camera.backend,
            "resolution": resolution,
            "target_resolution": f"{self.config.camera.width} × {self.config.camera.height}",
            "error": self._camera_error,
        }

    # ── Serial ────────────────────────────────────────────────────────

    def list_available_serial_ports(self) -> list[dict[str, Any]]:
        """Real COM/tty ports visible to the machine running the pipeline."""
        return list_serial_ports()

    def configure_serial(
        self,
        port: Optional[str] = None,
        baud: Optional[int] = None,
        mode: Optional[str] = None,
        connect: bool = True,
    ) -> tuple[bool, str]:
        """
        Switch the pipeline's serial link at runtime.

        The pipeline builds the new interface, connects it, and only then releases the
        previous one — so there is never a second serial owner and never a gap where two
        interfaces hold the same port.
        """
        with self._control_lock:
            if port is not None:
                self.config.serial.port = str(port)
            if baud is not None:
                try:
                    self.config.serial.baud_rate = int(baud)
                except (TypeError, ValueError):
                    return False, f"invalid baud rate: {baud!r}"

            if mode == "mock":
                self.config.serial.mock = True
                self.config.serial.enabled = True
            elif mode == "disabled":
                self.config.serial.enabled = False
                self.config.serial.mock = True
            elif mode == "real":
                self.config.serial.mock = False
                self.config.serial.enabled = True

            previous = self.serial
            if self.config.serial.mock:
                new_serial: SerialInterface = MockSerialInterface()
            else:
                new_serial = RealSerialInterface(config=self.config.serial)

            if self.config.serial.enabled and connect:
                if not new_serial.connect():
                    new_serial.disconnect()
                    self._last_error = f"could not open {self.config.serial.port}"
                    logger.error("Serial switch failed: %s", self._last_error)
                    return False, self._last_error

            # Swap, then release the old link.
            self.serial = new_serial
            self._frames_sent = 0
            # A brand-new link has transmitted nothing: keeping the previous port's frame here
            # would show a stale "Last TX" that this connection never sent.
            self._last_tx_frame = "NONE"
            self._last_error = None
            if previous is not None and previous is not new_serial:
                try:
                    previous.disconnect()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Error releasing previous serial interface: %s", exc)

            self._last_error = None
            logger.info(
                "Serial configured: mode=%s port=%s baud=%d",
                "mock" if self.config.serial.mock else "real",
                self.config.serial.port,
                self.config.serial.baud_rate,
            )
            return True, ""

    def disconnect_serial(self) -> tuple[bool, str]:
        """
        Close the pipeline's serial link — the counterpart of the UI Connect button.

        After this the pipeline reports *disconnected* and refuses to transmit. It must
        never leave a stand-in link in place claiming to be connected, because
        "Connected" in the UI has to mean the pipeline really holds a serial link.
        """
        with self._control_lock:
            if self.serial is not None and self.serial.is_connected:
                self.serial.disconnect()
            logger.info("Serial disconnected by UI request; transmissions are now refused.")
            return True, ""

    @property
    def serial_info(self) -> dict[str, Any]:
        """Truthful serial state for the UI."""
        state = getattr(self.serial, "state", None)
        mode = "disabled" if not self.config.serial.enabled else ("mock" if self.config.serial.mock else "real")
        return {
            "mode": mode,
            "port": self.config.serial.port,
            "baud": int(self.config.serial.baud_rate),
            "connected": bool(self.serial is not None and self.serial.is_connected),
            "state": state.name if state is not None else "DISCONNECTED",
            "frames_sent": int(self._frames_sent),
            "last_tx": getattr(self, "_last_tx_frame", "NONE"),
        }

    # ── Authoritative command dispatch ────────────────────────────────

    def dispatch_command(self, candidate: Any, source: str = "manual") -> tuple[bool, dict[str, Any], Optional[str]]:
        """
        Validate → encode → transmit one command through the pipeline's own serial link.

        This is the single authoritative path for manual buttons, voice commands and
        anything else that is not a camera detection. It never opens its own port and
        never bypasses the validator, so CALL/OK are accepted exactly like on the
        detection path.
        """
        with self._control_lock:
            result = self.validator.validate(candidate, pipeline_state=PipelineState.STABLE_CMD)

            # An explicit user request must not be silenced by the detector's duplicate
            # suppression — e.g. the operator presses CALL again after plugging the ESP32
            # back in. Clear the latch once and re-validate; the camera path keeps its
            # own de-duplication rules untouched.
            if result.was_deduplicated and source in self._EXPLICIT_COMMAND_SOURCES:
                self.validator.reset()
                result = self.validator.validate(candidate, pipeline_state=PipelineState.STABLE_CMD)

            self.last_val_result = result

            if not result.is_valid or result.command is None:
                self.transport.send_idle()
                reason = f"command rejected: {candidate!r}"
                logger.warning("[CONTROL] %s (source=%s)", reason, source)
                return False, {"command": None, "frame": None, "sent": False}, reason

            name = command_name(result.command)
            frame = _encoded_frame_for(result.command)
            # The two vocabularies use different names for the same gesture
            # (dataset GRAB == hardware GRIP). Telemetry must show what the ESP32 got,
            # so report the wire name, not the internal resolution name.
            wire_name = frame.split("|", 1)[1] if "|" in frame else name

            # Transport boundary records the emission (display/telemetry).
            self.transport.send_command(result.command)

            # The ONE serial owner transmits.
            sent = False
            if self.config.serial.enabled and self.serial is not None:
                try:
                    sent = bool(self.serial.send_command(result.command))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Unhandled serial transmission error: %s", exc)
                    sent = False

            if sent:
                self._last_tx_frame = frame
                self._frames_sent += 1
                self._manual_tx_count += 1
                self._last_error = None
                logger.info("[CONTROL TX] %s (source=%s)", frame, source)
            else:
                self._last_error = "serial link unavailable" if self.config.serial.enabled else "serial disabled"
                # Nothing reached the wire, so do not let the validator remember this as
                # an emitted command: the next explicit request must be able to retry it.
                self.validator.reset()
                logger.warning("[CONTROL] %s not transmitted (source=%s)", name, source)

            return (
                sent,
                {
                    "command": wire_name,
                    "frame": frame,
                    "sent": sent,
                    "source": source,
                    "mode": "mock" if self.config.serial.mock else "real",
                },
                None if sent else self._last_error,
            )

    # ── Voice ─────────────────────────────────────────────────────────

    def set_voice_state(self, state: str, engine: Optional[str] = None) -> None:
        """Record the UI voice-engine state so it is reported back from one place."""
        with self._control_lock:
            self._voice_state = str(state or "idle")
            if engine:
                self._voice_engine = str(engine)

    def handle_voice_transcript(self, text: str, engine: Optional[str] = None) -> tuple[bool, dict[str, Any], Optional[str]]:
        """
        Parse a voice transcript **inside the pipeline** and transmit any resulting command
        through the same authoritative path as camera and manual commands.
        """
        with self._control_lock:
            if engine:
                self._voice_engine = str(engine)
            self._voice_transcript = str(text or "")

            intent = parse_intent(text or "")
            payload: dict[str, Any] = {
                "transcript": self._voice_transcript,
                "intent": intent.kind,
                "gesture": intent.gesture,
                "reply": intent.reply,
            }

            if intent.kind == "status":
                payload["reply"] = self.status_summary()
                self._voice_command = None
                self._voice_reply = payload["reply"]
                return True, payload, None

            if not intent.is_command:
                self._voice_command = None
                self._voice_reply = intent.reply
                logger.info("[VOICE] no command for %r (intent=%s)", text, intent.kind)
                return True, payload, None

            ok_flag, data, error = self.dispatch_command(intent.gesture, source="voice")
            payload.update(data)
            self._voice_command = data.get("command")
            self._voice_reply = intent.reply
            return ok_flag, payload, error

    def status_summary(self) -> str:
        """Spoken status report derived from real backend state."""
        cam = self.camera_info
        ser = self.serial_info
        cam_txt = (
            f"Camera {cam['index']} is {'running' if cam['enabled'] and cam['opened'] else 'stopped'} at {cam['resolution']}."
        )
        vis = (
            f"Tracking {self.last_rec_result.label} at {self.current_fps:.0f} frames per second."
            if self.last_rec_result is not None
            else "No hand detected."
        )
        ser_txt = (
            f"Serial {ser['mode']} on {ser['port']} at {ser['baud']} baud is {'connected' if ser['connected'] else 'disconnected'}."
        )
        return f"{cam_txt} {vis} {ser_txt}"

    @property
    def voice_info(self) -> dict[str, Any]:
        return {
            "state": self._voice_state,
            "transcript": self._voice_transcript,
            "command": self._voice_command,
            "reply": self._voice_reply,
            "engine": self._voice_engine,
        }

    # ── Aggregate state for the UI ────────────────────────────────────

    def runtime_info(self) -> dict[str, Any]:
        """Everything the bridge adds on top of the immutable VisualizationState."""
        return {
            "camera_info": self.camera_info,
            "serial_info": self.serial_info,
            "voice_info": self.voice_info,
            "errors": [e for e in (self._camera_error, self._last_error) if e],
            "manual_tx_count": int(self._manual_tx_count),
        }

    def run_loop(self, max_frames: Optional[int] = None, callback: Optional[Callable] = None) -> None:
        """
        Execute real-time pipeline processing loop until stopped or max_frames reached.
        """
        if not self.initialize():
            logger.error("Cannot start run_loop: pipeline initialization failed.")
            return

        frames_processed = 0
        try:
            while self._is_running:
                frame_container, rendered, viz_state = self.process_frame()

                if callback is not None:
                    callback(frame_container, rendered, viz_state)

                frames_processed += 1
                if max_frames is not None and frames_processed >= max_frames:
                    logger.info("Max frames reached (%d). Exiting loop.", max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("Pipeline loop interrupted by user.")
        finally:
            self.shutdown()

    def stop(self) -> None:
        """Stop processing loop."""
        self._is_running = False

    def shutdown(self) -> None:
        """Cleanly release all hardware and component resources."""
        self._is_running = False
        self._is_initialized = False
        self._pipeline_state = PipelineState.SHUTDOWN

        # Close Camera
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.close()

        # Close Detector
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()

        # Close Serial
        if hasattr(self, "serial") and self.serial is not None:
            self.serial.disconnect()

        # Close TCP Bridge
        if hasattr(self, "tcp_bridge") and self.tcp_bridge is not None:
            self.tcp_bridge.close()

        logger.info("NeuroGrip Pipeline shutdown complete.")

    def format_diagnostic_line(self) -> str:
        """Format a compact diagnostic string for the most recent frame."""
        det_res = getattr(self, "last_detection_result", None)
        if det_res is None or det_res.is_empty:
            return "HAND=NONE RAW=NONE:0.00 SECOND=NONE:0.00 HYBRID=NO_HAND STABLE=UNKNOWN_STATE COMMAND=NONE"

        if det_res.is_ambiguous:
            return "HAND=AMBIGUOUS RAW=NONE:0.00 SECOND=NONE:0.00 HYBRID=AMBIGUOUS STABLE=UNKNOWN_STATE COMMAND=NONE"

        primary_hand = det_res.primary_hand
        hand_str = primary_hand.handedness.value if primary_hand else "UNKNOWN"

        rec_res = getattr(self, "last_rec_result", None)
        stab_res = getattr(self, "last_stab_result", None)
        val_res = getattr(self, "last_val_result", None)

        raw_cls, raw_prob = "NONE", 0.0
        second_cls, second_prob = "NONE", 0.0

        if rec_res and rec_res.all_scores:
            sorted_scores = sorted(rec_res.all_scores.items(), key=lambda item: item[1], reverse=True)
            if len(sorted_scores) > 0:
                raw_cls, raw_prob = sorted_scores[0]
            if len(sorted_scores) > 1:
                second_cls, second_prob = sorted_scores[1]

        hybrid_label = rec_res.label if rec_res else "UNKNOWN"
        hybrid_conf = rec_res.confidence if rec_res else 0.0

        stable_cmd = stab_res.stable_command if (stab_res and stab_res.stable_command) else "UNKNOWN"
        command_str = val_res.command.value if (val_res and val_res.command) else "NONE"

        return (
            f"HAND={hand_str} "
            f"RAW={raw_cls}:{raw_prob:.2f} "
            f"SECOND={second_cls}:{second_prob:.2f} "
            f"HYBRID={hybrid_label}:{hybrid_conf:.2f} "
            f"STABLE={stable_cmd} "
            f"COMMAND={command_str}"
        )

    def __enter__(self) -> "NeuroGripPipeline":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
