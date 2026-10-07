"""
neurogrip/gui/dashboard.py
───────────────────────────
Production Control Console & Desktop Dashboard for NeuroGrip.
High-contrast Black + White + Animated Sparkle visual identity.

Key Design Principles:
  - Presentation + Safe Operator Control ONLY (Single Source of Truth).
  - Consumes existing NeuroGripPipeline and VisualizationState snapshots.
  - Zero modification of ML architecture, feature extraction, 68-D vectors, or serial protocol.
  - 100% offline using built-in Tkinter and PIL (Pillow).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.commands.validator import PipelineState
from neurogrip.communication.base import SerialState
from neurogrip.config.settings import AppConfig
from neurogrip.gui.particles import SparkleParticleManager
from neurogrip.visualization.base import VisualizationState

logger = logging.getLogger(__name__)

# Dark High-Contrast Color Palette
COLOR_BG_DARK = "#050505"         # Pitch Black Background
COLOR_CARD_BG = "#121212"         # Card Background
COLOR_CARD_BORDER = "#252528"     # Subtle Card Border
COLOR_TEXT_WHITE = "#FFFFFF"      # Primary Text
COLOR_TEXT_MUTED = "#888890"      # Secondary Text
COLOR_TEXT_DIM = "#55555C"        # Dim Metadata
COLOR_ACCENT_GREEN = "#00FF66"    # Ready / Transmitted / Connected Accent
COLOR_ACCENT_RED = "#FF3333"      # STOP / Error Accent
COLOR_ACCENT_AMBER = "#FFBB00"    # Warning / Disarmed Accent
COLOR_ACCENT_BLUE = "#00DDFF"     # Informational Accent


class DashboardStateFormatter:
    """
    Pure state formatter mapping VisualizationState and NeuroGripPipeline diagnostics
    into formatted display strings and semantic indicator colors.
    Decoupled from Tkinter widget rendering for headless testing safety.
    """

    @staticmethod
    def format_recognition_data(state: Optional[VisualizationState]) -> Dict[str, Any]:
        if state is None:
            return {
                "gesture": "N/A",
                "confidence_pct": "0.0%",
                "confidence_val": 0.0,
                "route": "N/A",
                "hand_count": "0",
                "stability": "N/A",
                "raw_prediction": "N/A",
            }

        conf = float(state.confidence) if isinstance(state.confidence, (int, float)) else 0.0
        conf_pct = f"{conf * 100:.1f}%" if conf > 0.0 else "N/A"

        return {
            "gesture": str(state.command).upper(),
            "confidence_pct": conf_pct,
            "confidence_val": conf,
            "route": str(state.model_used),
            "hand_count": str(state.hand_count),
            "stability": str(state.stabilizer_state),
            "raw_prediction": str(state.raw_prediction),
        }

    @staticmethod
    def format_stop_diagnostic_data(state: Optional[VisualizationState]) -> Dict[str, str]:
        if state is None:
            return {
                "rule": "FALSE",
                "metrics": "N/A",
                "hagrid_raw": "N/A",
                "hagrid_conf": "0.0%",
                "taxonomy": "NO_COMMAND",
                "armed": "FALSE",
                "stabilized": "FALSE",
                "final": "NO_COMMAND",
                "tx": "NO",
            }
        return {
            "rule": "TRUE" if state.stop_rule_triggered else "FALSE",
            "metrics": state.stop_rule_metrics,
            "hagrid_raw": state.hagrid_raw,
            "hagrid_conf": f"{state.hagrid_conf * 100:.1f}%",
            "taxonomy": state.taxonomy_mapped,
            "armed": "TRUE" if state.is_armed else "FALSE",
            "stabilized": "TRUE" if state.is_stabilized else "FALSE",
            "final": state.final_command,
            "tx": "YES" if state.is_tx_permitted else "NO",
        }

    @staticmethod
    def format_command_protocol_data(
        state: Optional[VisualizationState],
        pipeline: Optional[NeuroGripPipeline],
    ) -> Dict[str, Any]:
        if state is None:
            return {
                "command": "NO_COMMAND",
                "wire_frame": "NONE",
                "tx_status": "SUPPRESSED",
                "tx_color": COLOR_TEXT_MUTED,
                "port_info": "DISABLED",
            }

        cmd_str = str(state.command).upper()
        last_tx = getattr(state, "last_tx", "NONE")
        if last_tx and last_tx != "NONE":
            wire_frame_str = last_tx if last_tx.endswith("\n") else f"{last_tx}\n"
        else:
            wire_frame_str = f"NG1|{cmd_str}\n" if cmd_str not in ("NO_COMMAND", "NO_HAND", "AMBIGUOUS", "UNKNOWN", "REST", "NONE") else "NONE"

        if cmd_str in ("NO_COMMAND", "NO_HAND", "AMBIGUOUS", "UNKNOWN", "REST", "NONE") or str(state.hand_count) == "0":
            return {
                "command": "NO_COMMAND",
                "wire_frame": wire_frame_str,
                "tx_status": "SUPPRESSED (NO_COMMAND)",
                "tx_color": COLOR_TEXT_MUTED,
                "port_info": getattr(state, "transport_status", "DISABLED"),
            }

        tx_status = "TRANSMITTED"
        tx_color = COLOR_ACCENT_GREEN

        return {
            "command": cmd_str,
            "wire_frame": wire_frame_str,
            "tx_status": tx_status,
            "tx_color": tx_color,
            "port_info": getattr(state, "transport_status", "DISABLED"),
        }

    @staticmethod
    def format_system_status(
        state: Optional[VisualizationState],
        pipeline: Optional[NeuroGripPipeline],
        config: AppConfig,
    ) -> Dict[str, Tuple[str, str]]:
        """Return dict of status items: key -> (label_text, color_hex)."""
        # Camera
        cam_ready = pipeline is not None and getattr(pipeline.camera, "is_opened", False)
        cam_status = ("● READY", COLOR_ACCENT_GREEN) if cam_ready else ("● CLOSED / OFFLINE", COLOR_ACCENT_RED)

        # ML Models
        ml_ready = pipeline is not None and getattr(pipeline, "recognizer", None) is not None
        ml_status = ("● LOADED (Extra Trees + ResNet18)", COLOR_ACCENT_GREEN) if ml_ready else ("● NOT LOADED", COLOR_ACCENT_RED)

        # Serial Transport
        if config.serial.enabled:
            if config.serial.mock:
                serial_status = ("● MOCK MODE", COLOR_ACCENT_BLUE)
            elif pipeline is not None and hasattr(pipeline, "serial") and pipeline.serial.is_connected:
                serial_status = (f"● CONNECTED ({config.serial.port})", COLOR_ACCENT_GREEN)
            else:
                serial_status = (f"● DISCONNECTED ({config.serial.port})", COLOR_ACCENT_RED)
        else:
            serial_status = ("● DISABLED", COLOR_TEXT_MUTED)

        # Safety Path
        safety_status = ("● NORMAL", COLOR_ACCENT_GREEN)

        # App Lifecycle
        app_running = pipeline is not None and pipeline.is_running
        app_status = ("● RUNNING", COLOR_ACCENT_GREEN) if app_running else ("● STOPPED", COLOR_TEXT_MUTED)

        return {
            "camera": cam_status,
            "ml_models": ml_status,
            "serial": serial_status,
            "safety": safety_status,
            "app": app_status,
        }


class NeuroGripDashboardApp:
    """
    Production Desktop GUI Control Console.
    Integrates non-blocking worker thread execution for NeuroGripPipeline
    with high-contrast dark theme Tkinter interface.
    """

    def __init__(self, config: Optional[AppConfig] = None, root: Optional[tk.Tk] = None) -> None:
        self.config = config or AppConfig.default()
        self._is_standalone_root = False

        if root is None:
            try:
                self.root = tk.Tk()
                self._is_standalone_root = True
            except Exception as e:
                logger.warning("Tkinter root initialization unavailable: %s", e)
                self.root = None  # type: ignore
        else:
            self.root = root

        # Pipeline & Thread Orchestration
        self.pipeline: Optional[NeuroGripPipeline] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._is_pipeline_running: bool = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.last_state: Optional[VisualizationState] = None
        self._tx_count: int = 0

        # Diagnostics & Performance Metrics
        self._last_gui_update_time: float = time.time()
        self.gui_fps: float = 0.0

        if self.root is not None:
            self._setup_window()
            self._build_ui()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_window(self) -> None:
        """Configure root Tk window settings."""
        self.root.title("NEUROGRIP — Multimodal Hand Control Console (Phase 3.5)")
        self.root.geometry("1400x820")
        self.root.minsize(1100, 700)
        self.root.configure(bg=COLOR_BG_DARK)

    def _build_ui(self) -> None:
        """Construct all GUI layout components."""
        # 1. Header Bar with Particle Sparkle Canvas
        self._build_header()

        # 2. Main Content Frame (Left: Video & Controls, Right: Telemetry Cards)
        self.main_container = tk.Frame(self.root, bg=COLOR_BG_DARK)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 2a. Left Column (Camera Feed + Action Bar)
        self.left_column = tk.Frame(self.main_container, bg=COLOR_BG_DARK)
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self._build_camera_section()
        self._build_control_action_bar()

        # 2b. Right Column (Telemetry & Cards)
        self.right_column = tk.Frame(self.main_container, bg=COLOR_BG_DARK, width=420)
        self.right_column.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_column.pack_propagate(False)

        self._build_recognition_card()
        self._build_command_protocol_card()
        self._build_stop_diagnostics_card()
        self._build_system_status_card()
        self._build_event_log_card()

        # 3. Footer Performance Telemetry Bar
        self._build_footer()

    def _build_header(self) -> None:
        """Build header bar with ambient sparkle particles."""
        self.header_frame = tk.Frame(self.root, bg="#0D0E12", height=60)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)
        self.header_frame.pack_propagate(False)

        # Sparkle Particle Canvas
        self.header_canvas = tk.Canvas(
            self.header_frame,
            bg="#0D0E12",
            highlightthickness=0,
            bd=0,
        )
        self.header_canvas.pack(fill=tk.BOTH, expand=True)

        self.particle_manager = SparkleParticleManager(
            canvas=self.header_canvas, max_particles=25, width=1400, height=60
        )
        self.header_canvas.bind(
            "<Configure>",
            lambda e: self.particle_manager.resize(e.width, e.height),
        )

        # Header Title Text
        self.lbl_title = tk.Label(
            self.header_canvas,
            text="NEUROGRIP",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_TEXT_WHITE,
            bg="#0D0E12",
        )
        self.header_canvas.create_window(20, 20, anchor="w", window=self.lbl_title)

        self.lbl_subtitle = tk.Label(
            self.header_canvas,
            text="MULTIMODAL HAND CONTROL CONSOLE  ·  PHASE 3.5 PRODUCTION GUI",
            font=("Consolas", 9),
            fg=COLOR_TEXT_MUTED,
            bg="#0D0E12",
        )
        self.header_canvas.create_window(20, 42, anchor="w", window=self.lbl_subtitle)

        # Top System Badge
        self.lbl_system_badge = tk.Label(
            self.header_canvas,
            text="● SYSTEM READY",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT_GREEN,
            bg="#0D0E12",
        )
        self.header_canvas.create_window(1360, 30, anchor="e", window=self.lbl_system_badge)

    def _build_camera_section(self) -> None:
        """Build high-resolution video preview canvas."""
        self.cam_frame_container = tk.Frame(
            self.left_column, bg=COLOR_CARD_BG, highlightbackground=COLOR_CARD_BORDER, highlightthickness=1
        )
        self.cam_frame_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.cam_canvas = tk.Canvas(
            self.cam_frame_container, bg="#080808", highlightthickness=0, bd=0
        )
        self.cam_canvas.pack(fill=tk.BOTH, expand=True)

        # Initial Offline Canvas Text
        self._show_offline_camera_message()

    def _show_offline_camera_message(self) -> None:
        self.cam_canvas.delete("all")
        w = self.cam_canvas.winfo_width() or 640
        h = self.cam_canvas.winfo_height() or 480
        self.cam_canvas.create_rectangle(0, 0, w, h, fill="#080808", outline="")
        self.cam_canvas.create_text(
            w // 2,
            h // 2 - 15,
            text="[ CAMERA OFFLINE / PIPELINE STOPPED ]",
            font=("Consolas", 12, "bold"),
            fill=COLOR_TEXT_MUTED,
        )
        self.cam_canvas.create_text(
            w // 2,
            h // 2 + 15,
            text="Click 'START SYSTEM' to launch pipeline.",
            font=("Segoe UI", 10),
            fill=COLOR_TEXT_DIM,
        )

    def _build_control_action_bar(self) -> None:
        """Build action buttons bar beneath camera preview."""
        self.action_bar = tk.Frame(self.left_column, bg=COLOR_BG_DARK)
        self.action_bar.pack(fill=tk.X, pady=(0, 5))

        btn_style = {
            "font": ("Segoe UI", 9, "bold"),
            "fg": COLOR_TEXT_WHITE,
            "bg": "#1A1A1E",
            "activebackground": "#2E2E35",
            "activeforeground": COLOR_TEXT_WHITE,
            "bd": 0,
            "padx": 12,
            "pady": 6,
            "relief": tk.FLAT,
        }

        self.btn_start = tk.Button(
            self.action_bar,
            text="▶ START SYSTEM",
            command=self.start_pipeline,
            **btn_style,
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_stop_sys = tk.Button(
            self.action_bar,
            text="⏹ STOP SYSTEM",
            command=self.stop_pipeline,
            **btn_style,
        )
        self.btn_stop_sys.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_reset_neutral = tk.Button(
            self.action_bar,
            text="↺ RESET / NEUTRAL",
            command=self.reset_neutral,
            **btn_style,
        )
        self.btn_reset_neutral.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_reset_stab = tk.Button(
            self.action_bar,
            text="↺ RESET STABILIZER",
            command=self.reset_stabilizer,
            **btn_style,
        )
        self.btn_reset_stab.pack(side=tk.LEFT, padx=(0, 6))

    def _create_card_frame(self, parent: tk.Widget, title: str) -> Tuple[tk.Frame, tk.Frame]:
        """Helper to build styled high-contrast dark card container."""
        card = tk.Frame(
            parent,
            bg=COLOR_CARD_BG,
            highlightbackground=COLOR_CARD_BORDER,
            highlightthickness=1,
            bd=0,
        )
        card.pack(fill=tk.X, pady=(0, 10))

        # Card Header Title
        hdr = tk.Frame(card, bg="#161618", height=26)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        lbl_hdr = tk.Label(
            hdr,
            text=f"  {title.upper()}",
            font=("Consolas", 8, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#161618",
            anchor="w",
        )
        lbl_hdr.pack(fill=tk.BOTH, expand=True)

        body = tk.Frame(card, bg=COLOR_CARD_BG, padx=12, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        return card, body

    def _build_recognition_card(self) -> None:
        _, body = self._create_card_frame(self.right_column, "Recognition Engine")

        # Gesture Display Label
        lbl_g_head = tk.Label(
            body, text="ACTIVE GESTURE", font=("Consolas", 7), fg=COLOR_TEXT_DIM, bg=COLOR_CARD_BG, anchor="w"
        )
        lbl_g_head.pack(fill=tk.X)

        self.lbl_gesture_val = tk.Label(
            body,
            text="NO_COMMAND",
            font=("Segoe UI", 18, "bold"),
            fg=COLOR_TEXT_WHITE,
            bg=COLOR_CARD_BG,
            anchor="w",
        )
        self.lbl_gesture_val.pack(fill=tk.X, pady=(0, 6))

        # Metadata grid
        grid = tk.Frame(body, bg=COLOR_CARD_BG)
        grid.pack(fill=tk.X)

        self.lbl_conf = self._add_grid_row(grid, "Confidence:", "0.0%", row=0)
        self.lbl_route = self._add_grid_row(grid, "Route:", "N/A", row=1)
        self.lbl_hands = self._add_grid_row(grid, "Hand Count:", "0", row=2)
        self.lbl_stability = self._add_grid_row(grid, "Stability:", "UNKNOWN_STATE", row=3)
        self.lbl_raw_pred = self._add_grid_row(grid, "Raw Prediction:", "N/A", row=4)

    def _add_grid_row(self, parent: tk.Frame, label_text: str, default_val: str, row: int) -> tk.Label:
        lbl_name = tk.Label(
            parent, text=label_text, font=("Segoe UI", 9), fg=COLOR_TEXT_MUTED, bg=COLOR_CARD_BG, anchor="w"
        )
        lbl_name.grid(row=row, column=0, sticky="w", pady=2)

        lbl_val = tk.Label(
            parent, text=default_val, font=("Consolas", 9, "bold"), fg=COLOR_TEXT_WHITE, bg=COLOR_CARD_BG, anchor="e"
        )
        lbl_val.grid(row=row, column=1, sticky="e", pady=2)

        parent.grid_columnconfigure(1, weight=1)
        return lbl_val

    def _build_command_protocol_card(self) -> None:
        _, body = self._create_card_frame(self.right_column, "Command & Protocol Boundary")

        grid_top = tk.Frame(body, bg=COLOR_CARD_BG)
        grid_top.pack(fill=tk.X)
        self.lbl_cmd_canonical = self._add_grid_row(grid_top, "Canonical Command:", "NO_COMMAND", row=0)

        # Encoded Wire Frame Box
        lbl_wf_head = tk.Label(
            body, text="WIRE PROTOCOL FRAME (NG1|<CMD>\\n)", font=("Consolas", 7), fg=COLOR_TEXT_DIM, bg=COLOR_CARD_BG, anchor="w"
        )
        lbl_wf_head.pack(fill=tk.X, pady=(6, 2))

        self.lbl_wire_frame = tk.Label(
            body,
            text="—",
            font=("Consolas", 10, "bold"),
            fg=COLOR_ACCENT_BLUE,
            bg="#0A0A0C",
            padx=8,
            pady=4,
            anchor="w",
            relief=tk.FLAT,
        )
        self.lbl_wire_frame.pack(fill=tk.X, pady=(0, 6))

        grid_bottom = tk.Frame(body, bg=COLOR_CARD_BG)
        grid_bottom.pack(fill=tk.X)
        self.lbl_tx_status = self._add_grid_row(grid_bottom, "Transmission Status:", "SUPPRESSED", row=0)

    def _build_stop_diagnostics_card(self) -> None:
        _, body = self._create_card_frame(self.right_column, "STOP Safety Diagnostics")

        grid = tk.Frame(body, bg=COLOR_CARD_BG)
        grid.pack(fill=tk.X)

        self.lbl_stop_rule = self._add_grid_row(grid, "Rule Triggered:", "FALSE", row=0)
        self.lbl_stop_metrics = self._add_grid_row(grid, "Rule Metrics:", "N/A", row=1)
        self.lbl_stop_hagrid_raw = self._add_grid_row(grid, "HaGRID Raw:", "N/A", row=2)
        self.lbl_stop_taxonomy = self._add_grid_row(grid, "Taxonomy Mapped:", "NO_COMMAND", row=3)
        self.lbl_stop_armed = self._add_grid_row(grid, "Software Armed:", "FALSE", row=4)
        self.lbl_stop_stabilized = self._add_grid_row(grid, "Stabilized:", "FALSE", row=5)
        self.lbl_stop_final = self._add_grid_row(grid, "Final Command:", "NO_COMMAND", row=6)
        self.lbl_stop_tx = self._add_grid_row(grid, "TX Permitted:", "NO", row=7)

    def _build_system_status_card(self) -> None:
        _, body = self._create_card_frame(self.right_column, "System Status")

        self.lbl_status_cam = self._add_grid_row(body, "Camera Interface:", "● CLOSED", row=0)
        self.lbl_status_ml = self._add_grid_row(body, "ML Models:", "● READY", row=1)
        self.lbl_status_serial = self._add_grid_row(body, "Serial Transport:", "● DISABLED", row=2)
        self.lbl_status_safety = self._add_grid_row(body, "Safety Path:", "● ARMED", row=3)

    def _build_event_log_card(self) -> None:
        _, body = self._create_card_frame(self.right_column, "Command & Safety Event Log")

        self.txt_log = tk.Text(
            body,
            height=6,
            bg="#08080A",
            fg=COLOR_TEXT_MUTED,
            font=("Consolas", 8),
            bd=0,
            highlightthickness=0,
            wrap=tk.WORD,
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        self.log_event("SYSTEM INITIALIZED — NEUROGRIP READY")

    def log_event(self, msg: str) -> None:
        """Append timestamped entry to event log."""
        if not hasattr(self, "txt_log") or self.txt_log is None:
            return
        t_str = time.strftime("%H:%M:%S")
        entry = f"[{t_str}] {msg}\n"

        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, entry)
        # Keep log size bounded (max 80 lines)
        lines = int(self.txt_log.index("end-1c").split(".")[0])
        if lines > 80:
            self.txt_log.delete("1.0", "2.0")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _build_footer(self) -> None:
        """Build telemetry status footer bar."""
        self.footer_frame = tk.Frame(self.root, bg="#0D0E12", height=28)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.footer_frame.pack_propagate(False)

        self.lbl_footer_telemetry = tk.Label(
            self.footer_frame,
            text="FPS: 0.0  |  LATENCY: 0.0 ms  |  HANDS: 0  |  ROUTE: N/A  |  TX COUNT: 0",
            font=("Consolas", 8),
            fg=COLOR_TEXT_MUTED,
            bg="#0D0E12",
            padx=15,
        )
        self.lbl_footer_telemetry.pack(side=tk.LEFT, fill=tk.Y)

    # ─────────────────────────────────────────────────────────
    # Pipeline Worker Thread & Lifecycle Controls
    # ─────────────────────────────────────────────────────────

    def start_pipeline(self) -> None:
        """Initialize and start background pipeline processing thread."""
        if self._is_pipeline_running:
            logger.info("Pipeline is already running.")
            return

        logger.info("Starting NeuroGrip Pipeline background thread...")
        self.pipeline = NeuroGripPipeline(config=self.config)

        if not self.pipeline.initialize():
            logger.error("Pipeline initialization failed.")
            self.log_event("ERROR: Pipeline initialization failed")
            return

        # Pre-arm software STOP safety path
        self.pipeline.arm_stop()

        self._is_pipeline_running = True
        self._worker_thread = threading.Thread(target=self._pipeline_worker_loop, daemon=True)
        self._worker_thread.start()

        self.log_event("PIPELINE STARTED — LIVE CAMERA ONLINE")
        self._schedule_gui_updates()

    def _pipeline_worker_loop(self) -> None:
        """Worker thread executing continuous frame processing."""
        while self._is_pipeline_running and self.pipeline and self.pipeline.is_running:
            try:
                container, rendered, viz_state = self.pipeline.process_frame()

                # Push latest frame & state to thread-safe queue (dropping stale frames)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put((rendered, viz_state))

                time.sleep(0.01)  # Throttle loop ~60 FPS max
            except Exception as e:
                logger.error("Error in pipeline worker thread: %s", e)
                break

    def stop_pipeline(self) -> None:
        """Stop background pipeline worker thread cleanly."""
        if not self._is_pipeline_running:
            return

        logger.info("Stopping NeuroGrip Pipeline worker thread...")
        self._is_pipeline_running = False

        if self.pipeline:
            self.pipeline.shutdown()
            self.pipeline = None

        self._show_offline_camera_message()
        self.log_event("PIPELINE STOPPED")

    def reset_neutral(self) -> None:
        """Send explicit physical NEUTRAL reset command over serial."""
        if self.pipeline:
            if hasattr(self.pipeline, "stabilizer") and self.pipeline.stabilizer:
                self.pipeline.stabilizer._last_emitted_command = "NEUTRAL"
            self.pipeline._last_tx_frame = "NG1|NEUTRAL"
            if hasattr(self.pipeline, "serial") and self.pipeline.serial:
                tx_ok = self.pipeline.serial.send_command("NEUTRAL")
                if tx_ok:
                    self.log_event("MANUAL COMMAND: TRANSMITTED NG1|NEUTRAL\n")

    def reset_stabilizer(self) -> None:
        """Reset temporal stabilizer memory."""
        if self.pipeline:
            self.pipeline.stabilizer.reset()
            self.log_event("STABILIZER RESET")

    # ─────────────────────────────────────────────────────────
    # GUI Render & Event Loop Updates (~30 FPS)
    # ─────────────────────────────────────────────────────────

    def _schedule_gui_updates(self) -> None:
        """Schedule main-thread Tkinter updates using after()."""
        if self.root is None or not self._is_pipeline_running:
            return

        self._update_gui()
        self.root.after(30, self._schedule_gui_updates)

    def _update_gui(self) -> None:
        """De-queue latest frame/state and refresh all UI components."""
        # 1. Update Sparkle Particles
        if hasattr(self, "particle_manager"):
            self.particle_manager.update()
            self.particle_manager.draw()

        # 2. Dequeue latest pipeline frame & state
        rendered_frame, state = None, None
        try:
            while not self._frame_queue.empty():
                rendered_frame, state = self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        if state is not None:
            self.last_state = state
            self._update_telemetry_ui(state)

        if rendered_frame is not None:
            self._render_camera_frame(rendered_frame)

    def _render_camera_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert BGR OpenCV image to PIL ImageTk and render on Canvas."""
        if bgr_frame is None or bgr_frame.size == 0 or self.cam_canvas is None:
            return

        canvas_w = max(100, self.cam_canvas.winfo_width())
        canvas_h = max(100, self.cam_canvas.winfo_height())

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Scale image keeping aspect ratio
        img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)

        # Keep reference to prevent GC
        self._photo_ref = photo

        self.cam_canvas.delete("all")
        self.cam_canvas.create_image(canvas_w // 2, canvas_h // 2, anchor="center", image=photo)

    def _update_telemetry_ui(self, state: VisualizationState) -> None:
        """Update telemetry text labels, colors, and badges."""
        # 1. Recognition Data
        rec_data = DashboardStateFormatter.format_recognition_data(state)
        self.lbl_gesture_val.config(text=rec_data["gesture"])
        self.lbl_conf.config(text=rec_data["confidence_pct"])
        self.lbl_route.config(text=rec_data["route"])
        self.lbl_hands.config(text=rec_data["hand_count"])
        self.lbl_stability.config(text=rec_data["stability"])
        self.lbl_raw_pred.config(text=rec_data["raw_prediction"])

        # 2. Command Protocol Data
        cmd_data = DashboardStateFormatter.format_command_protocol_data(state, self.pipeline)
        self.lbl_cmd_canonical.config(text=cmd_data["command"])
        self.lbl_wire_frame.config(text=cmd_data["wire_frame"])
        self.lbl_tx_status.config(text=cmd_data["tx_status"], fg=cmd_data["tx_color"])

        if cmd_data["tx_status"] == "TRANSMITTED":
            self._tx_count += 1

        # STOP Safety Diagnostics
        stop_diag = DashboardStateFormatter.format_stop_diagnostic_data(state)
        if hasattr(self, "lbl_stop_rule"):
            self.lbl_stop_rule.config(text=stop_diag["rule"])
            self.lbl_stop_metrics.config(text=stop_diag["metrics"])
            self.lbl_stop_hagrid_raw.config(text=f"{stop_diag['hagrid_raw']} ({stop_diag['hagrid_conf']})")
            self.lbl_stop_taxonomy.config(text=stop_diag["taxonomy"])
            self.lbl_stop_armed.config(text=stop_diag["armed"])
            self.lbl_stop_stabilized.config(text=stop_diag["stabilized"])
            self.lbl_stop_final.config(text=stop_diag["final"])
            self.lbl_stop_tx.config(text=stop_diag["tx"])

        # 3. System Status Badges
        sys_data = DashboardStateFormatter.format_system_status(state, self.pipeline, self.config)
        self.lbl_status_cam.config(text=sys_data["camera"][0], fg=sys_data["camera"][1])
        self.lbl_status_ml.config(text=sys_data["ml_models"][0], fg=sys_data["ml_models"][1])
        self.lbl_status_serial.config(text=sys_data["serial"][0], fg=sys_data["serial"][1])
        self.lbl_status_safety.config(text=sys_data["safety"][0], fg=sys_data["safety"][1])

        # Top System Badge
        self.lbl_system_badge.config(text="● SYSTEM READY", fg=COLOR_ACCENT_GREEN)

        # 4. Footer Telemetry Bar
        fps_val = float(state.fps) if isinstance(state.fps, (int, float)) else 0.0
        lat_val = float(state.latency_ms) if isinstance(state.latency_ms, (int, float)) else 0.0
        route_val = str(state.routing_str)

        self.lbl_footer_telemetry.config(
            text=f"FPS: {fps_val:.1f}  |  LATENCY: {lat_val:.1f} ms  |  HANDS: {state.hand_count}  |  ROUTE: {route_val}  |  TX COUNT: {self._tx_count}"
        )

    def on_close(self) -> None:
        """Clean window closure callback."""
        logger.info("Closing NeuroGrip Desktop GUI...")
        self.stop_pipeline()
        if self.root and self._is_standalone_root:
            try:
                self.root.destroy()
            except Exception:
                pass

    def run(self) -> None:
        """Launch Tkinter event loop."""
        if self.root is not None:
            self.start_pipeline()
            self.root.mainloop()
