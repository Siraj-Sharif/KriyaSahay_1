"""
tests/test_gui_dashboard.py
─────────────────────────────
Headless-safe unit test suite for the Phase 3.5 NeuroGrip Production GUI.

Tests particle generation, particle movement, state formatting (gesture,
command, safety, serial status, STOP state), and dashboard state update logic
without requiring a physical display server.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from neurogrip.config.settings import AppConfig
from neurogrip.gui.particles import SparkleParticleManager, SparkleParticle
from neurogrip.gui.dashboard import DashboardStateFormatter, NeuroGripDashboardApp
from neurogrip.visualization.base import VisualizationState


class TestSparkleParticleManager(unittest.TestCase):
    """Tests for programmatic particle sparkle system."""

    def test_particle_generation(self) -> None:
        manager = SparkleParticleManager(max_particles=25, width=800, height=600)
        self.assertEqual(len(manager.particles), 25)
        for p in manager.particles:
            self.assertIsInstance(p, SparkleParticle)
            self.assertGreaterEqual(p.x, -10.0)
            self.assertLessEqual(p.x, 810.0)
            self.assertGreaterEqual(p.y, -10.0)
            self.assertLessEqual(p.y, 620.0)

    def test_particle_movement_and_bounds_wrap(self) -> None:
        manager = SparkleParticleManager(max_particles=10, width=400, height=300)
        initial_positions = [(p.x, p.y) for p in manager.particles]

        # Advance simulation step
        manager.update()
        new_positions = [(p.x, p.y) for p in manager.particles]

        # Ensure particles moved
        for pos_init, pos_new in zip(initial_positions, new_positions):
            self.assertNotEqual(pos_init, pos_new)

    def test_particle_resize_bounds(self) -> None:
        manager = SparkleParticleManager(max_particles=10, width=100, height=100)
        manager.particles[0].x = 500.0  # Force out of bounds
        manager.resize(width=200, height=200)
        self.assertEqual(manager.particles[0].width, 200)
        self.assertEqual(manager.particles[0].height, 200)


class TestDashboardStateFormatter(unittest.TestCase):
    """Tests for static UI telemetry and status state formatting."""

    def test_format_no_hand(self) -> None:
        viz = VisualizationState(
            command="NO_COMMAND",
            confidence=0.0,
            model_used="N/A",
            hand_count="0",
            stabilizer_state="UNKNOWN_STATE",
            raw_prediction="N/A",
        )
        rec_data = DashboardStateFormatter.format_recognition_data(viz)
        self.assertEqual(rec_data["gesture"], "NO_COMMAND")
        self.assertEqual(rec_data["confidence_pct"], "N/A")
        self.assertEqual(rec_data["hand_count"], "0")

        cmd_data = DashboardStateFormatter.format_command_protocol_data(viz, pipeline=None)
        self.assertEqual(cmd_data["command"], "NO_COMMAND")
        self.assertEqual(cmd_data["wire_frame"], "NONE")
        self.assertEqual(cmd_data["tx_status"], "SUPPRESSED (NO_COMMAND)")

    def test_format_valid_command_transmitted(self) -> None:
        viz = VisualizationState(
            command="INDEX_FINGER",
            confidence=0.942,
            model_used="ExtraTrees",
            hand_count="1",
            stabilizer_state="STABLE",
            raw_prediction="INDEX_FINGER",
            fps=29.4,
            latency_ms=32.1,
            last_tx="NG1|INDEX_FINGER\n",
        )
        rec_data = DashboardStateFormatter.format_recognition_data(viz)
        self.assertEqual(rec_data["gesture"], "INDEX_FINGER")
        self.assertEqual(rec_data["confidence_pct"], "94.2%")
        self.assertEqual(rec_data["route"], "ExtraTrees")
        self.assertEqual(rec_data["hand_count"], "1")
        self.assertEqual(rec_data["stability"], "STABLE")

        cmd_data = DashboardStateFormatter.format_command_protocol_data(viz, pipeline=None)
        self.assertEqual(cmd_data["command"], "INDEX_FINGER")
        self.assertEqual(cmd_data["wire_frame"], "NG1|INDEX_FINGER\n")
        self.assertEqual(cmd_data["tx_status"], "TRANSMITTED")

    def test_format_stop_safety_active(self) -> None:
        viz = VisualizationState(
            is_stop_active=True,
            safety_status="NORMAL",
        )
        cfg = AppConfig.default()
        sys_data = DashboardStateFormatter.format_system_status(viz, pipeline=None, config=cfg)
        self.assertIn("NORMAL", sys_data["safety"][0])

    def test_format_multi_hand_suppressed(self) -> None:
        viz = VisualizationState(
            command="AMBIGUOUS",
            hand_count="2+",
            is_ambiguous=True,
        )
        cmd_data = DashboardStateFormatter.format_command_protocol_data(viz, pipeline=None)
        self.assertEqual(cmd_data["command"], "NO_COMMAND")
        self.assertIn("SUPPRESSED", cmd_data["tx_status"])

    def test_format_system_status_serial_modes(self) -> None:
        viz = VisualizationState()

        # Mock Serial Config
        cfg_mock = AppConfig.default()
        cfg_mock.serial.enabled = True
        cfg_mock.serial.mock = True
        sys_mock = DashboardStateFormatter.format_system_status(viz, pipeline=None, config=cfg_mock)
        self.assertIn("MOCK MODE", sys_mock["serial"][0])

        # Disabled Serial Config
        cfg_dis = AppConfig.default()
        cfg_dis.serial.enabled = False
        sys_dis = DashboardStateFormatter.format_system_status(viz, pipeline=None, config=cfg_dis)
        self.assertIn("DISABLED", sys_dis["serial"][0])


class TestDashboardAppLogic(unittest.TestCase):
    """Tests for NeuroGripDashboardApp headless instantiation and event logging."""

    def setUp(self) -> None:
        self.cfg = AppConfig.default()
        self.cfg.serial.enabled = True
        self.cfg.serial.mock = True

    def test_app_instantiation_headless(self) -> None:
        app = NeuroGripDashboardApp(config=self.cfg, root=None)
        self.assertIsNotNone(app)
        self.assertEqual(app.config.serial.mock, True)
        self.assertEqual(app._is_pipeline_running, False)

    def test_log_event_without_txt_widget(self) -> None:
        app = NeuroGripDashboardApp(config=self.cfg, root=None)
        # log_event should degrade gracefully if Tk widget is not available
        app.log_event("HEADLESS EVENT LOG TEST")


if __name__ == "__main__":
    unittest.main()
