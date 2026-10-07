"""Tests for the NeuroGrip application CLI."""
from __future__ import annotations

from pathlib import Path

from neurogrip.app import cli
from neurogrip.config.settings import AppConfig


def test_cli_parser_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.config is None
    assert args.no_gui is False


def test_cli_parser_config_and_no_gui() -> None:
    args = cli.build_parser().parse_args(["--config", "custom.yaml", "--no-gui"])
    assert args.config == Path("custom.yaml")
    assert args.no_gui is True


def test_cli_main_loads_config_and_runs(monkeypatch) -> None:
    config = AppConfig()
    configured = {}

    class FakePipeline:
        def __init__(self, config):
            configured["config"] = config
            self.stopped = False
            self.shutdown_calls = 0

        def initialize(self):
            return True

        def run_loop(self, callback=None):
            assert callback is not None

        def stop(self):
            self.stopped = True

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(cli.AppConfig, "default", classmethod(lambda cls: config))
    monkeypatch.setattr(cli, "NeuroGripPipeline", FakePipeline)
    monkeypatch.setattr(cli, "configure_logging", lambda _: None)

    assert cli.main(["--no-gui"]) == 0
    assert configured["config"].visualization.enabled is False


def test_cli_returns_nonzero_when_pipeline_initialization_fails(monkeypatch) -> None:
    config = AppConfig()
    calls = {"shutdown": 0}

    class FakePipeline:
        def __init__(self, config):
            pass

        def initialize(self):
            return False

        def shutdown(self):
            calls["shutdown"] += 1

    monkeypatch.setattr(cli.AppConfig, "default", classmethod(lambda cls: config))
    monkeypatch.setattr(cli, "NeuroGripPipeline", FakePipeline)
    monkeypatch.setattr(cli, "configure_logging", lambda _: None)

    assert cli.main([]) == 1
    assert calls["shutdown"] == 1
