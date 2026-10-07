"""
neurogrip/communication/serial_port.py
───────────────────────────────────────
Real PC Serial Interface / SerialTransport Implementation.
Uses PySerial to transmit formatted NG1 protocol frames over USB serial to the ESP32 microcontroller.
"""
from __future__ import annotations

import logging
from typing import Optional

import serial

from neurogrip.commands.definitions import CommandLike, NeuroGripCommand, command_name
from neurogrip.communication.base import SerialInterface, SerialState
from neurogrip.communication.protocol import ProtocolEncoder
from neurogrip.config.settings import AppConfig, SerialConfig

logger = logging.getLogger(__name__)


def list_serial_ports() -> list[dict[str, object]]:
    """
    Enumerate real serial ports visible to this machine.

    Called by the running pipeline (not by a throwaway probe process) so the port list the
    UI shows is the same one the pipeline would open.
    """
    ports: list[dict[str, object]] = []
    try:
        from serial.tools import list_ports as _list_ports

        for info in _list_ports.comports():
            ports.append(
                {
                    "device": info.device,
                    "label": info.description or info.device,
                    "hwid": info.hwid or "",
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not enumerate serial ports: %s", exc)
    return ports


class RealSerialInterface(SerialInterface):
    """
    Real USB serial transport wrapping PySerial.

    Parameters
    ----------
    config : Optional[SerialConfig]
        Serial configuration settings (port, baud_rate, timeout_s).
    """

    def __init__(self, config: Optional[SerialConfig] = None) -> None:
        if config is None:
            config = AppConfig.default().serial
        self.config = config

        self._port_instance: Optional[serial.Serial] = None
        self._state: SerialState = SerialState.DISCONNECTED

    @property
    def is_connected(self) -> bool:
        return (
            self._state == SerialState.CONNECTED
            and self._port_instance is not None
            and self._port_instance.is_open
        )

    @property
    def state(self) -> SerialState:
        return self._state

    def connect(self) -> bool:
        """
        Open the PySerial port using configured parameters.

        Returns
        -------
        bool
            True if connection established, False on failure.
        """
        if self.is_connected:
            return True

        logger.info(
            "Connecting to serial port '%s' at %d baud...",
            self.config.port,
            self.config.baud_rate,
        )

        try:
            self._port_instance = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baud_rate,
                timeout=self.config.timeout_s,
                write_timeout=getattr(self.config, "write_timeout", self.config.timeout_s),
            )
            self._state = SerialState.CONNECTED
            logger.info("Successfully connected to serial port '%s'.", self.config.port)
            return True
        except (serial.SerialException, OSError, IOError, Exception) as e:
            self._state = SerialState.ERROR
            self._port_instance = None
            logger.error("Failed to connect to serial port '%s': %s", self.config.port, e)
            return False

    def disconnect(self) -> None:
        """Close the PySerial port cleanly."""
        if self._port_instance is not None:
            try:
                if self._port_instance.is_open:
                    self._port_instance.close()
            except Exception as e:
                logger.warning("Error closing serial port: %s", e)
            finally:
                self._port_instance = None

        self._state = SerialState.DISCONNECTED
        logger.info("Serial port disconnected.")

    def send_command(self, command: CommandLike) -> bool:
        """
        Validate, format, and write a command frame over serial.

        Returns
        -------
        bool
            True if frame written successfully, False on failure.
        """
        if not self.is_connected or self._port_instance is None:
            logger.warning("Cannot send command: serial port is not connected.")
            return False

        cmd_str = command_name(command)
        frame_str = ProtocolEncoder.encode_command_string(cmd_str if cmd_str else None)

        if frame_str is None:
            logger.warning("Rejected invalid/NO_COMMAND candidate: '%s'.", command)
            return False

        try:
            data = frame_str.encode("ascii")
            self._port_instance.write(data)
            self._port_instance.flush()
            logger.debug("[SERIAL TX] %s", frame_str.strip())
            return True
        except (serial.SerialException, OSError, IOError, Exception) as e:
            self._state = SerialState.ERROR
            logger.error("Error writing to serial port '%s': %s", self.config.port, e)
            # Close port on error to prevent repeated uncontrolled writes
            try:
                if self._port_instance and self._port_instance.is_open:
                    self._port_instance.close()
            except Exception:
                pass
            self._port_instance = None
            return False

    def __enter__(self) -> "RealSerialInterface":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


# Alias for explicit SerialTransport requirement
SerialTransport = RealSerialInterface
