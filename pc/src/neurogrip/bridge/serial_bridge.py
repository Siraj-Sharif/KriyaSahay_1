"""
neurogrip/bridge/serial_bridge.py
-------------------------------
Small TCP bridge for serial control messages between Electron/React and the existing Python serial interface.
Uses existing RealSerialInterface; no second serial connection created.
"""
import json
import socket
from typing import Optional

from neurogrip.communication.serial_port import RealSerialInterface
from neurogrip.config.settings import AppConfig


class SerialTCPBridge:
    """TCP bridge specifically for serial control commands."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, serial_iface: Optional[RealSerialInterface] = None):
        self.host = host
        self.port = port
        self.serial = serial_iface or RealSerialInterface()
        self.socket: Optional[socket.socket] = None
        self._connect()

    def _connect(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((self.host, self.port))
            self.socket = sock
        except (ConnectionRefusedError, socket.timeout, OSError):
            self.socket = None

    def send_cmd(self, cmd: str) -> bool:
        """Send command through existing serial interface (NG1 protocol)."""
        return self.serial.send_command(cmd)

    def connect_serial(self) -> bool:
        return self.serial.connect()

    def disconnect_serial(self) -> None:
        self.serial.disconnect()

    def get_port(self) -> str:
        return self.serial.config.port

    def get_baud(self) -> int:
        return self.serial.config.baud_rate

    def is_connected(self) -> bool:
        return self.serial.is_connected

    def close(self) -> None:
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None