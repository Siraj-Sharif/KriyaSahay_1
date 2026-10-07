"""
neurogrip/bridge
----------------
Real-time bridge modules for external frontends (Electron/React).
"""
from neurogrip.bridge.tcp_bridge import TCPIPBridge
from neurogrip.bridge.serial_bridge import SerialTCPBridge

__all__ = ["TCPIPBridge", "SerialTCPBridge"]