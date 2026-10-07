#!/usr/bin/env python3
"""
Test script to verify the TCP bridge works correctly.
"""
import time
import json
import socket
from neurogrip.bridge import TCPIPBridge
from neurogrip.visualization.base import VisualizationState


def test_bridge():
    """Test the TCP bridge by sending a sample state."""
    print("Testing TCP bridge...")

    # Create bridge instance
    bridge = TCPIPBridge(host="127.0.0.1", port=8765)

    # Create a sample visualization state
    sample_state = VisualizationState(
        command="GRIP",
        confidence=0.95,
        fps=30.0,
        latency_ms=15.5,
        hand_count="1",
        pipeline_state="TRACKING",
        model_used="HYBRID",
        raw_prediction="GRIP",
        routing_str="HYBRID -> GRIP",
        transport_status="DISCONNECTED",
        last_tx="NONE",
        is_stop_active=False,
        is_ambiguous=False,
        message="Test message",
        safety_status="ARMED",
        handedness="RIGHT",
        landmarks=[],
        stop_rule_triggered=False,
        stop_rule_metrics="N/A",
        hagrid_raw="N/A",
        hagrid_conf=0.0,
        taxonomy_mapped="GRIP",
        is_armed=False,
        is_stabilized=False,
        final_command="GRIP",
        is_tx_permitted=False
    )

    print("Sending sample state...")
    bridge.send(sample_state)

    # Give it a moment to send
    time.sleep(0.1)

    # Clean up
    bridge.close()
    print("Test completed.")


if __name__ == "__main__":
    test_bridge()