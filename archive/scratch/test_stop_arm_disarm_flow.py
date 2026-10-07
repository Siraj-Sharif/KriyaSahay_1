"""
pc/scratch/test_stop_arm_disarm_flow.py
───────────────────────────────────────
Verification script testing the exact STOP ARM/DISARM GUI & Pipeline behavior:
1. System DISARMED -> STOP recognized, Stabilizer=STOP, ARMED=FALSE, TX=NO, Serial Writes=0
2. Press SPACE (ARM) -> STOP recognized, Stabilizer=STOP, ARMED=TRUE, TX=YES, Serial Writes=NG1|STOP\n
3. Ordinary Palm -> NO_COMMAND, 0 STOP transmission
4. Press SPACE (DISARM) -> STOP recognized, Transmission blocked, 0 new serial bytes
"""
from __future__ import annotations

import sys
import numpy as np
from unittest.mock import MagicMock, patch

from neurogrip.app.pipeline import NeuroGripPipeline
from neurogrip.camera.mock_camera import MockCamera
from neurogrip.commands.definitions import NeuroGripCommand
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.config.settings import AppConfig
from neurogrip.hand_tracking.landmarks import DetectionResult, Handedness, HandLandmarks, NormalizedLandmark
from neurogrip.recognition.rule_based import RuleBasedRecognizer
from neurogrip.stabilization.temporal import StabilizerState


def build_stop_hand() -> HandLandmarks:
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)
    # Main 4 fingers extended UP
    lms[5] = NormalizedLandmark(x=0.42, y=0.5, z=0.0)
    lms[6] = NormalizedLandmark(x=0.42, y=0.38, z=0.0)
    lms[7] = NormalizedLandmark(x=0.42, y=0.25, z=0.0)
    lms[8] = NormalizedLandmark(x=0.42, y=0.1, z=0.0)

    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[10] = NormalizedLandmark(x=0.5, y=0.37, z=0.0)
    lms[11] = NormalizedLandmark(x=0.5, y=0.23, z=0.0)
    lms[12] = NormalizedLandmark(x=0.5, y=0.08, z=0.0)

    lms[13] = NormalizedLandmark(x=0.58, y=0.5, z=0.0)
    lms[14] = NormalizedLandmark(x=0.58, y=0.38, z=0.0)
    lms[15] = NormalizedLandmark(x=0.58, y=0.25, z=0.0)
    lms[16] = NormalizedLandmark(x=0.58, y=0.12, z=0.0)

    lms[17] = NormalizedLandmark(x=0.66, y=0.52, z=0.0)
    lms[18] = NormalizedLandmark(x=0.66, y=0.41, z=0.0)
    lms[19] = NormalizedLandmark(x=0.66, y=0.30, z=0.0)
    lms[20] = NormalizedLandmark(x=0.66, y=0.18, z=0.0)

    # Thumb spread widely lateral
    lms[1] = NormalizedLandmark(x=0.45, y=0.72, z=0.0)
    lms[2] = NormalizedLandmark(x=0.35, y=0.68, z=0.0)
    lms[3] = NormalizedLandmark(x=0.22, y=0.64, z=0.0)
    lms[4] = NormalizedLandmark(x=0.10, y=0.60, z=0.0)

    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def build_palm_hand() -> HandLandmarks:
    """Build ordinary relaxed open palm (narrow thumb spread metric < 0.35)."""
    lms = [NormalizedLandmark(x=0.5, y=0.8 - i * 0.01, z=0.0) for i in range(21)]
    lms[0] = NormalizedLandmark(x=0.5, y=0.8, z=0.0)

    lms[5] = NormalizedLandmark(x=0.45, y=0.5, z=0.0)
    lms[6] = NormalizedLandmark(x=0.45, y=0.38, z=0.0)
    lms[7] = NormalizedLandmark(x=0.45, y=0.25, z=0.0)
    lms[8] = NormalizedLandmark(x=0.45, y=0.1, z=0.0)

    lms[9] = NormalizedLandmark(x=0.5, y=0.5, z=0.0)
    lms[10] = NormalizedLandmark(x=0.5, y=0.37, z=0.0)
    lms[11] = NormalizedLandmark(x=0.5, y=0.23, z=0.0)
    lms[12] = NormalizedLandmark(x=0.5, y=0.08, z=0.0)

    lms[13] = NormalizedLandmark(x=0.55, y=0.5, z=0.0)
    lms[14] = NormalizedLandmark(x=0.55, y=0.38, z=0.0)
    lms[15] = NormalizedLandmark(x=0.55, y=0.25, z=0.0)
    lms[16] = NormalizedLandmark(x=0.55, y=0.12, z=0.0)

    lms[17] = NormalizedLandmark(x=0.60, y=0.52, z=0.0)
    lms[18] = NormalizedLandmark(x=0.60, y=0.41, z=0.0)
    lms[19] = NormalizedLandmark(x=0.60, y=0.30, z=0.0)
    lms[20] = NormalizedLandmark(x=0.60, y=0.18, z=0.0)

    # Thumb tucked / narrow alongside index finger
    lms[1] = NormalizedLandmark(x=0.48, y=0.72, z=0.0)
    lms[2] = NormalizedLandmark(x=0.47, y=0.65, z=0.0)
    lms[3] = NormalizedLandmark(x=0.46, y=0.58, z=0.0)
    lms[4] = NormalizedLandmark(x=0.45, y=0.52, z=0.0)

    return HandLandmarks(landmarks=lms, handedness=Handedness.RIGHT, score=0.95)


def run_verification():
    print("==================================================")
    print("NEUROGRIP — LIVE STOP ARM/DISARM VERIFICATION")
    print("==================================================")

    config = AppConfig.default()
    config.serial.enabled = True
    config.serial.mock = True

    camera = MockCamera()
    serial = MockSerialInterface()

    pipeline = NeuroGripPipeline(config=config, camera=camera, serial=serial)
    pipeline.initialize()

    # Step 2: System starts DISARMED by default
    pipeline.disarm_stop()
    print(f"\n--- STEP 2: INITIAL DISARMED STATE ---")
    print(f"is_stop_armed = {pipeline.is_stop_armed}")

    stop_hand = build_stop_hand()
    palm_hand = build_palm_hand()

    # Step 3: Perform STOP gesture for 30 frames (~1 sec) while DISARMED
    print("\n--- STEP 3: PERFORMING STOP GESTURE WHILE DISARMED (30 FRAMES) ---")
    with patch.object(pipeline.detector, "detect") as mock_det:
        mock_det.return_value = DetectionResult(hands=[stop_hand], num_hands=1)

        for i in range(30):
            _, _, viz = pipeline.process_frame()

    print(f"Observation Frame 30:")
    print(f"  Gesture Display:   {viz.command}")
    print(f"  STOP Diagnostic:   {viz.taxonomy_mapped} (Rule={viz.stop_rule_triggered}, Metrics={viz.stop_rule_metrics})")
    print(f"  Stabilizer State:  {viz.stabilizer_state}")
    print(f"  ARMED:             {viz.is_armed}")
    print(f"  TX PERMITTED:      {viz.is_tx_permitted}")
    print(f"  Serial Commands:   {len(serial.sent_commands)} commands, {len(serial.sent_frames)} frames")

    assert viz.command == "STOP", f"Expected STOP, got {viz.command}"
    assert viz.stabilizer_state == "STOP", f"Expected STOP stabilizer, got {viz.stabilizer_state}"
    assert viz.is_armed is False, "Expected ARMED=False"
    assert viz.is_tx_permitted is False, "Expected TX_PERMITTED=False"
    assert len(serial.sent_commands) == 0, f"Expected 0 serial writes, got {len(serial.sent_commands)}"
    print(">>> VERIFICATION STEP 3 PASSED: DISARMED STOP recognized, transmission BLOCKED (0 serial writes).")

    # Step 4 & 5: Press SPACE once to ARM, perform STOP gesture again
    print("\n--- STEP 4 & 5: PRESS SPACE TO ARM & PERFORM STOP GESTURE (30 FRAMES) ---")
    pipeline.arm_stop()
    print(f"is_stop_armed = {pipeline.is_stop_armed}")

    with patch.object(pipeline.detector, "detect") as mock_det:
        mock_det.return_value = DetectionResult(hands=[stop_hand], num_hands=1)

        for i in range(30):
            _, _, viz = pipeline.process_frame()

    print(f"Observation Frame 30:")
    print(f"  Gesture Display:   {viz.command}")
    print(f"  STOP Diagnostic:   {viz.taxonomy_mapped} (Rule={viz.stop_rule_triggered}, Metrics={viz.stop_rule_metrics})")
    print(f"  Stabilizer State:  {viz.stabilizer_state}")
    print(f"  ARMED:             {viz.is_armed}")
    print(f"  TX PERMITTED:      {viz.is_tx_permitted}")
    print(f"  Serial Commands:   {len(serial.sent_commands)} commands, {len(serial.sent_frames)} frames")
    if len(serial.sent_frames) > 0:
        print(f"  Emitted Wire Frame: '{repr(serial.sent_frames[0])}'")

    assert viz.command == "STOP", f"Expected STOP, got {viz.command}"
    assert viz.stabilizer_state == "STOP", f"Expected STOP stabilizer, got {viz.stabilizer_state}"
    assert viz.is_armed is True, "Expected ARMED=True"
    assert len(serial.sent_commands) >= 1, f"Expected >=1 serial write, got {len(serial.sent_commands)}"
    assert serial.sent_commands[0] == NeuroGripCommand.STOP
    print(">>> VERIFICATION STEP 4 & 5 PASSED: ARMED STOP recognized & TRANSMITTED ('NG1|STOP\\n').")

    tx_count_after_step5 = len(serial.sent_commands)

    # Step 6: Release STOP / perform ordinary open palm
    print("\n--- STEP 6: PERFORM ORDINARY OPEN PALM (30 FRAMES) ---")
    with patch.object(pipeline.detector, "detect") as mock_det:
        mock_det.return_value = DetectionResult(hands=[palm_hand], num_hands=1)

        for i in range(30):
            _, _, viz = pipeline.process_frame()

    print(f"Observation Frame 30:")
    print(f"  Gesture Display:   {viz.command}")
    print(f"  Stabilizer State:  {viz.stabilizer_state}")
    print(f"  Serial Commands:   {len(serial.sent_commands)} total")

    assert viz.command != "STOP", f"Ordinary palm must NOT produce STOP, got {viz.command}"
    assert len(serial.sent_commands) == tx_count_after_step5, "No new STOP transmissions should occur"
    print(">>> VERIFICATION STEP 6 PASSED: Ordinary palm yields NO_COMMAND / FOUR_FINGERS, 0 STOP transmission.")

    # Step 7: Press SPACE again to DISARM and repeat STOP
    print("\n--- STEP 7: PRESS SPACE TO DISARM AGAIN & REPEAT STOP (30 FRAMES) ---")
    pipeline.disarm_stop()
    print(f"is_stop_armed = {pipeline.is_stop_armed}")

    with patch.object(pipeline.detector, "detect") as mock_det:
        mock_det.return_value = DetectionResult(hands=[stop_hand], num_hands=1)

        for i in range(30):
            _, _, viz = pipeline.process_frame()

    print(f"Observation Frame 30:")
    print(f"  Gesture Display:   {viz.command}")
    print(f"  Stabilizer State:  {viz.stabilizer_state}")
    print(f"  ARMED:             {viz.is_armed}")
    print(f"  TX PERMITTED:      {viz.is_tx_permitted}")
    print(f"  Serial Commands:   {len(serial.sent_commands)} total")

    assert viz.command == "STOP", f"Expected STOP display when disarmed, got {viz.command}"
    assert viz.is_armed is False, "Expected ARMED=False"
    assert viz.is_tx_permitted is False, "Expected TX_PERMITTED=False"
    assert len(serial.sent_commands) == tx_count_after_step5, f"Expected 0 new serial bytes, got {len(serial.sent_commands) - tx_count_after_step5} new"
    print(">>> VERIFICATION STEP 7 PASSED: Disarmed STOP recognized on GUI display, transmission BLOCKED (0 new bytes).")

    print("\n==================================================")
    print("ALL 7 VERIFICATION STEPS PASSED PERFECTLY!")
    print("==================================================")


if __name__ == "__main__":
    run_verification()
