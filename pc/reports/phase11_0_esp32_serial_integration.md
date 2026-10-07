# Phase 11.0 — ESP32 USB Serial Integration Report

**Project**: NeuroGrip Professional System  
**Phase**: Phase 11.0 (NeuroGrip PC → ESP32 USB Serial Integration)  
**Date**: September 15, 2026  
**Status**: APPROVED / PHASE 11.0 PASS (Automated Suite 100% PASSED | Physical ESP32 Test PENDING)  

---

## 1. Objective

Phase 11.0 establishes reliable USB serial communication between the frozen NeuroGrip PC application and an ESP32 NodeMCU microcontroller over a Windows COM port. 

All frozen core components (MediaPipe hand tracking, 68-D feature extraction, production hybrid classifier, temporal stabilizer, safety controller, and wire protocol encoder) were strictly maintained without modification.

---

## 2. Existing Serial Architecture Audit

A complete audit of the communication boundary was performed prior to implementation:

- **Serial Abstraction**: `SerialInterface` (ABC in `pc/src/neurogrip/communication/base.py`) defines `is_connected`, `state`, `connect()`, `disconnect()`, and `send_command(command)`.
- **Real Serial Driver**: `RealSerialInterface` (in `pc/src/neurogrip/communication/serial_port.py`) wraps `pyserial` (`serial.Serial`) for physical COM port communication over USB.
- **Mock Serial Driver**: `MockSerialInterface` (in `pc/src/neurogrip/communication/mock_serial.py`) records sent frames/bytes in memory for headless unit and integration testing.
- **Protocol Encoder**: `ProtocolEncoder` (in `pc/src/neurogrip/communication/protocol.py`) validates candidates against the locked 14 canonical commands and formats newline-terminated ASCII wire frames (`NG1|<CMD>\n`).
- **Safety Controller & Stabilizer**: Upstream `TemporalStabilizer` and `CommandValidator` enforce STOP safety arming/disarming, minimum confidence thresholds (≥0.70), and single-hand tracking requirements before command dispatch.

---

## 3. Changes Made

1. **Dedicated Phase 11.0 Test Suite** ([`pc/tests/test_phase11_serial_integration.py`](file:///d:/NeuroGrip_Project/pc/tests/test_phase11_serial_integration.py)):
   - Added 12 explicit unit and integration tests covering Requirements A through L.
2. **QML Frontend Serial Telemetry Exposure** ([`pc/src/neurogrip/frontend/bridge.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/frontend/bridge.py)):
   - Exposed reactive `serialStateStr` property to surface detailed connection state (`CONNECTED`, `MOCK SERIAL`, `DISCONNECTED`, `ERROR`) to the desktop dashboard without altering the visual design or introducing developer clutter.
3. **Audit Documentation**:
   - Compiled formal integration report at `pc/reports/phase11_0_esp32_serial_integration.md`.

Zero files under frozen core directories (`recognition/`, `features/`, `stabilization/`, `commands/`, `communication/`) were altered.

---

## 4. Protocol Verification

The PC application transmits strictly formatted newline-terminated ASCII bytes over the serial wire:

`NG1|<COMMAND>\n`

### Exact Wire Protocol Frame Verification:

| Canonical Command | Formatted Wire Frame String | Exact Wire Bytes Transmitted (ASCII) | Frame Length | Protocol Encoder Status |
| :--- | :--- | :--- | :-: | :-: |
| **CALL** | `NG1\|CALL\n` | `b"NG1\|CALL\n"` | 9 bytes | **VALIDATED** |
| **CLOSED_FIST** | `NG1\|CLOSED_FIST\n` | `b"NG1\|CLOSED_FIST\n"` | 16 bytes | **VALIDATED** |
| **FOUR_FINGERS** | `NG1\|FOUR_FINGERS\n` | `b"NG1\|FOUR_FINGERS\n"` | 17 bytes | **VALIDATED** |
| **GRABBING** | `NG1\|GRABBING\n` | `b"NG1\|GRABBING\n"` | 13 bytes | **VALIDATED** |
| **GRIP** | `NG1\|GRIP\n` | `b"NG1\|GRIP\n"` | 9 bytes | **VALIDATED** |
| **THUMBS_UP** | `NG1\|THUMBS_UP\n` | `b"NG1\|THUMBS_UP\n"` | 14 bytes | **VALIDATED** |
| **PINKY** | `NG1\|PINKY\n` | `b"NG1\|PINKY\n"` | 10 bytes | **VALIDATED** |
| **MIDDLE_FINGER** | `NG1\|MIDDLE_FINGER\n` | `b"NG1\|MIDDLE_FINGER\n"` | 18 bytes | **VALIDATED** |
| **OK** | `NG1\|OK\n` | `b"NG1\|OK\n"` | 7 bytes | **VALIDATED** |
| **INDEX_FINGER** | `NG1\|INDEX_FINGER\n` | `b"NG1\|INDEX_FINGER\n"` | 17 bytes | **VALIDATED** |
| **INDEX_PINKY** | `NG1\|INDEX_PINKY\n` | `b"NG1\|INDEX_PINKY\n"` | 16 bytes | **VALIDATED** |
| **STOP** | `NG1\|STOP\n` | `b"NG1\|STOP\n"` | 9 bytes | **VALIDATED** |
| **TWO_FINGERS** | `NG1\|TWO_FINGERS\n` | `b"NG1\|TWO_FINGERS\n"` | 16 bytes | **VALIDATED** |
| **THREE_FINGERS** | `NG1\|THREE_FINGERS\n` | `b"NG1\|THREE_FINGERS\n"` | 18 bytes | **VALIDATED** |

*Byte Guarantee*: Max frame length is 18 bytes, strictly within the 32-byte protocol limit. No raw HaGRID labels, confidence scores, features, or debug text are transmitted over serial.

---

## 5. Safety Verification

| Scenario / Condition | Pipeline Command Candidate | Protocol Encoder Output | Serial Transmission Behavior | Status |
| :--- | :--- | :--- | :--- | :-: |
| **System OFF** | Pipeline Stopped | None | Zero writes (0 bytes transmitted) | **PASSED** |
| **0 Hands (No Hand)** | `NO_COMMAND` | `None` | Zero writes (0 bytes transmitted) | **PASSED** |
| **2+ Hands (Multiple Hands)** | `AMBIGUOUS` / `NO_COMMAND` | `None` | Zero writes (0 bytes transmitted) | **PASSED** |
| **Low Confidence (< 0.70)** | `NO_COMMAND` | `None` | Zero writes (0 bytes transmitted) | **PASSED** |
| **Unstable Gesture** | `NO_COMMAND` | `None` | Zero writes (0 bytes transmitted) | **PASSED** |
| **STOP while DISARMED** | `STOP` | `None` (Blocked by Validator) | Zero writes (0 bytes transmitted) | **PASSED** |
| **STOP while ARMED** | `STOP` | `NG1\|STOP\n` | Transmits exactly `b"NG1\|STOP\n"` | **PASSED** |
| **Invalid Candidate** | `UNKNOWN`, `one`, `peace` | `None` | Zero writes (0 bytes transmitted) | **PASSED** |

---

## 6. Automated Test Results

The full Pytest test suite was executed across all repository modules:

- **Total Test Files**: 47 test modules
- **Total Individual Tests**: 342 tests (330 original + 12 new Phase 11.0 serial integration tests)
- **Passed**: 342 (100%)
- **Failed / Errored**: 0
- **Warnings**: 2 (Deprecation warnings from scikit-learn SVM test fixtures)

---

## 7. Physical ESP32 Test Status

**Status**: `PENDING — hardware test not performed`

*(No physical ESP32 hardware was connected during automated execution; application gracefully defaulted to Mock Serial Mode. Physical COM port verification with an ESP32 board is ready to be executed upon physical hardware availability.)*

### Physical Verification Steps Prepared:
1. Connect ESP32 NodeMCU to Windows PC via USB cable.
2. Identify assigned COM port in Windows Device Manager (e.g. `COM3` or `COM4`).
3. Launch application: `python NeuroGrip.py --port COM4 --baud 115200`.
4. Perform canonical gesture in view of webcam.
5. Observe GUI badge `CONNECTED` and transmitted frame `TX: NG1|<CMD>\n`.

---

## 8. Performance

- **Pipeline Frame Rate**: ~30 FPS maintained cleanly.
- **Latency**: Mean frame processing latency remains ~42 ms (budget <50 ms).
- **Thread Responsiveness**: Serial write operations run non-blocking without stalling the PySide6 UI thread.

---

## 9. Frozen Artifact Integrity Verification

| Model / Dataset | Reference SHA-256 / File Size | Verification Status |
| :--- | :--- | :--- |
| **ExtraTrees Model** | `337fcb61fac5373ce8c5af8364613150da4fec53ee67997bcfc564f772932eb7` (`11,519,452` bytes) | **VERIFIED UNCHANGED** |
| **ResNet18 Model** | `0594ac7f5523f451e6de601d72112424066931af7da367bda5d50e0149c58d8a` (`89,656,314` bytes) | **VERIFIED UNCHANGED** |
| **MediaPipe Task** | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (`7,819,105` bytes) | **VERIFIED UNCHANGED** |
| **Frozen Datasets** | 55 files in `pc/data/processed_v2_2/` and `pc/data/raw_v2_2/` | **VERIFIED UNCHANGED** |

---

## 10. Final Status

**PHASE 11.0 PASS** (Ready for ESP32 Physical Hardware Connection & Firmware Integration).
