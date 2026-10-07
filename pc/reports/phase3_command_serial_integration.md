# Phase 3 — NeuroGrip Production Command & Serial Transport Integration Report

## 1. Executive Summary

Phase 3 — PC-Side Production Command Protocol and USB Serial Transport Integration is complete. The system bridges the verified Phase 2.4 hybrid Computer Vision recognition engine to a dedicated, hardware-decoupled production command protocol and serial transport abstraction layer.

All hardware-free and hardware-connected communication requirements have been verified, with **299/299 (100% GREEN)** tests passing across the repository test suite.

---

## 2. Architecture & Pipeline Dataflow

The end-to-end production architecture is:

```
USB Webcam (OpenCV)
    ↓
MediaPipe Hand Landmark Detection (1 hand vs 2+ hands vs 0 hands)
    ↓
68-D Feature Extraction
    ↓
Landmark Routing Gate
    ├── [INDEX_FINGER, TWO_FINGERS, PINKY] → Extra Trees ML Model
    └── [Other gestures] → HaGRID ResNet18 Model
    ↓
Taxonomy Mapping & Verification (Locked 14 Canonical Commands)
    ↓
Temporal Stabilization (TemporalStabilizer)
    ↓
STOP Safety Authorization (Safety-Gated Geometry & Software Arming)
    ↓
Canonical Command Decision
    ↓
Protocol Encoder (ProtocolEncoder: NG1|<CMD>\n)
    ↓
Serial Transport Abstraction (SerialTransport / RealSerialInterface / MockSerialTransport)
    ↓
USB Serial (PySerial @ 115200 Baud)
```

---

## 3. Locked Wire Protocol Specification

- **Protocol Format:** `NG1|<CMD>\n`
- **Prefix:** `NG1`
- **Separator:** `|`
- **Terminator:** Newline (`\n` / ASCII 0x0A)
- **Encoding:** Deterministic UTF-8 / ASCII bytes
- **Max Frame Size:** Bound strictly to 32 bytes

### Canonical Command Frames (14 Allowed Gestures)

| Command Name | Encoded Wire Frame (`bytes`) |
| :--- | :--- |
| `CALL` | `b"NG1|CALL\n"` |
| `CLOSED_FIST` | `b"NG1|CLOSED_FIST\n"` |
| `FOUR_FINGERS` | `b"NG1|FOUR_FINGERS\n"` |
| `GRABBING` | `b"NG1|GRABBING\n"` |
| `GRIP` | `b"NG1|GRIP\n"` |
| `THUMBS_UP` | `b"NG1|THUMBS_UP\n"` |
| `PINKY` | `b"NG1|PINKY\n"` |
| `MIDDLE_FINGER` | `b"NG1|MIDDLE_FINGER\n"` |
| `OK` | `b"NG1|OK\n"` |
| `INDEX_FINGER` | `b"NG1|INDEX_FINGER\n"` |
| `INDEX_PINKY` | `b"NG1|INDEX_PINKY\n"` |
| `STOP` | `b"NG1|STOP\n"` |
| `TWO_FINGERS` | `b"NG1|TWO_FINGERS\n"` |
| `THREE_FINGERS` | `b"NG1|THREE_FINGERS\n"` |

### Non-Transmitted States (ZERO Serial Writes)
- `NO_COMMAND` → **NO WRITE**
- `UNKNOWN` → **NO WRITE**
- `INVALID` → **NO WRITE**
- `one`, `peace`, `peace_inverted` → **NO WRITE**
- `REST` → **NO WRITE**

---

## 4. Safety Invariants & Rules

1. **NO_COMMAND Suppression:** `NO_COMMAND` is never transmitted over serial or transport.
2. **0 Hands Safety:** Detecting 0 hands forces `NO_COMMAND` and resets the temporal stabilizer, producing ZERO serial writes.
3. **Multi-Hand Safety:** Detecting 2+ hands triggers `AMBIGUOUS` state and resets the temporal stabilizer, producing ZERO serial writes.
4. **STOP Safety Gate:** The `STOP` safety command reaches transport ONLY when authorized by Phase 2.4 safety logic (exactly 1 hand, valid STOP geometry, software armed, temporal stabilization requirements satisfied). Raw predictions or disarmed `STOP` yield ZERO serial writes.
5. **De-duplication & Emissions:** Commands are emitted only when the production stabilizer signals a valid transition/emission event. Repeated identical predictions during stable hold do not generate duplicate redundant serial frames.

---

## 5. Serial Failure & Error Handling

- **Resilience:** Serial exceptions (`serial.SerialException`, `OSError`, `IOError`, timeouts, device disconnects) are caught cleanly inside `RealSerialInterface` / `SerialTransport`.
- **CV Application Survival:** Serial failures **NEVER** crash the Computer Vision application loop.
- **State Management:** Upon write or connection failure, the transport marks its internal state as `SerialState.ERROR` or `DISCONNECTED` and closes the port handle, preventing repeated uncontrolled write attempts.
- **Visualization HUD:** The status HUD explicitly displays transport status (`Transport: DISCONNECTED (COM3)` or `Transport: CONNECTED (COM3)` or `Transport: MOCK`).

---

## 6. Files Added & Modified

### Files Added
- [`pc/src/neurogrip/communication/protocol.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/communication/protocol.py): Hardware-decoupled protocol encoder component (`ProtocolEncoder`, `encode_protocol_frame()`).
- [`pc/tests/test_command_transport_phase3.py`](file:///d:/NeuroGrip_Project/pc/tests/test_command_transport_phase3.py): Dedicated Phase 3 test suite covering protocol encoding, suppression, mock transport, STOP safety, multi-hand/no-hand safety, and transport failure.
- [`pc/reports/phase3_command_serial_integration.md`](file:///d:/NeuroGrip_Project/pc/reports/phase3_command_serial_integration.md): Phase 3 architecture, protocol, and verification report.

### Files Modified
- [`pc/src/neurogrip/commands/definitions.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/commands/definitions.py): Preserved V1 13-member `NeuroGripCommand` enum definition with alias resolution in `from_string`.
- [`pc/src/neurogrip/communication/base.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/communication/base.py): Updated `SerialInterface.format_frame` to delegate to `ProtocolEncoder` and added `send()` / `close()` alias methods.
- [`pc/src/neurogrip/communication/mock_serial.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/communication/mock_serial.py): Enhanced `MockSerialInterface` and exported `MockSerialTransport` alias.
- [`pc/src/neurogrip/communication/serial_port.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/communication/serial_port.py): Added error resilience, timeout configuration, and `SerialTransport` alias.
- [`pc/src/neurogrip/communication/transport.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/communication/transport.py): Integrated `ProtocolEncoder` into `CommandTransport` boundary.
- [`pc/src/neurogrip/visualization/base.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/visualization/base.py): Added `transport_status` and `last_tx` fields to `VisualizationState`.
- [`pc/src/neurogrip/visualization/renderer.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/visualization/renderer.py): Rendered real-time transport status and last transmission frame in HUD overlay.
- [`pc/src/neurogrip/app/pipeline.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/app/pipeline.py): Connected transport status tracking and safe exception handling.
- [`pc/src/neurogrip/app/cli.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/app/cli.py): Added `--serial-port`, `--baudrate`, `--serial`, and `--serial-disabled` CLI arguments.
- [`pc/tests/test_command_transport.py`](file:///d:/NeuroGrip_Project/pc/tests/test_command_transport.py): Updated protocol assertions to match canonical wire frame strings.
- [`pc/tests/test_serial_protocol.py`](file:///d:/NeuroGrip_Project/pc/tests/test_serial_protocol.py): Updated test cases for 14 canonical command wire formatting.
- [`pc/tests/test_pipeline_integration.py`](file:///d:/NeuroGrip_Project/pc/tests/test_pipeline_integration.py): Updated integration test wire frame expectation.

---

## 7. Verification & Test Results

### 1. Focused Command Protocol & Transport Test Suite (`test_command_transport_phase3.py`)
- **Passed:** 11 / 11 tests (100% PASS)

### 2. Boundary Layer Transport Suite (`test_command_transport.py`)
- **Passed:** 3 / 3 tests (100% PASS)

### 3. Full Repository Pytest Suite (`pc/tests`)
- **Command:** `.venv\Scripts\python.exe -m pytest pc/tests -q`
- **Result:** **299 passed, 0 failed, 0 skipped** (100% GREEN in 66.03s)

### 4. Headless CV Smoke Test
- **Command:** `.venv\Scripts\python.exe pc/scripts/run_live_cv.py --camera 0 --no-gui --max-frames 30`
- **Result:** 30 frames processed cleanly with `MockSerialInterface` in disarmed/armed modes.

---

## 8. Model & Data Integrity Confirmation

- **Extra Trees Model (`pc/models/NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl`):** Unmodified (11,519,452 bytes).
- **HaGRID ResNet18 Model (`pc/models/hagrid/ResNet18.pth`):** Unmodified (89,656,314 bytes).
- **MediaPipe Model (`pc/models/mediapipe/hand_landmarker.task`):** Unmodified (7,819,105 bytes).
- **68-D Feature Definition:** Unmodified.
- **Datasets:** Unmodified (0 retrained models, 0 weight changes).
