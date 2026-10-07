# NeuroGrip CV Phase 2.2 — Real-World Recognition Evaluation Report

**Date:** September 14, 2026  
**Environment:** Windows (x86_64), Python 3.11, PyTorch 2.14.0+cpu, Ultralytics 8.4.150, OpenCV 5.0.0  
**Project Root:** `D:\NeuroGrip_Project`  
**Status:** Evaluation Tool & Pipeline Implemented — Human Operator Evaluation Pending  

---

## 1. Executive Summary & Objective

Phase 2.2 evaluates the real-world recognition performance, position/distance robustness, and frame-to-frame stability of **Architecture A (Full-Frame Baseline)** vs **Architecture B (Hand-Crop Pipeline)** on live USB webcam gesture feeds.

The objective is to determine whether Hand-Crop's recognition and robustness advantages justify its higher CPU latency cost (+56.69 ms added latency).

> [!IMPORTANT]
> - Phase 2.2 is strictly an evaluation phase.
> - Legacy Extra Trees / Random Forest code and serial communication modules were **not** modified or deleted.
> - ESP32 hardware integration, Webots control, motor control, hardware command dispatch, and dataset collection/training were **not** performed.
> - Phase 3 will **not** start automatically.

---

## 2. Verified Phase 2.1 Baseline Controlled Latency Figures

Verified empirical baseline metrics from Phase 2.1 controlled benchmarks (CPU Execution Mode):

### Architecture A (Full-Frame Baseline)
- **Mean Total Latency:** **28.94 ms**
- **Median Latency:** **28.65 ms**
- **P95 Latency:** **34.82 ms**
- **ResNet18 Inference Latency:** **26.37 ms**
- **Theoretical Max FPS:** **34.6 FPS**

### Architecture B (Hand-Crop Pipeline — 1-Hand Steady State)
- **Mean Total Latency:** **85.63 ms**
- **Median Latency:** **83.26 ms**
- **P95 Latency:** **103.25 ms**
- **YOLOv10n Detector Latency:** **53.63 ms**
- **ResNet18 Crop Inference Latency:** **30.67 ms**
- **Theoretical Max FPS:** **11.7 FPS**

---

## 3. Evaluation Tool & Real-Time HUD (`hagrid_phase2_realworld_evaluation.py`)

An interactive side-by-side USB webcam diagnostic tool was constructed under `pc/tools/hagrid_phase2_realworld_evaluation.py`.

### Visual HUD Overlay Features
- **Architecture A Display:** Canonical command, Raw HaGRID class, Softmax confidence %, Inference latency (ms), Rolling FPS.
- **Architecture B Display:** Hand count, Bounding box rendering, Canonical command, Raw HaGRID class, Softmax confidence %, Detector latency, Classifier latency, Total latency, Rolling FPS, Safety rejection reason.
- **Diagnostic Metrics:** Real-time A/B agreement status (`AGREE` / `DISAGREE`), Frame number, Expected gesture label, Trial elapsed time banner.

### 14-Gesture On-Screen Key Mapping
The evaluator selects the expected gesture using dedicated keys:
- `[1]`: CALL | `[2]`: CLOSED_FIST | `[3]`: FOUR_FINGERS | `[4]`: GRABBING | `[5]`: GRIP
- `[6]`: THUMBS_UP | `[7]`: PINKY | `[8]`: MIDDLE_FINGER | `[9]`: OK | `[0]`: INDEX_FINGER
- `[-]`: INDEX_PINKY | `[=]`: STOP | `[`: TWO_FINGERS | `]`: THREE_FINGERS

### 5-Second Trial Logger
Pressing `SPACE` triggers a 5-second evaluation trial (~150 frames) for the selected expected gesture, recording:
- `manual_recognition_consistency` = `(matching_frames / total_observed_frames) * 100`
- Wrong command occurrences & `NO_COMMAND` occurrences
- Frame-to-frame prediction switch count
- A/B diagnostic agreement rate %
- Structured JSON output saved to `pc/reports/realworld_trial_logs.json`

---

## 4. Safety Rule Verification

1. **Zero-Hand Safety Rule (`0 hands`):**  
   Architecture B returns `NO_COMMAND` with reason `NO_HAND`. ResNet18 classifier execution is bypassed completely.
2. **Multi-Hand Safety Rule (`>1 hands`):**  
   Architecture B detects all hand bounding boxes, renders all boxes in orange/red in the GUI, and returns `NO_COMMAND` with reason `MULTI_HAND_AMBIGUITY`. Bypasses classification to prevent executing commands from unintended secondary hands.
3. **Class `one` Explicit Removal:**  
   HaGRID class `one` (index 19) is explicitly mapped to `NO_COMMAND`.

---

## 5. Difficult Gesture-Pair Protocol

The trial evaluator is instructed to record specific observations for:
1. `GRIP` vs `GRABBING`
2. `CLOSED_FIST` vs `GRIP`
3. `FOUR_FINGERS` vs `THREE_FINGERS`
4. `INDEX_FINGER` vs `INDEX_PINKY`
5. `TWO_FINGERS` vs `THREE_FINGERS`
6. `STOP` vs open-hand gestures

---

## 6. Architecture Recommendation

### Recommendation: **`INCONCLUSIVE`**

#### Rationale:
Phase 2.1 controlled benchmarks establish that **Architecture A (Full-Frame)** is significantly faster on CPU (28.94 ms / 34.6 FPS vs 85.63 ms / 11.7 FPS). However, the real-world recognition evaluation tool (`hagrid_phase2_realworld_evaluation.py`) has been implemented and smoke-tested, but **live manual human webcam trials across all 14 gestures, distance variations, and lighting conditions are currently pending**.

To reach a final architectural determination, human operator evaluation trials using `python pc/tools/hagrid_phase2_realworld_evaluation.py` must be performed to compare manual recognition consistency % and position/distance robustness against Architecture B's +56.69 ms latency penalty.

---

## 7. Created & Modified Files

- [`pc/tools/hagrid_phase2_realworld_evaluation.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_phase2_realworld_evaluation.py) *(NEW)*
- [`pc/tests/test_hagrid_realworld_eval.py`](file:///d:/NeuroGrip_Project/pc/tests/test_hagrid_realworld_eval.py) *(NEW)*
- [`pc/reports/hagrid_phase2_realworld_evaluation.md`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_phase2_realworld_evaluation.md) *(NEW)*
- Checkpoints:
  - [`pc/models/hagrid/ResNet18.pth`](file:///d:/NeuroGrip_Project/pc/models/hagrid/ResNet18.pth) (85.5 MB)
  - [`pc/models/hagrid/YOLOv10n_hands.pt`](file:///d:/NeuroGrip_Project/pc/models/hagrid/YOLOv10n_hands.pt) (5.76 MB)
