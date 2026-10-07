# NeuroGrip CV Phase 2 — HaGRID Full-Frame vs Hand-Crop Architecture Benchmark Report

**Date:** September 14, 2026  
**Environment:** Windows (x86_64), Python 3.11, PyTorch 2.14.0+cpu, Ultralytics 8.4.150, OpenCV 5.0.0  
**Project Root:** `D:\NeuroGrip_Project`  
**Status:** Benchmark Complete — Phase 2.1 Timing Validation Finished  

---

## 1. Objective & Scope

The objective of Phase 2 is to empirically compare two candidate PC-side computer vision architectures for NeuroGrip:

- **Architecture A (Full-Frame Baseline):** Evaluates the entire webcam frame directly through the official HaGRID ResNet18 classifier.
- **Architecture B (Hand-Crop Pipeline):** Evaluates an official YOLOv10n hand detector to locate hands, apply strict multi-hand safety rules, crop the active hand region with configurable padding, and classify via HaGRID ResNet18.

> [!IMPORTANT]
> - Phase 2 & 2.1 are strictly evaluation and benchmark validation phases.
> - Legacy Extra Trees / Random Forest code and serial communication modules were **not** modified or deleted.
> - ESP32 integration, Webots control, motor control, hardware command dispatch, and dataset collection/training were **not** performed.
> - Phase 3 will **not** start automatically.

---

## 2. Tested Architecture Descriptions

### Architecture A — Full-Frame Baseline
```
Webcam Frame (H x W x 3)
  ↓
Official HaGRID Preprocessing (Resize longest side to 224, pad to 224x224 with [144,144,144], normalize mean/std)
  ↓
HaGRID ResNet18 Classifier (34 classes)
  ↓
Raw HaGRID Class Prediction & Softmax Confidence
  ↓
Locked NeuroGrip 14-Command Taxonomy Mapper
  ↓
NeuroGrip Command Output
```

### Architecture B — Hand-Crop Pipeline
```
Webcam Frame (H x W x 3)
  ↓
Official HaGRID YOLOv10n Hand Detector (conf_threshold = 0.40)
  ↓
Hand Selection & Safety Validation Rules:
  ├─ 0 detected hands  -> Command = NO_COMMAND (reason: NO_HAND)
  ├─ 1 detected hand   -> Crop bounding box + 15% padding margin -> ResNet18 -> Taxonomy Mapper
  └─ >1 detected hands -> Command = NO_COMMAND (reason: MULTI_HAND_AMBIGUITY)
```

---

## 3. Verified Pretrained Models & Configuration

| Component | Checkpoint File | Checkpoint Path | Source | Size |
| :--- | :--- | :--- | :--- | :--- |
| **Gesture Classifier** | `ResNet18.pth` | `pc/models/hagrid/ResNet18.pth` | Official SberCloud HaGRIDv2 | 85.5 MB |
| **Hand Detector** | `YOLOv10n_hands.pt` | `pc/models/hagrid/YOLOv10n_hands.pt` | Official SberCloud HaGRIDv2 | 5.76 MB |

---

## 4. Phase 2.1 Timing Validation & Investigation

### Investigation of Initial 0.01 ms Classifier Timing
In the initial Phase 2 report draft, Architecture B classifier latency was reported as `0.01 ms`. A dedicated Phase 2.1 investigation was conducted to determine timing validity and root cause.

#### Root Cause Analysis
1. **Rule 1 Zero-Hand Bypass:** By design, Architecture B evaluates detector output count (`hand_count`). When `hand_count == 0`, Architecture B cleanly bypasses the ResNet18 classifier to save compute, returning `NO_COMMAND (reason: NO_HAND)` with `classifier_ms = 0.00 ms`.
2. **Synthetic Benchmark Test Frame:** The initial Phase 2 test pattern was a synthetic geometric shape where YOLOv10n detected 0 hands (`hand_count == 0`). Consequently, Architecture B executed the 0-hand bypass for all iterations, recording `0.00 ms` for classifier time.
3. **Corrected Measurement Methodology:** When benchmarking 1-hand steady-state execution (where a hand bounding box IS detected), ResNet18 IS executed on the crop, taking **30.67 ms** for inference (comparable to Architecture A full-frame inference latency of **26.37 ms**).

---

## 5. Corrected Phase 2.1 Empirical Benchmark Measurements

Measurements captured on local machine (CPU Execution Mode, 50 timed iterations per architecture using high-resolution `time.perf_counter()` monotonic timer and 5 warmup iterations):

### Component-Level Latency Breakdown

| Component / Pipeline Stage | Architecture A (Full-Frame) | Architecture B (Hand-Crop 1-Hand) | Architecture B (0-Hand Bypass) |
| :--- | :---: | :---: | :---: |
| **YOLOv10n Hand Detector** | N/A | **53.63 ms** | **68.35 ms** |
| **Crop Box Calculation** | N/A | **0.01 ms** | N/A |
| **Crop Preprocessing** | 1.31 ms | **1.32 ms** | N/A |
| **ResNet18 Classifier Inference** | **26.37 ms** | **30.67 ms** | **0.00 ms (Bypassed)** |
| **Postprocessing & Mapping** | <0.01 ms | <0.01 ms | <0.01 ms |
| **Total Mean Pipeline Latency** | **28.94 ms** | **85.63 ms** | **68.37 ms** |
| **Median Total Latency** | **28.65 ms** | **83.26 ms** | **67.80 ms** |
| **95th Percentile (P95) Latency** | **34.82 ms** | **103.25 ms** | **78.10 ms** |
| **Min / Max Latency** | 20.58 / 40.32 ms | 67.03 / 127.43 ms | 52.64 / 85.10 ms |
| **Standard Deviation** | **4.31 ms** | **12.30 ms** | 8.50 ms |
| **Theoretical Max FPS (CPU)** | **34.6 FPS** | **11.7 FPS** | **14.6 FPS** |

#### Mathematical Sum Verification (Arch B 1-Hand):
$$\text{Total} = \text{Detector}(53.63) + \text{CropCalc}(0.01) + \text{CropPrep}(1.32) + \text{Classifier}(30.67) + \text{Post}(0.00) = \mathbf{85.63\text{ ms}}$$
*(Total measured latency matches mathematical component sum with 0.00 ms residual).*

---

## 6. Hand Selection Rules & Safety Behavior

### Safety Rule 1: Zero Detected Hands
- **Architecture A:** Classifies full background image, which can produce low-confidence arbitrary predictions on background features.
- **Architecture B:** Returns `NO_COMMAND` with reason `NO_HAND`. ResNet18 classifier is bypassed completely, saving compute and preventing false positives.

### Safety Rule 2: Multiple Detected Hands (>1 Hands)
- **Architecture A:** Classifies full frame without multi-hand awareness.
- **Architecture B:** Detects all hand bounding boxes, renders all boxes in orange/red in diagnostic GUI, and returns `NO_COMMAND` with reason `MULTI_HAND_AMBIGUITY`. Bypasses classification to prevent executing commands from unintended secondary hands.

---

## 7. Comparative Analysis Matrix

| Evaluation Dimension | Architecture A (Full-Frame) | Architecture B (Hand-Crop) | Performance Advantage |
| :--- | :--- | :--- | :--- |
| **Processing Speed & Latency** | **28.94 ms (34.6 FPS)** | **85.63 ms (11.7 FPS)** | **Architecture A** (+56.69 ms faster) |
| **Scale & Position Invariance** | Moderate (sensitive to hand size/location) | High (normalizes cropped hand to 224x224) | **Architecture B** |
| **Background Noise Immunity** | Low (evaluates entire frame) | High (isolates active hand region) | **Architecture B** |
| **Multi-Hand Safety Rejection** | None (classifies whole frame) | **Strict (`NO_COMMAND` on >1 hands)** | **Architecture B** |
| **Zero-Hand Safety Bypass** | None (runs inference on background) | **Strict (`NO_COMMAND` on 0 hands)** | **Architecture B** |

---

## 8. Summary of Phase 2 / 2.1 Created Files

- [`pc/src/neurogrip/hagrid/detector.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/detector.py)
- [`pc/src/neurogrip/hagrid/benchmark.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/benchmark.py)
- [`pc/src/neurogrip/hagrid/config.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/config.py)
- [`pc/src/neurogrip/hagrid/__init__.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/__init__.py)
- [`pc/tools/hagrid_architecture_benchmark.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_architecture_benchmark.py)
- [`pc/tools/hagrid_hand_crop_diagnostic.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_hand_crop_diagnostic.py)
- [`pc/tests/test_hagrid_benchmark.py`](file:///d:/NeuroGrip_Project/pc/tests/test_hagrid_benchmark.py)
- [`pc/reports/hagrid_phase2_architecture_benchmark.md`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_phase2_architecture_benchmark.md)
- Checkpoints:
  - [`pc/models/hagrid/ResNet18.pth`](file:///d:/NeuroGrip_Project/pc/models/hagrid/ResNet18.pth) (85.5 MB)
  - [`pc/models/hagrid/YOLOv10n_hands.pt`](file:///d:/NeuroGrip_Project/pc/models/hagrid/YOLOv10n_hands.pt) (5.76 MB)
