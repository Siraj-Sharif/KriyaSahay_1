# NeuroGrip CV Phase 1 — HaGRID Pretrained Model Verification Report

**Date:** September 14, 2026  
**Environment:** Windows (x86_64), Python 3.11, PyTorch 2.14.0+cpu, Torchvision 0.29.0+cpu, OpenCV 5.0.0  
**Project Root:** `D:\NeuroGrip_Project`  
**Status:** Verification Complete — All Criteria Verified & Passed  

---

## 1. Executive Summary & Objective

The objective of Phase 1 is to verify official pretrained **HaGRID / HaGRIDv2** hand gesture recognition models on the local PC system before making architecture decisions or integrating with serial hardware/ESP32.

> [!IMPORTANT]
> - Existing legacy Extra Trees / Random Forest models and serial protocol drivers were **not** modified or deleted.
> - Serial communication, ESP32 hardware transmission, and Webots control were **not** integrated.
> - No new dataset collection or model training was performed.

---

## 2. Model Metadata & Verification

| Property | Verified Value |
| :--- | :--- |
| **Official Model Name** | HaGRIDv2 ResNet18 Full-Frame Classifier |
| **Official Source Repository** | [`hukenovs/hagrid`](https://github.com/hukenovs/hagrid) (Alexander Kapitanov et al., SberAI) |
| **arXiv Publications** | arXiv:2206.08219 (v1), arXiv:2412.01508 (v2) |
| **Official License** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0 variant) |
| **Official Checkpoint Source** | `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth` |
| **Local Checkpoint Path** | `D:\NeuroGrip_Project\pc\models\hagrid\ResNet18.pth` (Size: 85.5 MB) |
| **Architecture** | `torchvision.models.resnet18(num_classes=34)` |
| **State Dict Key** | `snapshot["MODEL_STATE"]` |
| **Model Load Status** | **YES** (Successfully loaded state dict into PyTorch module) |
| **Input Resolution** | 224 x 224 RGB image tensor |
| **Output Shape** | Tensor `(batch_size, 34)` containing raw class logits |

---

## 3. Preprocessing & Input Pipeline Verification

Verified directly from official HaGRID configuration (`configs/ResNet18.yaml`):

1. **Longest Side Resize**: Resizes longest image side to 224 pixels (`LongestMaxSize(max_size=224)`).
2. **Padding**: Equal padding to 224 x 224 using fill color `[144, 144, 144]` (`PadIfNeeded`).
3. **Normalization**:
   - Rescale pixel range to `[0.0, 1.0]`
   - `mean`: `[0.54, 0.499, 0.474]`
   - `std`: `[0.234, 0.235, 0.231]`
4. **Channel Order**: RGB order (converted from OpenCV BGR default).

---

## 4. Class Taxonomy & NeuroGrip Mapping Audit

### Complete Official HaGRID 34-Class Ordering
Verified from official `constants.py`:

```
 0: grabbing       1: grip           2: holy           3: point
 4: call           5: three3         6: timeout        7: xsign
 8: hand_heart     9: hand_heart2   10: little_finger 11: middle_finger
12: take_picture  13: dislike       14: fist          15: four
16: like          17: mute          18: ok            19: one (REMOVED)
20: palm          21: peace         22: peace_inverted 23: rock
24: stop          25: stop_inverted 26: three         27: three2
28: two_up        29: two_up_inverted 30: three_gun   31: thumb_index
32: thumb_index2  33: no_gesture
```

> [!NOTE]
> The `no_gesture` class is explicitly present in the pretrained checkpoint at index **33**.

### Locked NeuroGrip Taxonomy (14 Allowed Commands) & Mapping Table

| HaGRID Class Name | HaGRID Index | Mapped NeuroGrip Command | Status |
| :--- | :---: | :--- | :--- |
| `call` | 4 | `CALL` | Allowed |
| `fist` | 14 | `CLOSED_FIST` | Allowed |
| `four` | 15 | `FOUR_FINGERS` | Allowed |
| `grabbing` | 0 | `GRABBING` | Allowed |
| `grip` | 1 | `GRIP` | Allowed |
| `like` | 16 | `THUMBS_UP` | Allowed |
| `little_finger` | 10 | `PINKY` | Allowed |
| `middle_finger` | 11 | `MIDDLE_FINGER` | Allowed |
| `ok` | 18 | `OK` | Allowed |
| `point` | 3 | `INDEX_FINGER` | Allowed |
| `rock` | 23 | `INDEX_PINKY` | Allowed |
| `stop`, `stop_inverted` | 24, 25 | `STOP` | Allowed (Multi-alias) |
| `two_up`, `two_up_inverted` | 28, 29 | `TWO_FINGERS` | Allowed (Multi-alias) |
| `three`, `three2`, `three3`, `three_gun` | 26, 27, 5, 30 | `THREE_FINGERS` | Allowed (Multi-alias) |
| `one` | **19** | **`NO_COMMAND`** | **Explicitly Removed** |
| *All other classes* | *Various* | **`NO_COMMAND`** | **Blocked** |

> [!WARNING]
> `NO_COMMAND` must **never** be transmitted to hardware.

---

## 5. Measured PC Benchmarks & Performance

Measurements conducted on local machine (`CPU Execution Mode`):

| Metric | Measured Value |
| :--- | :--- |
| **Model Load Time** | **145.71 ms** |
| **Warm-Up Runs** | 5 iterations |
| **Benchmark Runs** | 100 timed frame iterations |
| **Average CPU Inference Latency** | **19.97 ms** |
| **Median CPU Inference Latency** | **19.51 ms** |
| **95th Percentile (P95) Latency** | **24.97 ms** |
| **Standard Deviation** | **2.64 ms** |
| **Theoretical Max FPS** | **50.1 FPS** |
| **Live Webcam Diagnostic FPS** | **~30.0 - 50.0 FPS** (Smooth live feedback) |

---

## 6. Unit Test Verification Results

Executed full pytest suite via `.venv\Scripts\pytest.exe pc/tests`:

```
====================== 243 passed, 2 warnings in 32.83s =======================
```

- **Total Tests Passed:** 243 / 243 (100% pass rate)
- **New HaGRID Tests:** 7 dedicated unit tests covering class order, taxonomy mapping, `one` removal, invalid class blocking, tensor preprocessing, and checkpoint loading.

---

## 7. Observations, Risks, & Architecture Analysis

1. **Full-Frame Classification Performance**:
   - The full-frame ResNet18 model runs extremely fast (~19.97 ms per frame on CPU).
   - However, when the hand occupies a small fraction of the webcam field of view, full-frame classification accuracy drops or becomes sensitive to background clutter.

2. **No-Hand / Distraction Handling**:
   - The presence of `no_gesture` (index 33) provides a baseline for non-gesture frames, but full-frame model confidence can fluctuate when background features resemble trained patterns.

3. **Phase 2 Architectural Recommendation**:
   - Rather than relying solely on full-frame classification, Phase 2 should evaluate a **two-stage architecture**:
     1. **Stage 1 (Hand Detection & Cropping)**: Use MediaPipe or YOLOv10n hand detector to locate and crop the active hand bounding box.
     2. **Stage 2 (Gesture Classification)**: Pass the cropped hand region into the verified HaGRID classifier.

---

## 8. Summary of Created Files

- [`pc/src/neurogrip/hagrid/__init__.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/__init__.py)
- [`pc/src/neurogrip/hagrid/config.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/config.py)
- [`pc/src/neurogrip/hagrid/taxonomy.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/taxonomy.py)
- [`pc/src/neurogrip/hagrid/model.py`](file:///d:/NeuroGrip_Project/pc/src/neurogrip/hagrid/model.py)
- [`pc/tools/hagrid_model_test.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_model_test.py)
- [`pc/tools/hagrid_webcam_diagnostic.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_webcam_diagnostic.py)
- [`pc/tests/test_hagrid_model.py`](file:///d:/NeuroGrip_Project/pc/tests/test_hagrid_model.py)
- [`pc/reports/hagrid_phase1_verification.md`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_phase1_verification.md)
- Checkpoint directory: [`pc/models/hagrid/ResNet18.pth`](file:///d:/NeuroGrip_Project/pc/models/hagrid/ResNet18.pth)
