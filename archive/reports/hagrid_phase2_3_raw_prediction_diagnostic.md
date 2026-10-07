# NeuroGrip CV Phase 2.3 — Raw HaGRID Prediction Diagnostic Report

**Date:** September 14, 2026  
**Environment:** Windows (x86_64), Python 3.11, PyTorch 2.14.0+cpu, OpenCV 5.0.0  
**Project Root:** `D:\NeuroGrip_Project`  
**Status:** Diagnostic Tool & Pipeline Implemented — Ready for Manual Target Testing  

---

## 1. Executive Summary & Purpose

Phase 2.3 provides an isolated raw prediction diagnostic tool to investigate the raw, un-gated output probabilities of the official HaGRID ResNet18 full-frame classifier for 3 problematic gestures:

1. **`INDEX_FINGER`** (Expected HaGRID Raw Class: `point`)
2. **`TWO_FINGERS`** (Expected HaGRID Raw Classes: `two_up`, `two_up_inverted`)
3. **`PINKY`** (Expected HaGRID Raw Class: `little_finger`)

> [!IMPORTANT]
> - Phase 2.3 is an isolated diagnostic phase.
> - Production HaGRID model weights, confidence thresholds, taxonomy mappings, and legacy Extra Trees / Random Forest code were **not** modified or deleted.
> - Serial transmission, ESP32 integration, Webots control, motor control, and model training/fine-tuning were **not** performed.
> - No confidence cutoff is applied during diagnostic logging; raw probabilities down to float precision are recorded.

---

## 2. Diagnostic Architecture & Pipeline

```
USB Webcam Frame (H x W x 3)
  ↓
Existing HaGRID ResNet18 Preprocessing (Longest side 224, pad 224x224 with [144,144,144], normalize mean/std)
  ↓
Official HaGRID ResNet18 Classifier (34 logits)
  ↓
Softmax Un-Gated Probability Vector (34 floats)
  ↓
Top-5 Prediction Extraction (Preserving full float precision)
  ↓
Existing Taxonomy Mapping (Displayed FOR REFERENCE ONLY, after raw class)
```

---

## 3. Target Gesture Specifications

| Canonical Target Gesture | Keyboard Shortcut | Expected Official HaGRID Raw Class(es) |
| :--- | :---: | :--- |
| **`INDEX_FINGER`** | `[1]` | `point` |
| **`TWO_FINGERS`** | `[2]` | `two_up`, `two_up_inverted` |
| **`PINKY`** | `[3]` | `little_finger` |

---

## 4. Structured JSON Output Schema

Diagnostic trial recordings are saved directly to:  
[`pc/reports/hagrid_raw_prediction_diagnostic.json`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_raw_prediction_diagnostic.json)

### Trial JSON Format Example
```json
[
  {
    "timestamp": "2026-09-14T12:35:00.123456",
    "target_gesture": "INDEX_FINGER",
    "expected_raw_classes": ["point"],
    "duration_s": 5.0,
    "total_frames": 150,
    "summary_stats": {
      "target_gesture": "INDEX_FINGER",
      "expected_raw_classes": ["point"],
      "total_frames": 150,
      "hit_rate_pct": 65.0,
      "mean_confidence_pct": 42.15,
      "median_confidence_pct": 41.80,
      "p95_confidence_pct": 58.40,
      "unsupported_pct": 35.0,
      "switch_count": 8,
      "distribution_pct": {
        "point": 65.0,
        "one": 20.0,
        "rock": 15.0
      },
      "top_confusions": [
        ["one", 20.0],
        ["rock", 15.0]
      ]
    },
    "frames": [
      {
        "frame_index": 0,
        "timestamp": "2026-09-14T12:35:00.123456",
        "raw_top1_class": "point",
        "raw_top1_probability": 0.384721123456,
        "top5": [
          {"class": "point", "probability": 0.384721123456},
          {"class": "one", "probability": 0.217843112233},
          {"class": "rock", "probability": 0.071532112233},
          {"class": "stop", "probability": 0.043217112233},
          {"class": "two_up", "probability": 0.031184112233}
        ],
        "canonical_command": "INDEX_FINGER"
      }
    ]
  }
]
```

---

## 5. Launch Instructions

To run the raw prediction diagnostic GUI:

```powershell
.venv\Scripts\python.exe pc/tools/hagrid_raw_prediction_diagnostic.py
```

### Controls:
- **`1`**: Select `INDEX_FINGER` (Expected raw: `point`)
- **`2`**: Select `TWO_FINGERS` (Expected raw: `two_up`, `two_up_inverted`)
- **`3`**: Select `PINKY` (Expected raw: `little_finger`)
- **`SPACE`**: Start 5-second raw diagnostic trial recording
- **`ESC`**: Exit diagnostic GUI

---

## 6. Created Files Overview

- [`pc/tools/hagrid_raw_prediction_diagnostic.py`](file:///d:/NeuroGrip_Project/pc/tools/hagrid_raw_prediction_diagnostic.py) *(NEW)*
- [`pc/tests/test_hagrid_raw_prediction_diagnostic.py`](file:///d:/NeuroGrip_Project/pc/tests/test_hagrid_raw_prediction_diagnostic.py) *(NEW)*
- [`pc/reports/hagrid_phase2_3_raw_prediction_diagnostic.md`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_phase2_3_raw_prediction_diagnostic.md) *(NEW)*
- Output JSON: [`pc/reports/hagrid_raw_prediction_diagnostic.json`](file:///d:/NeuroGrip_Project/pc/reports/hagrid_raw_prediction_diagnostic.json)
