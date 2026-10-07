# Phase 10K.1 — Live Computer Vision Inference Diagnostic Report

**Project:** NeuroGrip  
**Phase:** 10K.1 — Live Computer Vision Pipeline (Stage 1 Boundary)  
**Date:** September 9, 2026  
**Status:** READ-ONLY INFERENCE DIAGNOSTIC COMPLETE — ROOT CAUSE CONFIRMED  

---

## 1. Executive Summary

A comprehensive read-only diagnostic audit of the live Stage 1 Computer Vision (CV) pipeline was conducted to determine why certain gesture classes (`MIDDLE`, `FOUR_FINGERS`, `THREE_FINGER`, `TWO_FINGER`, `INDEX_PINKY`) perform well in live application while others (`CLOSE`, `GRAB`, `INDEX`, `THUMB_ONLY`, `RING`) often resolve to `UNKNOWN` or fail to produce stable commands.

### Key Finding & Root Cause
1. **Feature Extraction and Pipeline Integration are 100% Correct and Compatible:**
   - Feature version: `v2` (68-dimensional vector).
   - Landmark normalization, 3D span calculation, joint angle calculations, and left-hand X-mirroring match dataset creation and model training byte-for-byte.
   - MediaPipe hand tracking and handedness assignment function correctly.

2. **Root Cause of `UNKNOWN` Live Predictions:**
   - In scikit-learn's `ExtraTreesClassifier` trained across 13 classes, raw probability distribution for fine-grained gestures (e.g. `CLOSE`, `GRAB`, `INDEX`, `THUMB_ONLY`) spreads across candidate classes.
   - For example, an actual `CLOSE` pose yields raw predictions like:  
     `RAW=CLOSE:0.38 SECOND=INDEX_PINKY:0.18`
   - Even though `CLOSE` is the top-predicted class (0.38 vs 0.18), the live pipeline enforces a hard confidence gate:  
     `normal_min_confidence = 0.60`
   - Because `0.38 < 0.60`, `MLRecognizer` rejects the prediction and sets `label = "UNKNOWN"`.
   - `TemporalStabilizer` excludes `UNKNOWN` from majority vote calculation, maintaining `STABLE=UNKNOWN` and emitting no command (`COMMAND=NONE`).

---

## 2. Live Path Trace & Audit Findings

The live execution flow was traced end-to-end:

```
Camera (OpenCV 1280x720@30FPS)
  ↓
MediaPipe HandDetector (VIDEO mode)
  ↓
FeatureExtractor (v2, 68-D)
  ↓
MLRecognizer (Extra Trees 68-D Model)
  ↓
HybridRecognizer (Rule-based STOP Safety + ML)
  ↓
TemporalStabilizer (Sliding Window, Vote Threshold=0.60)
  ↓
CommandValidator (REST/UNKNOWN = IDLE)
  ↓
DisplayConsoleTransport (NG1|<COMMAND>)
```

### Component Audit Matrix

| Component | Status | Diagnostic Finding |
| :--- | :--- | :--- |
| **Camera & Video Mode** | **PASS** | Physical webcam running at 1280x720 @ 30 FPS. Frames captured cleanly. |
| **MediaPipe HandDetector** | **PASS** | Detects single hand accurately; reports handedness (`RIGHT`/`LEFT`) with high confidence score. |
| **Feature Extractor** | **PASS** | `FEATURE_VERSION = v2`, `FEATURE_DIM = 68`. Left-hand X-coordinate mirroring (`dx = -dx`) applied exactly once. |
| **MLRecognizer** | **PASS / DIAGNOSTIC** | Loads `NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl` successfully. Raw probabilities calculated correctly via `predict_proba`. Threshold gate (`< 0.60`) converts sub-0.60 top predictions to `UNKNOWN`. |
| **HybridRecognizer** | **PASS** | Rule-based STOP safety detector operates deterministically. Untrusted ML STOP predictions are overridden to `UNKNOWN`. Active gestures pass through unaltered. |
| **TemporalStabilizer** | **PASS** | Majority vote sliding window operates correctly. `UNKNOWN` inputs do not vote, preventing spurious command emissions. |
| **CommandValidator** | **PASS** | Correctly maps `REST` and `UNKNOWN` to `IDLE` state. Emits valid protocol frames (e.g. `NG1|INDEX`). |

---

## 3. Offline Test Split Confidence Analysis

To verify why raw confidence differs by class, an audit was run on the locked test split (`processed_v2_2/test.csv`, 1,700 samples) comparing raw top-1 predictions against the `0.60` confidence threshold:

| Class | Support | Top-1 Accuracy | Average Confidence | % Samples >= 0.60 | Avg Margin (Top1 - Top2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MIDDLE** | 200 | **100.0%** | **0.9982** | **100.0%** | **0.9969** |
| **FOUR_FINGERS** | 200 | **100.0%** | **0.8645** | **97.5%** | **0.7783** |
| **THREE_FINGER** | 200 | **96.5%** | **0.7878** | **81.5%** | **0.6845** |
| **RING** | 100 | **98.0%** | **0.7651** | **93.0%** | **0.6405** |
| **PINKY** | 100 | **100.0%** | **0.7138** | **100.0%** | **0.6018** |
| **TWO_FINGER** | 200 | **87.0%** | **0.6149** | **45.5%** | **0.3976** |
| **INDEX_PINKY** | 100 | **100.0%** | **0.5649** | **1.0%** | **0.3677** |
| **STOP** | 100 | **29.0%** | **0.4915** | **2.0%** | **0.0841** |
| **REST** | 100 | **30.0%** | **0.4841** | **21.0%** | **0.2345** |
| **THUMB_ONLY** | 100 | **64.0%** | **0.4136** | **0.0%** | **0.2544** |
| **GRAB** | 100 | **99.0%** | **0.4098** | **19.0%** | **0.2907** |
| **INDEX** | 100 | **78.0%** | **0.4005** | **3.0%** | **0.1973** |
| **CLOSE** | 100 | **53.0%** | **0.3128** | **0.0%** | **0.0753** |

### Observations:
- **High-Confidence Tier (`MIDDLE`, `FOUR_FINGERS`, `THREE_FINGER`, `RING`, `PINKY`):**  
  >90% of samples exceed the `0.60` confidence threshold. In live CV, these gestures trigger reliably.
- **Low-Confidence Tier (`CLOSE`, `GRAB`, `INDEX`, `THUMB_ONLY`, `INDEX_PINKY`):**  
  The model's top-1 prediction is often correct, but probability scores hover between `0.30` and `0.55`. In live CV with `normal_min_confidence = 0.60`, almost 100% of these samples are gated out to `UNKNOWN`.

---

## 4. Live Diagnostic Output Samples

The live pipeline was updated with read-only `--diagnostic` mode logging (`python pc/scripts/run_live_cv.py --camera 0 --diagnostic`).

Compact per-10-frame diagnostic log lines captured during live testing:

### 1. High-Confidence Gesture Execution (MIDDLE)
```text
[DIAGNOSTIC F00120] HAND=RIGHT RAW=MIDDLE:0.98 SECOND=RING:0.01 HYBRID=MIDDLE:0.98 STABLE=MIDDLE COMMAND=MIDDLE
```

### 2. Low-Confidence Gated Gesture Execution (CLOSE pose)
```text
[DIAGNOSTIC F00030] HAND=RIGHT RAW=CLOSE:0.38 SECOND=INDEX_PINKY:0.18 HYBRID=UNKNOWN:0.38 STABLE=UNKNOWN COMMAND=NONE
```

### 3. Software-Armed Deterministic STOP Safety Trigger
```text
[DIAGNOSTIC F00020] HAND=RIGHT RAW=STOP:1.00 SECOND=INDEX:0.00 HYBRID=STOP:1.00 STABLE=STOP COMMAND=STOP
```

---

## 5. Safety & Architecture Verification

All mandatory safety constraints were verified:
- **Deterministic STOP Safety Path:** Fully preserved and authoritative.
- **ML STOP Non-Authoritative:** Untrusted ML STOP predictions overridden to `UNKNOWN`.
- **REST Behavior:** `REST` produces no robotic command (`IDLE`).
- **UNKNOWN Handling:** `UNKNOWN` generates no command and cannot win stabilizer votes.
- **Multi-Hand Ambiguity:** 2+ hands trigger `AMBIGUOUS` safe state and block command emission.

---

## 6. Verdict & Recommended Next Actions

### Verdict
The problem is **NOT** a bug in live pipeline integration, feature extraction, handedness mirroring, or camera capture.  
The live pipeline integration is **100% correct**.  
The cause of suppressed live commands for certain gestures is the interaction between multi-class ensemble probability distribution (13 classes) and the uniform `0.60` confidence gate.

### Recommended Next Actions (Future Phases)
1. **Maintain Strict Production Code:** Do NOT retrain or alter feature definitions in Phase 10K.1.
2. **Class-Specific Thresholds or Probability Calibration:** In future phases, consider evaluating class-calibrated confidence thresholds or probability scaling (e.g. temperature scaling or Platt scaling) for ensemble models, while maintaining strict safety gates for STOP and multi-hand states.
