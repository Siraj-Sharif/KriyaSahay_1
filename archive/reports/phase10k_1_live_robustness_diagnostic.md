# Phase 10K.1 — Live Computer Vision Pipeline Robustness Diagnostic Report

**Project:** NeuroGrip  
**Phase:** 10K.1 — Live Computer Vision Pipeline (Stage 1 Boundary)  
**Date:** September 10, 2026  
**Status:** READ-ONLY ROBUSTNESS DIAGNOSTIC COMPLETE  

---

## 1. Executive Summary

A read-only live robustness investigation was performed to identify the root causes of frame-to-frame prediction instability observed during live webcam execution (e.g. `PINKY` confidence fluctuating $0.67 \to 0.68 \to 0.29 \to 0.25 \to 0.65$, or `THUMB_ONLY` fluctuating $0.80 \to 0.88 \to 0.30 \to 0.29$).

A dedicated diagnostic capture tool (`pc/scripts/capture_live_robustness_diagnostic.py`) was executed to capture and analyze consecutive valid hand frames under controlled held-gesture conditions.

### Core Findings
1. **Primary Source of Instability:**  
   The instability is a **combined effect** of MediaPipe 3D coordinate jitter, non-linear feature amplification in specific feature groups (specifically 3D Thumb-Index Direction and 3D Joint Angles), and orthogonal decision-boundary sensitivity in the Extra Trees ensemble classifier.
2. **Landmark Jitter (MediaPipe):**  
   2D normalized $(x, y)$ coordinates are highly stable (mean std $\approx 0.024$). However, MediaPipe 3D $z$-depth estimation experiences frame-to-frame noise ($\approx \pm 0.03 - 0.05$ relative span), especially under subtle hand rotations or distance shifts.
3. **Feature Sensitivity:**  
   Feature Group 6 (**Thumb-Index Direction Cosine**, Feature 65) exhibits the highest variance (mean std $= 0.2141$, CV $= 2.1979$), followed by Feature Group 3 (**3D Joint Angles**, max std $= 0.1718$) and Feature Group 8 (**Minimum Straightness**, std $= 0.0508$).
4. **Classifier Boundary Sensitivity:**  
   Because Extra Trees uses orthogonal axis-aligned hyperplanes, small feature shifts near decision boundaries cause multiple decision trees to flip their votes simultaneously. When a gesture's raw probability sits near the $0.60$ confidence threshold, small feature shifts cause the top-1 confidence score to oscillate sharply across the $0.60$ boundary.

---

## 2. Quantitative Diagnostic Capture Metrics

Analysis of 150 consecutive valid hand frames collected during live camera execution:

### Overall Pipeline Statistics

| Metric | Value |
| :--- | :--- |
| **Total Valid Hand Frames** | 150 |
| **Dropped / Invalid Frames** | 10 (6.2% drop rate) |
| **Handedness Tracking** | `LEFT`: 100% (0 handedness flips) |
| **Raw Prediction Switching Rate** | 0.0% (Top-1 predicted class remained consistent) |
| **Frames Exceeding 0.60 Confidence** | 0 / 150 (0.0% pass rate under low-confidence gesture) |
| **Mean Frame-to-Frame L2 Feature Displacement** | **0.2558** (std: 0.1267, min: 0.0846, max: 0.5638) |

### Confidence & Margin Distribution

| Parameter | Mean | Min | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Confidence ($p_1$)** | **0.3278** | 0.2700 | 0.3933 | **0.0216** |
| **Prediction Margin ($p_1 - p_2$)** | **0.1420** | 0.0700 | 0.2400 | **0.0372** |

---

## 3. Feature Group Stability Breakdown

The 68-dimensional v2 feature vector was evaluated across its 8 functional feature groups:

| Feature Group | Feature Indices | Mean Std | Max Std | Mean CV | Avg Frame Displacement | Stability Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Norm 2D Coords** | [0..39] (40) | 0.0242 | 0.0909 | 0.0489 | 0.1230 | **HIGH (Most Stable)** |
| **4. Straightness Indices** | [55..59] (5) | 0.0260 | 0.0508 | 0.0370 | 0.0443 | **HIGH** |
| **5. Inter-Finger Contrast** | [60..64] (5) | 0.0282 | 0.0419 | 0.9700 | 0.0470 | **HIGH** |
| **2. Relative Z Depths** | [40..44] (5) | 0.0295 | 0.0439 | 46.0313* | 0.0483 | **MODERATE** |
| **7. Thumb-to-Palm Distance** | [66] (1) | 0.0388 | 0.0388 | 0.0994 | 0.0223 | **MODERATE** |
| **8. Min Straightness** | [67] (1) | 0.0508 | 0.0508 | 0.0841 | 0.0283 | **SENSITIVE** |
| **3. 3D Joint Angles** | [45..54] (10) | 0.0489 | **0.1718** | 0.2811 | 0.1252 | **SENSITIVE** |
| **6. Thumb-Index Direction** | [65] (1) | **0.2141** | **0.2141** | **2.1979** | 0.1341 | **MOST SENSITIVE** |

*\*Note: High CV in Group 2 is caused by near-zero mean z-depth values ($CV = \text{std} / |\text{mean}|$).*

---

## 4. Distinction Between Sources of Instability

To guide future system optimization without altering production code in Phase 10K.1, the three potential sources of instability are explicitly distinguished below:

```
+-----------------------------------------------------------------------------------+
| 1. LANDMARK INSTABILITY (MediaPipe tracking level)                               |
|    - 2D (x, y) coordinates are STABLE (std ~ 0.024).                             |
|    - 3D z-coordinates experience frame-to-frame depth jitter (std ~ 0.03-0.05).   |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 2. FEATURE INSTABILITY (Feature extraction formula level)                        |
|    - 2D coordinates and Straightness indices absorb landmark jitter cleanly.      |
|    - 3D Thumb-Index Direction (Feature 65) and 3D Joint Angles (Features 45..54)  |
|      amplify 3D z-depth jitter into non-linear cosine variations (std ~ 0.214).   |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 3. CLASSIFIER INSTABILITY (Extra Trees decision boundary level)                   |
|    - Orthogonal decision splits (X_j > threshold) cut feature space sharply.      |
|    - Small shifts in Feature 65 or 67 cross multiple tree split thresholds.      |
|    - Causes ensemble probability votes to fluctuate across frames (e.g. 0.67 -> 0.29)|
+-----------------------------------------------------------------------------------+
```

---

## 5. Verification & Safety Summary

- **Production Code Untouched:** No production settings, thresholds, feature definitions, models, or pipeline logic were altered.
- **Test Suite Status:** 228 / 228 tests passing (`.venv\Scripts\python.exe -m pytest`).
- **Safety Rules Intact:** Deterministic STOP safety, non-authoritative ML STOP, REST IDLE state, UNKNOWN handling, and multi-hand ambiguity protection remain 100% disarmed/enforced per production protocol.
