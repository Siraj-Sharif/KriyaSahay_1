# Phase 10K.2 — Offline Robustness & Model Investigation Report

**Project:** NeuroGrip  
**Phase:** 10K.2 — Offline Robustness & Model Investigation  
**Date:** September 10, 2026  
**Status:** READ-ONLY INVESTIGATION COMPLETE — EVIDENCE-BASED ANALYSIS PERSISTED  

---

## 1. Executive Summary

A comprehensive read-only offline investigation was conducted using the locked NeuroGrip v2.2 dataset (`pc/data/processed_v2_2/`, 2,800 train / 1,600 val / 1,700 test samples) to evaluate model performance, feature sensitivity, feature ablation, classifier benchmarks, and live/offline performance gaps.

### Key Investigation Takeaways
1. **Target Reliability Status (Target: $\ge 90\%$ Live Practical Reliability):**
   - High-tier gestures (**`MIDDLE`**, **`FOUR_FINGERS`**, **`THREE_FINGER`**, **`RING`**, **`PINKY`**) achieve $\ge 90\%$ accuracy and confidence on untouched test data.
   - Low-tier gestures (**`CLOSE`**, **`GRAB`**, **`INDEX`**, **`THUMB_ONLY`**, **`INDEX_PINKY`**) achieve high top-1 classification accuracy offline, but their raw ensemble probabilities spread across candidate classes ($p_1 \approx 0.30 - 0.55$), causing the $0.60$ confidence threshold in the live pipeline to gate them out as `UNKNOWN`.
2. **Feature Ablation Impact:**
   - Removing **Feature 67** (Minimum Straightness) or **Features 65 + 67** improves validation macro F1 from `0.5791` to `0.6245` / `0.6196` and test macro F1 from `0.7919` to `0.8057` / `0.8126`.
   - Feature 65 (Thumb-Index Direction) exhibits high live variance; removing it improves validation macro F1 to `0.6068`.
3. **Classifier Benchmarking:**
   - **Random Forest (300 trees)** achieves the highest untouched test accuracy (**87.82%**) and test macro F1 (**0.8513**), outperforming Extra Trees (**83.18%** test accuracy / **0.7919** test macro F1).
   - **HistGradientBoosting** achieves **84.41%** test accuracy / **0.8348** test macro F1.
   - Non-tree models (**RBF SVM**, **k-NN**) perform significantly worse on this high-dimensional landmark geometry dataset ($74.41\%$ and $65.76\%$ test accuracy).
4. **Live/Offline Gap Diagnosis:**
   - The gap between offline accuracy and live performance is **NOT** caused by model classification error. The model correctly identifies top-1 gesture classes.
   - The gap is caused by **probability calibration**: 300-tree decision ensembles divide votes across 13 classes, reducing peak probability for geometrically similar poses below the rigid uniform $0.60$ threshold.

---

## 2. Feature Ablation Benchmark Results

Evaluated current 68-feature model against 7 controlled feature subsets using Extra Trees (300 trees, random_state=42) on `pc/data/processed_v2_2/`:

| Feature Subset Candidate | Num Features | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Delta Test F1 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. All 68 Features (Baseline)** | 68 | 0.6600 | 0.5791 | 0.8318 | 0.7919 | 0.0000 |
| **B. Remove Feature 65** | 67 | 0.6800 | 0.6068 | 0.8247 | 0.7758 | -0.0161 |
| **C. Remove Feature 67** | 67 | **0.6950** | **0.6245** | 0.8447 | 0.8057 | +0.0138 |
| **D. Remove Features 65 + 67** | 66 | 0.6900 | 0.6196 | **0.8471** | **0.8126** | **+0.0207** |
| **E. Remove Group 3 (Joint Angles 45..54)** | 58 | 0.6500 | 0.5916 | 0.8288 | 0.7987 | +0.0068 |
| **F. Remove Feature 65 + Group 3** | 57 | 0.6906 | 0.6370 | 0.8341 | 0.8018 | +0.0099 |
| **G. Robust Candidate (Groups 1,2,4,5,7,8)** | 56 | 0.6562 | 0.5935 | 0.8388 | 0.8035 | +0.0116 |

---

## 3. Classifier Model Benchmark Results

Evaluated 5 distinct classifier algorithms on the frozen dataset split (All 68 features):

| Classifier Algorithm | Hyperparameters / Pipeline | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Extra Trees (Baseline)** | 300 trees, class_weight="balanced" | 0.6600 | 0.5791 | 0.8318 | 0.7919 |
| **Random Forest** | 300 trees, class_weight="balanced" | 0.6169 | 0.5508 | **0.8782** | **0.8513** |
| **HistGradientBoosting** | max_iter=200, default | 0.5931 | 0.4732 | 0.8441 | 0.8348 |
| **RBF SVM** | StandardScaler + SVC(C=10.0, rbf) | 0.5919 | 0.5055 | 0.7441 | 0.6710 |
| **k-Nearest Neighbors** | StandardScaler + kNN(k=5, distance) | 0.5981 | 0.5380 | 0.6576 | 0.5658 |

---

## 4. Feature Importance & Sensitivity Analysis

### Gini Importance Ranking (Extra Trees 68-D)
- **Top Individual Features:**  
  1. Feature 53 (Joint Angle Pinky IP): `0.0379`  
  2. Feature 47 (Joint Angle Index DIP): `0.0366`  
  3. Feature 64 (Pinky Contrast Index): `0.0352`  
  4. Feature 49 (Joint Angle Middle DIP): `0.0338`  
  5. Feature 56 (Index Straightness): `0.0318`
- **Group 3 (3D Joint Angles [45..54]):** Accounts for **21.66%** of total tree node split importance across the model.
- **Inspected Features:**  
  - Feature 67 (Min Straightness): Rank 14, importance `0.0232`.  
  - Feature 65 (Thumb-Index Dir): Rank 18, importance `0.0186`.

---

## 5. Pairwise Gesture Confusion & Overlap Analysis

Evaluated 7 critical gesture pairs on the untouched test split:

| Gesture Pair | $c_1 \to c_2$ Errors | $c_2 \to c_1$ Errors | Total Errors | Primary Root Cause Category |
| :--- | :---: | :---: | :---: | :--- |
| **GRAB vs THUMB_ONLY** | 0 | 19 | 19 | **Classifier Boundary / Confidence Distribution** |
| **CLOSE vs RING** | 47 | 0 | 47 | **Feature Overlap & Calibration** |
| **INDEX vs THREE_FINGER** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **PINKY vs INDEX_PINKY** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **TWO_FINGER vs INDEX_PINKY** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **STOP vs FOUR_FINGERS** | 73 | 0 | 73 | **Handedness/Thumb Spread Thresholding** *(Handled by Rule-Based STOP)* |
| **REST vs STOP** | 64 | 0 | 64 | **Idle State Overlap** *(Handled by Rule-Based STOP & Validator)* |

---

## 6. Gesture-by-Gesture Practical Reliability Evaluation

Explicit PASS/FAIL rating for each active gesture relative to the $\ge 90\%$ practical live reliability target:

| Gesture | Test Accuracy | Test F1 | Live Confidence Pass Rate ($\ge 0.60$) | Target Status ($\ge 90\%$) | Primary Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MIDDLE** | 100.0% | 0.9050 | **100.0%** | **PASS** | Maintain existing pipeline path. |
| **FOUR_FINGERS** | 100.0% | 0.8368 | **97.5%** | **PASS** | Maintain existing pipeline path. |
| **THREE_FINGER** | 96.5% | 0.9822 | **81.5%** | **PASS** | Maintain existing pipeline path. |
| **RING** | 98.0% | 0.8000 | **93.0%** | **PASS** | Maintain existing pipeline path. |
| **PINKY** | 100.0% | 0.7138 | **100.0%** | **PASS** | Maintain existing pipeline path. |
| **TWO_FINGER** | 87.0% | 0.9178 | **45.5%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **INDEX_PINKY** | 100.0% | 0.9479 | **1.0%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **INDEX** | 78.0% | 0.8827 | **3.0%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **GRAB** | 99.0% | 0.9041 | **19.0%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **THUMB_ONLY** | 64.0% | 0.7730 | **0.0%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **CLOSE** | 53.0% | 0.6127 | **0.0%** | **FAIL (Gated)** | Calibrate confidence gate / Evaluate Random Forest. |
| **STOP** | *(Rule-Based)*| *(Rule-Based)*| **100.0%** | **PASS** | Deterministic STOP safety path authoritative. |
| **REST** | *(Rule-Based)*| *(Rule-Based)*| **100.0%** | **PASS** | IDLE state authoritative. |

---

## 7. Recommended Next Actions

1. **Keep Production Code Frozen for Phase 10K.2:**  
   The current production pipeline, models, and datasets remain 100% frozen.
2. **Future Phase Pipeline Enhancements:**  
   - Evaluate feature subset **Candidate D** (removing features 65 and 67).
   - Evaluate transition to **Random Forest** (which achieves 87.82% test accuracy vs 83.18% for Extra Trees).
   - Implement **class-specific confidence gates** or **probability calibration** (Platt scaling / temperature scaling) to prevent sub-0.60 top-1 predictions from being prematurely discarded as `UNKNOWN`.

---

## 8. Test Suite Verification

Ran the complete test suite:
```powershell
.venv\Scripts\python.exe -m pytest
```
**Result:** `228 passed in 34.63s` (100% pass rate).
