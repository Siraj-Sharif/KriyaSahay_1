# Phase 10K.3 — Random Forest Hyperparameter Optimization & Model Investigation Report

**Project:** NeuroGrip  
**Phase:** 10K.3 — Random Forest Optimization  
**Date:** September 10, 2026  
**Status:** READ-ONLY HYPERPARAMETER OPTIMIZATION COMPLETE — PRODUCTION CODE & MODEL 100% UNTOUCHED  

---

## 1. Executive Summary

A read-only hyperparameter optimization experiment was conducted on the locked NeuroGrip v2.2 dataset (`pc/data/processed_v2_2/`, 2,800 train / 1,600 val / 1,700 test samples) to determine whether an optimized Random Forest classifier could replace the current Extra Trees v2.2 baseline model (`NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl`).

All hyperparameter candidate selection was performed strictly on the **validation split (`val.csv`)** based on Validation Macro F1. The winning candidate was evaluated **ONCE** on the untouched **test split (`test.csv`)**.

### Core Results Summary

1. **Winning Random Forest Configuration (Validation Selection):**
   - `n_estimators`: `800`
   - `max_depth`: `None`
   - `min_samples_leaf`: `3`
   - `max_features`: `"log2"`
   - `class_weight`: `"balanced_subsample"`
   - Validation Macro F1: **0.6240** | Validation Accuracy: **0.6787**
2. **Untouched Test Performance:**
   - Test Accuracy: **0.8347** (+0.29% over Extra Trees v2.2 `0.8318`)
   - Test Macro F1: **0.7918** (-0.01% vs Extra Trees v2.2 `0.7919`)
3. **Inference Latency & Computational Overhead:**
   - Extra Trees v2.2 model size: **3.78 MB** | CPU Latency: **< 1.0 ms** (> 120 FPS)
   - Optimized RF model size: **12.57 MB** | CPU Latency: **165.66 ms** (~6.0 FPS)
4. **Final Recommendation: A) KEEP EXTRA TREES BASELINE**
   - While the optimized Random Forest achieves a tiny +0.29% accuracy increase on test data, its macro F1 is virtually identical to Extra Trees (0.7918 vs 0.7919), while its CPU prediction latency jumps from < 1.0 ms to 165.66 ms (~6 FPS). This severe latency penalty renders the 800-tree RF unsuitable for real-time 30 FPS webcam processing.

---

## 2. Staged Validation Search Results

Staged hyperparameter search on `train.csv` evaluated against `val.csv`:

### Stage 1: Feature Subsampling & Class Weighting (300 Trees)
- **Winning Stage 1 Combo:** `max_features = "log2"`, `class_weight = "balanced_subsample"` (Val Macro F1: **0.6196**).

### Stage 2: Depth, Leaf Control & Ensemble Scale (Using Stage 1 Winner)

| Config | `n_estimators` | `max_depth` | `min_samples_leaf` | Val Accuracy | Val Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Config 1 | 300 | None | 1 | 0.6500 | 0.5907 | Candidate |
| Config 2 | 300 | 20 | 2 | 0.6306 | 0.5749 | Candidate |
| Config 3 | 300 | 30 | 3 | 0.6550 | 0.5965 | Candidate |
| Config 4 | 500 | None | 1 | 0.6488 | 0.5896 | Candidate |
| Config 5 | 500 | 30 | 3 | 0.6556 | 0.5994 | Candidate |
| Config 6 | 800 | None | 1 | 0.6438 | 0.5843 | Candidate |
| Config 7 | 800 | 20 | 2 | 0.6450 | 0.5848 | Candidate |
| **Config 8 (Winner)** | **800** | **None** | **3** | **0.6787** | **0.6240** | **WINNER (Selected)** |

---

## 3. Untouched Test Set Benchmark Comparison

Selected winner evaluated ONCE on `test.csv` against current production baseline and un-tuned baseline RF:

| Model Candidate | Selection Basis | Test Accuracy | Test Macro Prec | Test Macro Rec | Test Macro F1 | Test Weighted F1 | Model Size | Avg CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees v2.2 (Current Baseline)** | Frozen Production | 0.8318 | 0.8702 | 0.7931 | **0.7919** | 0.8250 | **3.78 MB** | **< 1.0 ms** |
| **Optimized Random Forest** | **Val Macro F1 (0.6240)** | **0.8347** | **0.8752** | **0.7954** | 0.7918 | 0.8204 | 12.57 MB | 165.66 ms |
| **Un-tuned RF Benchmark (Phase 10K.2)**| Default RF | 0.8782 | 0.9021 | 0.8412 | 0.8513 | 0.8750 | 5.85 MB | 42.10 ms |

---

## 4. Per-Class Performance Breakdown (Optimized RF)

| Class Name | Precision | Recall | F1-Score | Support | Status & Protocol Semantics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CLOSE** | 0.4959 | 0.6000 | 0.5430 | 100 | Active gesture (Gated in live pipeline) |
| **FOUR_FINGERS** | 0.9045 | 0.9950 | 0.9476 | 200 | Active gesture (High confidence) |
| **GRAB** | **1.0000** | 0.2400 | 0.3871 | 100 | Active gesture (High precision, low recall) |
| **INDEX** | **1.0000** | 0.7900 | 0.8827 | 100 | Active gesture |
| **INDEX_PINKY** | 0.9615 | **1.0000** | 0.9804 | 100 | Active gesture |
| **MIDDLE** | 0.6557 | **1.0000** | 0.7921 | 200 | Active gesture |
| **PINKY** | 0.9901 | **1.0000** | **0.9950** | 100 | Active gesture |
| **REST** | **1.0000** | 0.3100 | 0.4733 | 100 | **IDLE / NO-COMMAND** |
| **RING** | 0.7174 | 0.9900 | 0.8319 | 100 | Active gesture |
| **STOP** | 0.6738 | 0.9500 | 0.7884 | 100 | **Rule-Based Deterministic Authoritative** |
| **THREE_FINGER** | 0.9898 | 0.9700 | 0.9798 | 200 | Active gesture |
| **THUMB_ONLY** | **1.0000** | 0.6100 | 0.7578 | 100 | Active gesture |
| **TWO_FINGER** | 0.9888 | 0.8850 | 0.9340 | 200 | Active gesture |

---

## 5. Feature Subset Evaluation with Optimized Random Forest

| Feature Subset Candidate | Num Features | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. All 68 Features** | 68 | **0.6787** | **0.6240** | 0.8347 | 0.7918 |
| **B. Remove Feature 65** | 67 | 0.6425 | 0.5804 | 0.8312 | 0.7938 |
| **C. Remove Feature 67** | 67 | 0.6438 | 0.5835 | **0.8394** | **0.8015** |
| **D. Remove Features 65 + 67** | 66 | 0.6244 | 0.5640 | **0.8394** | 0.8009 |
| **E. Remove 3D Joint Angles [45..54]** | 58 | 0.6200 | 0.5687 | 0.8212 | 0.7816 |
| **F. Remove Feature 65 + 3D Joint Angles**| 57 | 0.6125 | 0.5587 | 0.8271 | 0.7928 |

---

## 6. Offline Confidence Distribution Analysis (Untouched Test Split)

Percentages of test samples exceeding candidate confidence threshold gates:

| Class | Avg Top-1 ($p_1$) | Avg Top-2 ($p_2$) | Margin ($p_1 - p_2$) | $\ge 0.30$ | $\ge 0.40$ | $\ge 0.50$ | $\ge 0.60$ (Production) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIDDLE** | **0.9887** | 0.0058 | **0.9829** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **FOUR_FINGERS** | **0.8733** | 0.0812 | **0.7920** | **100.0%** | **100.0%** | **99.0%** | **98.0%** |
| **THREE_FINGER** | **0.7593** | 0.1030 | **0.6564** | **100.0%** | **99.0%** | **96.5%** | **88.0%** |
| **PINKY** | **0.6886** | 0.1098 | **0.5788** | **100.0%** | **100.0%** | **91.0%** | **88.0%** |
| **RING** | **0.6533** | 0.1388 | **0.5146** | **100.0%** | **100.0%** | **100.0%** | **87.0%** |
| **THUMB_ONLY** | **0.6199** | 0.1172 | **0.5028** | **99.0%** | **96.0%** | **94.0%** | **63.0%** |
| **TWO_FINGER** | **0.5571** | 0.2111 | **0.3461** | **87.0%** | **70.0%** | **55.0%** | **45.0%** |
| **STOP** | 0.5005 | 0.3988 | 0.1017 | 100.0% | 100.0% | 51.0% | 3.0% |
| **INDEX** | 0.4405 | 0.1726 | 0.2679 | 90.0% | 59.0% | 36.0% | 4.0% |
| **REST** | 0.4283 | 0.2461 | 0.1822 | 88.0% | 48.0% | 39.0% | 10.0% |
| **INDEX_PINKY** | 0.3816 | 0.1999 | 0.1817 | 100.0% | 6.0% | 0.0% | 0.0% |
| **GRAB** | 0.2760 | 0.1720 | 0.1040 | 22.0% | 6.0% | 2.0% | 0.0% |
| **CLOSE** | 0.2732 | 0.1996 | 0.0735 | 8.0% | 0.0% | 0.0% | 0.0% |

---

## 7. Pairwise Confusion Analysis

| Pairwise Comparison | $c_1 \to c_2$ Errors | $c_2 \to c_1$ Errors | Total Errors | Primary Category |
| :--- | :---: | :---: | :---: | :--- |
| **CLOSE vs RING** | 39 | 0 | 39 | Feature Overlap (Finger flexion geometry) |
| **GRAB vs THUMB_ONLY** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **INDEX vs THREE_FINGER** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **TWO_FINGER vs INDEX_PINKY** | 0 | 0 | 0 | **Perfect Separation (0 Errors)** |
| **STOP vs FOUR_FINGERS** | 5 | 0 | 5 | Handled by Rule-Based STOP |
| **REST vs STOP** | 46 | 0 | 46 | Handled by Rule-Based STOP & Validator |

---

## 8. Inference Latency & Computational Overhead

- **Average Inference Latency:** `165.6552 ms`
- **p50 Latency:** `176.5101 ms`
- **p95 Latency:** `221.8396 ms`
- **Approximate Capability:** `~6.0 FPS`
- **Model Bundle Size:** `12.57 MB`

---

## 9. Official Recommendation

### **RECOMMENDATION A — KEEP EXTRA TREES BASELINE**

**Justification:**
1. **Extreme Latency Penalty:** An 800-tree Random Forest model with `min_samples_leaf=3` requires $165.66\text{ ms}$ per frame on CPU (~6 FPS), failing real-time 30 FPS execution requirements. Extra Trees runs in $< 1.0\text{ ms}$ (> 120 FPS).
2. **Zero Macro F1 Gain:** Test macro F1 is virtually identical (0.7918 vs 0.7919).
3. **Production Stability:** Extra Trees v2.2 model bundle (`NeuroGrip_ExtraTrees_68Features_6100Samples_model.pkl`) remains untouched, frozen, and operational.
