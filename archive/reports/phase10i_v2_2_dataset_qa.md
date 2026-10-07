# NeuroGrip Phase 10I.1 — Dataset v2.2 QA Report

> [!IMPORTANT]
> **Dataset Status**: **PASS**  
> All 6,100 samples across 50 unique sessions and 13 locked command classes pass dataset quality assurance with zero missing values, non-finite features, schema errors, or duplicate rows. Legacy datasets `raw_v2_pilot`, `raw_v2_phase10h`, `raw_v2_1`, `processed_v2_1`, and model artifacts (`neurogrip_extra_trees_v2_1.pkl`) remain 100% frozen and untouched.

---

## 1. Executive Summary & Core Metrics

| Metric | Target / Spec | Achieved v2.2 Value | QA Status |
| :--- | :--- | :--- | :---: |
| **Total Samples** | 6,100 samples | **6,100 samples** | **PASS** |
| **Total Sessions** | 50 unique sessions | **50 unique sessions** | **PASS** |
| **Total CSV Files** | 50 CSV files | **50 CSV files** | **PASS** |
| **Command Classes** | 13 locked classes | **13 locked classes** | **PASS** |
| **`MIDDLE` Samples** | 900 samples (500 v2.1 + 400 new) | **900 samples** | **PASS** |
| **`REST` Class** | 300 samples (Preserved for comparison) | **300 samples** | **PASS** |
| **Feature Dimensions** | 68-D (v2 features) | **68-D (v2 features)** | **PASS** |
| **Session Leakage** | 0 session overlap | **0 session overlap** | **PASS** |
| **Duplicate Rows** | 0 duplicates | **0 duplicates** | **PASS** |
| **Non-Finite Feature Values** | 0 NaN/Inf | **0 NaN/Inf** | **PASS** |

---

## 2. Per-Class Sample, Session & Handedness Distribution

The v2.2 dataset combines the 5,700 samples from `raw_v2_1` (48 sessions) with 400 newly collected targeted `MIDDLE` samples from `raw_v2_phase10h_middle` (2 sessions: `3529a937` [RIGHT] & `cf9b7593` [LEFT]).

| Class Name | Total Samples | Total Sessions | Handedness (Right / Left) | Handedness Balance |
| :--- | :---: | :---: | :---: | :---: |
| **`CLOSE`** | 300 | 3 | 216 / 84 | 72% R / 28% L |
| **`FOUR_FINGERS`** | 700 | 5 | 305 / 395 | 44% R / 56% L |
| **`GRAB`** | 300 | 3 | 201 / 99 | 67% R / 33% L |
| **`INDEX`** | 300 | 3 | 200 / 100 | 67% R / 33% L |
| **`INDEX_PINKY`** | 500 | 4 | 294 / 206 | 59% R / 41% L |
| **`MIDDLE`** | **900** | **6** | **507 / 393** | **56% R / 44% L** |
| **`PINKY`** | 300 | 3 | 200 / 100 | 67% R / 33% L |
| **`REST`** | 300 | 3 | 200 / 100 | 67% R / 33% L |
| **`RING`** | 300 | 3 | 201 / 99 | 67% R / 33% L |
| **`STOP`** | 500 | 4 | 283 / 217 | 57% R / 43% L |
| **`THREE_FINGER`** | 700 | 5 | 415 / 285 | 59% R / 41% L |
| **`THUMB_ONLY`** | 300 | 3 | 200 / 100 | 67% R / 33% L |
| **`TWO_FINGER`** | 700 | 5 | 430 / 270 | 61% R / 39% L |
| **TOTAL** | **6,100** | **50** | **3,652 / 2,448** | **60% R / 40% L** |

---

## 3. Session-Isolated Data Splits (`processed_v2_2`)

Splits were generated using `SessionSplitter` (seed = 42) guaranteeing **zero session leakage** across train, validation, and test sets.

### 3.1 Split Overview

| Split | Total Samples | Share (%) | Session Count | Handedness (Right / Left) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 2,800 | 45.9% | 24 sessions | 1,815 / 985 |
| **Validation** | 1,600 | 26.2% | 13 sessions | 826 / 774 |
| **Test** | 1,700 | 27.9% | 13 sessions | 1,011 / 689 |
| **TOTAL** | **6,100** | **100.0%** | **50 sessions** | **3,652 / 2,448** |

### 3.2 Session Assignment Lists
- **Train Sessions (24)**: `00975a40`, `04795eb9`, `0c09728d`, `12146e1e`, `12b37a51`, `1e0fa45b`, `2f6cb1a9`, **`3529a937` (New `MIDDLE`)**, `4440714e`, `47df7776`, `4c1af894`, `66c78629`, `6e6156a9`, `921bebc9`, `93093601`, `9e729013`, `9ed6e8a4`, `a3d7731f`, `adcc2a63`, `bc8cdd5d`, `cad64713`, `dc093c62`, `f201633c`, `f5dea6a0`
- **Validation Sessions (13)**: `23d2fc74`, `30b9512a`, `50c25a3a`, `59b5deb3`, `5b0ebee6`, `6719f9a8`, `704874c0`, `71d28457`, `7423ad8a`, `888b8faf`, `a1e7305c`, `cc78f23d`, `d11c8328`
- **Test Sessions (13)**: `074efa27`, `2a894dad`, `44a8d2b1`, `4a56a531`, `53233b15`, `60128564`, `66204906`, `9298d4a7`, `9746bae6`, `a532bc8c`, `cbe96b07`, `ce8ffb3d`, **`cf9b7593` (New `MIDDLE`)**

### 3.3 Class Sample Distribution Across Splits

| Class Name | Train Samples | Val Samples | Test Samples | Total Class Samples |
| :--- | :---: | :---: | :---: | :---: |
| `CLOSE` | 100 | 100 | 100 | 300 |
| `FOUR_FINGERS` | 400 | 100 | 200 | 700 |
| `GRAB` | 100 | 100 | 100 | 300 |
| `INDEX` | 100 | 100 | 100 | 300 |
| `INDEX_PINKY` | 200 | 200 | 100 | 500 |
| **`MIDDLE`** | **500** | **200** | **200** | **900** |
| `PINKY` | 100 | 100 | 100 | 300 |
| `REST` | 100 | 100 | 100 | 300 |
| `RING` | 100 | 100 | 100 | 300 |
| `STOP` | 200 | 200 | 100 | 500 |
| `THREE_FINGER` | 400 | 100 | 200 | 700 |
| `THUMB_ONLY` | 100 | 100 | 100 | 300 |
| `TWO_FINGER` | 400 | 100 | 200 | 700 |
| **TOTAL** | **2,800** | **1,600** | **1,700** | **6,100** |

---

## 4. Verification Assertions & Quality Assurance

1. **Zero Session Leakage**:  
   - $\text{Train} \cap \text{Validation} = \emptyset$
   - $\text{Train} \cap \text{Test} = \emptyset$
   - $\text{Validation} \cap \text{Test} = \emptyset$
2. **Every Class Represented**: Confirmed. All 13 command classes have active representation across Train, Validation, and Test splits.
3. **Targeted `MIDDLE` Sessions**: New session `3529a937` adds training diversity to `train.csv` (raising `MIDDLE` training samples to 500 across 4 sessions), while `cf9b7593` provides an unseen test evaluation session for `MIDDLE` in `test.csv`.
4. **Data Integrity**: 0 duplicate rows, 0 non-finite values (0 NaN/Inf), 0 malformed columns across all 50 CSV files.
5. **No Model Training**: No models have been trained or modified in this phase.
