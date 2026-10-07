# NeuroGrip Phase 10H — Targeted Generalization Data Collection Instructions

## Overview

Phase 10H collects **1 new independent session per target class** (200 valid samples/session $\times$ 6 target classes = **1,200 total new samples**).

The purpose of Phase 10H is to introduce controlled physical and spatial variations during recording to eliminate single-session posture rigidity and improve cross-session generalization.

---

## Target Classes & Scope

| Priority | Target Class | Target Samples | New Sessions | Target Directory |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `FOUR_FINGERS` | 200 | 1 | `pc/data/raw_v2_phase10h` |
| 2 | `STOP` | 200 | 1 | `pc/data/raw_v2_phase10h` |
| 3 | `TWO_FINGER` | 200 | 1 | `pc/data/raw_v2_phase10h` |
| 4 | `THREE_FINGER` | 200 | 1 | `pc/data/raw_v2_phase10h` |
| 5 | `MIDDLE` | 200 | 1 | `pc/data/raw_v2_phase10h` |
| 6 | `INDEX_PINKY` | 200 | 1 | `pc/data/raw_v2_phase10h` |

---

## Human Operator Checklist: Intended Gesture Variations

During each 200-sample recording session, the operator should deliberately introduce the following controlled variations:

1. **Horizontal & Vertical Position Shift**:
   - Move hand across left, center, right, upper, and lower regions of the camera field of view.
2. **Camera Distance Variation**:
   - Vary distance to camera from close ($\sim 30\text{ cm}$) to medium ($\sim 50\text{ cm}$) to far ($\sim 70\text{ cm}$).
3. **Wrist Orientation & Pitch/Yaw Tilt**:
   - Vary wrist angle slightly ($\pm 15^\circ$ tilt/roll) to break rigid 2D coordinate alignment.
4. **Natural Finger Spacing Variation**:
   - Allow natural micro-variations in finger splay (slightly wider vs tighter spacing) while preserving the core gesture.
5. **Handedness Distribution**:
   - Record approximately 100 samples with Left hand and 100 samples with Right hand (or alternate mid-session) to prevent handedness bias.
6. **Strict Gesture Definition Retention**:
   - **Do NOT change the underlying gesture posture during a sample session.**
   - Avoid rapid twitching or transitioning between gestures while samples are actively being collected.

---

## Manual Execution CLI Commands

Run each command sequentially in your terminal:

```bash
# 1. FOUR_FINGERS (200 samples)
python pc/scripts/collect_dataset.py --label FOUR_FINGERS --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2

# 2. STOP (200 samples)
python pc/scripts/collect_dataset.py --label STOP --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2

# 3. TWO_FINGER (200 samples)
python pc/scripts/collect_dataset.py --label TWO_FINGER --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2

# 4. THREE_FINGER (200 samples)
python pc/scripts/collect_dataset.py --label THREE_FINGER --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2

# 5. MIDDLE (200 samples)
python pc/scripts/collect_dataset.py --label MIDDLE --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2

# 6. INDEX_PINKY (200 samples)
python pc/scripts/collect_dataset.py --label INDEX_PINKY --samples 200 --output-dir pc/data/raw_v2_phase10h --feature-version v2
```

---

## Post-Collection Verification

After completing collection, run the Phase 10H QA verification script:

```bash
python pc/qa_v2_phase10h.py --data-dir pc/data/raw_v2_phase10h
```

This will verify 68-D schema compliance, feature finiteness, unique session IDs, sample counts, and generate session summary JSON artifacts.
