import csv
import glob
import os
import math
from collections import Counter

files = sorted(glob.glob("data/raw_v2_pilot/dataset_v2_*.csv"))

expected_classes = {
    "INDEX", "MIDDLE", "RING", "PINKY", "THUMB_ONLY",
    "TWO_FINGER", "THREE_FINGER", "INDEX_PINKY",
    "FOUR_FINGERS", "CLOSE", "GRAB", "REST", "STOP"
}

all_rows = []
errors = []

for filepath in files:
    name = os.path.basename(filepath)

    with open(filepath, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 100:
        errors.append(f"{name}: expected 100 rows, got {len(rows)}")

    if rows:
        columns = list(rows[0].keys())

        if len(columns) != 72:
            errors.append(f"{name}: expected 72 columns, got {len(columns)}")

        labels = {r["label"] for r in rows}
        sessions = {r["session_id"] for r in rows}
        sample_ids = [r["sample_id"] for r in rows]

        if len(labels) != 1:
            errors.append(f"{name}: multiple labels {labels}")

        if len(sessions) != 1:
            errors.append(f"{name}: multiple sessions {sessions}")

        expected_ids = [str(i) for i in range(1, 101)]
        if sorted(sample_ids, key=int) != expected_ids:
            errors.append(f"{name}: sample IDs are not exactly 1-100")

        feature_rows = []
        finite = True

        for r in rows:
            try:
                values = [float(r[f"feature_{i}"]) for i in range(68)]
                feature_rows.append(tuple(values))

                if not all(math.isfinite(x) for x in values):
                    finite = False

            except Exception:
                finite = False

        if not finite:
            errors.append(f"{name}: non-finite or invalid feature value")

        duplicates = len(feature_rows) - len(set(feature_rows))

        if duplicates:
            errors.append(f"{name}: {duplicates} duplicate feature rows")

        all_rows.extend(rows)

print("=" * 70)
print("NeuroGrip v2 DATASET QA")
print("=" * 70)

print(f"CSV files       : {len(files)}")
print(f"Total samples   : {len(all_rows)}")

labels = Counter(r["label"] for r in all_rows)
sessions = Counter(r["session_id"] for r in all_rows)
hands = Counter(r["handedness"] for r in all_rows)

print("\nCLASS COUNTS")
for label in sorted(labels):
    print(f"{label:16} : {labels[label]}")

print("\nHANDEDNESS")
for hand, count in hands.items():
    print(f"{hand:16} : {count}")

print("\nSESSIONS")
print(f"Unique sessions : {len(sessions)}")
print(f"Samples/session : {Counter(sessions.values())}")

print("\nEXPECTED")
print(f"Classes         : {len(expected_classes)}")
print("Samples/class   : 300")
print("Samples/session : 100")
print("Features        : 68")
print("Columns         : 72")

missing = expected_classes - set(labels)
unexpected = set(labels) - expected_classes

print("\nCLASS VALIDATION")
print("Missing classes :", missing if missing else "NONE")
print("Unexpected      :", unexpected if unexpected else "NONE")

print("\nERRORS")
if errors:
    for error in errors:
        print("❌", error)
else:
    print("NONE")

print("\nFINAL VERDICT")

if (
    len(files) == 39
    and len(all_rows) == 3900
    and not missing
    and not unexpected
    and not errors
    and all(count == 300 for count in labels.values())
    and len(sessions) == 39
    and all(count == 100 for count in sessions.values())
):
    print("✅ PASS — v2 raw dataset structure is clean.")
else:
    print("❌ REVIEW REQUIRED — see errors above.")

print("=" * 70)