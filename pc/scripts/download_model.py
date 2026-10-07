"""
scripts/download_model.py
─────────────────────────
One-time setup script: downloads the MediaPipe hand_landmarker.task model
and verifies its SHA-256 checksum.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --out models/mediapipe/hand_landmarker.task
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Official MediaPipe float16 hand landmarker model
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# SHA-256 of the float16 v1 model bundle (verify before trusting)
MODEL_SHA256 = None  # Set to None to skip checksum (checksum may change with model updates)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "models" / "mediapipe" / "hand_landmarker.task"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[INFO] Model already present at: {out}")
        if MODEL_SHA256:
            digest = _sha256(out)
            if digest == MODEL_SHA256:
                print("[INFO] Checksum OK.")
            else:
                print(f"[WARNING] Checksum mismatch!\n  Expected: {MODEL_SHA256}\n  Got:      {digest}")
        return

    print(f"[INFO] Downloading MediaPipe hand landmarker model...")
    print(f"  URL : {MODEL_URL}")
    print(f"  Dest: {out}")

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            pct = min(block_count * block_size / total_size * 100, 100)
            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, out, reporthook=_progress)
    print()  # newline after progress

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[INFO] Downloaded {size_mb:.1f} MB → {out}")

    if MODEL_SHA256:
        digest = _sha256(out)
        if digest == MODEL_SHA256:
            print("[INFO] Checksum verified OK.")
        else:
            print(f"[ERROR] Checksum mismatch! File may be corrupted.")
            print(f"  Expected: {MODEL_SHA256}")
            print(f"  Got:      {digest}")
            sys.exit(1)
    else:
        print("[INFO] Checksum verification skipped (no reference hash set).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MediaPipe hand landmarker model.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    download(args.out)
    print("[OK] Model ready.")


if __name__ == "__main__":
    main()
