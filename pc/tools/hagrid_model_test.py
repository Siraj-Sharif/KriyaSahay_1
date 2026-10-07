"""Isolated Single-Image / Synthetic Model Test Tool for HaGRID Verification.

Usage:
    python pc/tools/hagrid_model_test.py [--image PATH_TO_IMAGE] [--model ResNet18|MobileNetV3_large]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure pc/src is on sys.path
pc_src = Path(__file__).resolve().parent.parent / "src"
if str(pc_src) not in sys.path:
    sys.path.insert(0, str(pc_src))

from neurogrip.hagrid import HaGRIDClassifier, HaGRIDConfig, map_hagrid_to_neurogrip


def create_synthetic_test_image() -> np.ndarray:
    """Create a synthetic test image with a mock hand-like shape on neutral background."""
    img = np.full((480, 640, 3), 144, dtype=np.uint8)  # Neutral background
    # Draw simple geometric shapes representing a test pattern
    cv2.circle(img, (320, 240), 80, (200, 180, 150), -1)  # Palm circle
    cv2.rectangle(img, (300, 100), (340, 200), (200, 180, 150), -1)  # Finger rect
    cv2.putText(
        img,
        "SYNTHETIC TEST INPUT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    return img


def run_model_test(image_path: str = None, model_name: str = "ResNet18", iterations: int = 100):
    """Run model loading and inference benchmarks."""
    print("=" * 70)
    print(" NeuroGrip Phase 1 — Isolated HaGRID Model Test")
    print("=" * 70)

    config = HaGRIDConfig(model_name=model_name)
    print(f"[*] Target Architecture: {config.model_name}")
    print(f"[*] Checkpoint Path:     {config.checkpoint_path}")
    print(f"[*] Expected Resolution: {config.img_size}x{config.img_size}")
    print(f"[*] Mean Normalization:  {config.img_mean}")
    print(f"[*] Std Normalization:   {config.img_std}")

    t_load_start = time.perf_counter()
    classifier = HaGRIDClassifier(config=config, auto_load=True)
    t_load_end = time.perf_counter()
    load_time_ms = (t_load_end - t_load_start) * 1000.0

    print(f"[*] Load Time:           {load_time_ms:.2f} ms")
    print(f"[*] Execution Device:    {classifier.device}")

    # Load or generate test image
    if image_path and Path(image_path).exists():
        print(f"[*] Input Image Source:  Real image file ({image_path})")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] Error: Could not read image file {image_path}. Falling back to synthetic.")
            image = create_synthetic_test_image()
            is_synthetic = True
        else:
            is_synthetic = False
    else:
        print("[*] Input Image Source:  SYNTHETIC TEST PATTERN (No image path provided)")
        print("    (NOTE: Synthetic image is used strictly for pipeline & latency validation, NOT accuracy!)")
        image = create_synthetic_test_image()
        is_synthetic = True

    # Validate Preprocessing & Input Tensor Shape
    tensor = classifier.preprocess(image, is_bgr=True)
    print(f"[*] Preprocessed Tensor: Shape={list(tensor.shape)}, Dtype={tensor.dtype}, Device={tensor.device}")

    # Warmup iterations
    for _ in range(5):
        _ = classifier.predict(image, is_bgr=True)

    # Benchmark Latency over N iterations
    latencies = []
    for _ in range(iterations):
        res = classifier.predict(image, is_bgr=True)
        latencies.append(res["latency_ms"])

    avg_lat = np.mean(latencies)
    med_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    std_lat = np.std(latencies)

    print("\n" + "-" * 70)
    print(" INFERENCE RESULTS & TOP-5 PREDICTIONS")
    print("-" * 70)
    res = classifier.predict(image, is_bgr=True, top_k=5)
    print(f"[*] Top-1 Raw HaGRID Class:  '{res['raw_class']}' (Index: {res['raw_index']})")
    print(f"[*] Top-1 Confidence:        {res['confidence'] * 100:.2f}%")
    print(f"[*] Mapped NeuroGrip Cmd:   '{res['mapped_command']}'")

    print("\n[*] Top-5 Class Probability Distribution:")
    for rank, p in enumerate(res["top_k"], start=1):
        print(
            f"    {rank}. Class: {p['class']:<16} | Mapped: {p['mapped_command']:<15} | Prob: {p['probability']*100:6.2f}%"
        )

    print("\n" + "-" * 70)
    print(" LATENCY BENCHMARK PERFORMANCE (CPU / SINGLE FRAME)")
    print("-" * 70)
    print(f"[*] Warm-Up / Iterations:   5 warmup / {iterations} timed runs")
    print(f"[*] Average Latency:        {avg_lat:.2f} ms")
    print(f"[*] Median Latency:         {med_lat:.2f} ms")
    print(f"[*] P95 Latency:            {p95_lat:.2f} ms")
    print(f"[*] Standard Deviation:     {std_lat:.2f} ms")
    print(f"[*] Theoretical Max FPS:    {1000.0 / avg_lat:.1f} FPS")
    print("=" * 70)

    if is_synthetic:
        print("\n[!] REMINDER: Synthetic test input validates execution pipeline and latency only.")
        print("    To test real gesture predictions, pass a real image: python pc/tools/hagrid_model_test.py --image my_hand.jpg")

    return res, latencies


def main():
    parser = argparse.ArgumentParser(description="HaGRID Pretrained Model Isolated Diagnostic Test")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to input test image")
    parser.add_argument("-m", "--model", type=str, default="ResNet18", choices=["ResNet18", "MobileNetV3_large"], help="Model architecture")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of latency benchmark iterations")
    args = parser.parse_args()

    run_model_test(image_path=args.image, model_name=args.model, iterations=args.iterations)


if __name__ == "__main__":
    main()
