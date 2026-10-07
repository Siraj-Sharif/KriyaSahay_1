"""Live USB Webcam Diagnostic Tool for HaGRID Model Verification.

This diagnostic tool runs live full-frame inference using the official HaGRID
model on video frames captured from the default webcam.

IMPORTANT SAFETY RULES:
- DIAGNOSTIC ONLY.
- Does NOT send serial commands.
- Does NOT interact with ESP32 or Webots.
- Does NOT modify legacy Extra Trees or production command pipeline.

Usage:
    python pc/tools/hagrid_webcam_diagnostic.py [--camera 0] [--model ResNet18] [--threshold 0.5]
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


def run_webcam_diagnostic(camera_index: int = 0, model_name: str = "ResNet18", threshold: float = 0.5):
    """Run real-time webcam diagnostic with HaGRID classifier."""
    print("=" * 70)
    print(" NeuroGrip Phase 1 — HaGRID Live Webcam Diagnostic")
    print("=" * 70)
    print("Press 'q' or 'ESC' to exit the diagnostic window.")
    print("=" * 70)

    config = HaGRIDConfig(model_name=model_name, confidence_threshold=threshold)
    classifier = HaGRIDClassifier(config=config, auto_load=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open USB webcam at index {camera_index}.")
        print("    Please verify webcam connection and availability.")
        sys.exit(1)

    print(f"[*] Webcam initialized successfully (Index: {camera_index})")
    print(f"[*] Running live full-frame inference using {model_name} on {classifier.device}...")

    fps_count = 0
    t_start = time.perf_counter()
    fps = 0.0

    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[!] Error reading frame from webcam. Exiting loop.")
                break

            # Run HaGRID inference on live frame (OpenCV BGR format)
            result = classifier.predict(frame, is_bgr=True, top_k=3)

            raw_class = result["raw_class"]
            confidence = result["confidence"]
            mapped_cmd = result["mapped_command"]
            latency_ms = result["latency_ms"]

            # Evaluate confidence against provisional threshold
            is_confident = confidence >= threshold
            display_cmd = mapped_cmd if is_confident else f"NO_COMMAND (Low Conf {confidence*100:.1f}%)"

            # Calculate FPS over 10-frame rolling window
            fps_count += 1
            if fps_count % 10 == 0:
                t_now = time.perf_counter()
                fps = 10.0 / (t_now - t_start)
                t_start = t_now

            # Render Overlay on Diagnostic Display Window
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), -1)

            # Header & Hardware Specs
            cv2.putText(
                frame,
                f"NeuroGrip HaGRID Phase 1 Diagnostic [{model_name} | {classifier.device.type.upper()}]",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            # Raw Class & Confidence
            color_raw = (0, 255, 255) if is_confident else (128, 128, 128)
            cv2.putText(
                frame,
                f"Raw HaGRID: {raw_class} ({confidence*100:.1f}%)",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_raw,
                2,
            )

            # Mapped NeuroGrip Command
            color_cmd = (0, 255, 0) if (is_confident and mapped_cmd != "NO_COMMAND") else (0, 0, 255)
            cv2.putText(
                frame,
                f"Mapped Cmd: {display_cmd}",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_cmd,
                2,
            )

            # Latency and FPS metrics
            metrics_str = f"Latency: {latency_ms:4.1f} ms | FPS: {fps:4.1f} | Resolution: {w}x{h}"
            cv2.putText(
                frame,
                metrics_str,
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            # Top-3 predictions panel at bottom
            y_offset = h - 60
            cv2.rectangle(frame, (0, h - 70), (w, h), (10, 10, 10), -1)
            top_str = " | ".join(
                [f"{p['class']}: {p['probability']*100:.1f}% -> {p['mapped_command']}" for p in result["top_k"]]
            )
            cv2.putText(
                frame,
                f"Top 3: {top_str}",
                (15, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            cv2.imshow("NeuroGrip Phase 1 — HaGRID Webcam Diagnostic", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("[*] User exit requested. Closing webcam diagnostic.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Webcam released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="HaGRID Live Webcam Diagnostic GUI")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("-m", "--model", type=str, default="ResNet18", choices=["ResNet18", "MobileNetV3_large"], help="Model architecture")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Provisional confidence threshold")
    args = parser.parse_args()

    run_webcam_diagnostic(camera_index=args.camera, model_name=args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()
