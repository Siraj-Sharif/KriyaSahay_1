"""Live USB Webcam Diagnostic GUI comparing Architecture A (Full-Frame) and Architecture B (Hand-Crop).

IMPORTANT SAFETY RULES:
- DIAGNOSTIC ONLY.
- Does NOT send serial commands.
- Does NOT interact with ESP32, Webots, or motor hardware.
- Strictly enforces multi-hand safety rule: >1 detected hands -> NO_COMMAND.

Usage:
    python pc/tools/hagrid_hand_crop_diagnostic.py [--camera 0] [--detector_conf 0.40] [--pad 0.15]
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

from neurogrip.hagrid import (
    HaGRIDClassifier,
    HaGRIDConfig,
    HaGRIDHandDetector,
    predict_architecture_a,
    predict_architecture_b,
)


def run_crop_diagnostic(
    camera_index: int = 0,
    model_name: str = "ResNet18",
    detector_conf: float = 0.40,
    pad_ratio: float = 0.15,
):
    """Run real-time USB webcam diagnostic GUI comparing Arch A and Arch B."""
    print("=" * 80)
    print(" NeuroGrip Phase 2 — Hand-Crop vs Full-Frame Webcam Diagnostic")
    print("=" * 80)
    print("Press 'm' to toggle display mode (Side-by-Side vs Single Overlay)")
    print("Press 'q' or 'ESC' to exit diagnostic.")
    print("=" * 80)

    config = HaGRIDConfig(
        model_name=model_name,
        detector_conf_threshold=detector_conf,
        crop_pad_ratio=pad_ratio,
    )

    classifier = HaGRIDClassifier(config=config, auto_load=True)
    detector = HaGRIDHandDetector(config=config, auto_load=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open USB webcam at index {camera_index}.")
        sys.exit(1)

    print(f"[*] Webcam initialized successfully (Index: {camera_index})")

    fps_count = 0
    t_start = time.perf_counter()
    fps_a = 0.0
    fps_b = 0.0

    side_by_side = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error reading webcam frame. Exiting.")
                break

            h, w, _ = frame.shape

            # Run Architecture A (Full Frame)
            res_a = predict_architecture_a(classifier, frame, is_bgr=True)

            # Run Architecture B (Hand Crop)
            res_b = predict_architecture_b(
                detector,
                classifier,
                frame,
                is_bgr=True,
                conf_threshold=detector_conf,
                pad_ratio=pad_ratio,
            )

            fps_count += 1
            if fps_count % 10 == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                fps_a = 10.0 / elapsed
                fps_b = 10.0 / elapsed
                t_start = t_now

            # Render Display Frames
            frame_a = frame.copy()
            frame_b = frame.copy()

            # Render Architecture A Visual Overlay
            cv2.rectangle(frame_a, (0, 0), (w, 110), (20, 20, 20), -1)
            cv2.putText(
                frame_a,
                "[ARCH A: FULL-FRAME BASELINE]",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame_a,
                f"Raw Class: {res_a['raw_class']} ({res_a['confidence']*100:.1f}%)",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )
            cmd_color_a = (0, 255, 0) if res_a["mapped_command"] != "NO_COMMAND" else (0, 0, 255)
            cv2.putText(
                frame_a,
                f"Mapped Cmd: {res_a['mapped_command']}",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                cmd_color_a,
                2,
            )
            cv2.putText(
                frame_a,
                f"Latency: {res_a['latency_ms']:4.1f} ms | FPS: {fps_a:4.1f}",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            # Render Architecture B Visual Overlay
            cv2.rectangle(frame_b, (0, 0), (w, 140), (20, 20, 20), -1)
            cv2.putText(
                frame_b,
                "[ARCH B: HAND-CROP PIPELINE]",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Render all detected bounding boxes
            hand_count = res_b["detected_hands_count"]
            boxes = res_b["detected_boxes"]

            if hand_count == 0:
                cv2.putText(
                    frame_b,
                    "Hands Detected: 0 -> NO_COMMAND (NO_HAND)",
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (128, 128, 128),
                    2,
                )
            elif hand_count == 1:
                box = boxes[0]
                crop_box = res_b["selected_crop_box"]
                # Draw detected box in green
                cv2.rectangle(frame_b, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw crop padded box in cyan
                if crop_box:
                    cv2.rectangle(frame_b, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255, 255, 0), 1)

                cv2.putText(
                    frame_b,
                    f"Raw Class: {res_b['raw_class']} ({res_b['confidence']*100:.1f}%)",
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                )
                cmd_color_b = (0, 255, 0) if res_b["mapped_command"] != "NO_COMMAND" else (0, 0, 255)
                cv2.putText(
                    frame_b,
                    f"Mapped Cmd: {res_b['mapped_command']}",
                    (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cmd_color_b,
                    2,
                )
            else:
                # Multi-hand case -> Draw ALL boxes in RED/ORANGE
                for idx, box in enumerate(boxes):
                    cv2.rectangle(frame_b, (box[0], box[1]), (box[2], box[3]), (0, 140, 255), 2)
                    cv2.putText(
                        frame_b,
                        f"Hand {idx+1}",
                        (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 140, 255),
                        1,
                    )

                cv2.putText(
                    frame_b,
                    f"Hands Detected: {hand_count} -> NO_COMMAND (MULTI-HAND AMBIGUITY)",
                    (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame_b,
                    "Mapped Cmd: NO_COMMAND (SAFETY REJECT)",
                    (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            bd = res_b["breakdown"]
            metrics_b = f"Detector: {bd['detector_ms']:4.1f}ms | Classifier: {bd['classifier_ms']:4.1f}ms | Total: {bd['total_ms']:4.1f}ms | FPS: {fps_b:4.1f}"
            cv2.putText(
                frame_b,
                metrics_b,
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            # Display Window Composition
            if side_by_side:
                # Resize both to half width for side-by-side view
                disp_w = 640
                disp_h = 480
                f_a_res = cv2.resize(frame_a, (disp_w, disp_h))
                f_b_res = cv2.resize(frame_b, (disp_w, disp_h))
                combined = np.hstack((f_a_res, f_b_res))
                cv2.imshow("NeuroGrip Phase 2 Diagnostic [Left: Arch A Full-Frame | Right: Arch B Hand-Crop]", combined)
            else:
                cv2.imshow("NeuroGrip Phase 2 Diagnostic [Arch B Hand-Crop View]", frame_b)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[*] Exit requested by user.")
                break
            elif key == ord("m"):
                side_by_side = not side_by_side
                cv2.destroyAllWindows()
                print(f"[*] Switched display mode. Side-by-side: {side_by_side}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Webcam released cleanly.")


def main():
    parser = argparse.ArgumentParser(description="HaGRID Phase 2 Live Webcam Diagnostic GUI")
    parser.add_argument("-c", "--camera", type=int, default=0, help="USB webcam index (default: 0)")
    parser.add_argument("-m", "--model", type=str, default="ResNet18", help="Classifier model name")
    parser.add_argument("-d", "--detector_conf", type=float, default=0.40, help="Detector threshold")
    parser.add_argument("-p", "--pad", type=float, default=0.15, help="Hand bounding box padding margin ratio")
    args = parser.parse_args()

    run_crop_diagnostic(
        camera_index=args.camera,
        model_name=args.model,
        detector_conf=args.detector_conf,
        pad_ratio=args.pad,
    )


if __name__ == "__main__":
    main()
