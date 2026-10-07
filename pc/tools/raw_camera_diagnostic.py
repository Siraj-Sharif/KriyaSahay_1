"""Minimal Raw Camera Diagnostic Tool for NeuroGrip.

Tests raw OpenCV video capture from USB webcam (Index 0) at 1280x720 @ 30 FPS
WITHOUT loading MediaPipe, HaGRID, Extra Trees, or any downstream CV pipeline processing.

Controls:
  Q / ESC : Exit raw camera diagnostic
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2


def run_raw_camera_diagnostic(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Raw Camera Diagnostic Tool.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Target width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Target height (default: 720).")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30).")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames before exit.")
    args = parser.parse_args(argv)

    print("==========================================================================")
    print(" NEUROGRIP — RAW CAMERA DIAGNOSTIC TOOL")
    print("==========================================================================")
    print(f"  Target Camera Index : {args.camera}")
    print(f"  Target Resolution   : {args.width}x{args.height} @ {args.fps} FPS")
    print("==========================================================================")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open cv2.VideoCapture({args.camera}). Device locked or unavailable.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    act_fps = float(cap.get(cv2.CAP_PROP_FPS))

    print(f"[INFO] Camera {args.camera} opened successfully.")
    print(f"       Actual Hardware Parameters: {act_w}x{act_h} @ {act_fps:.1f} FPS")

    window_name = "NeuroGrip — Raw Camera Diagnostic"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, act_w, act_h)

    frame_count = 0
    fail_count = 0
    start_t = time.time()
    last_fps_t = start_t
    fps_history = []
    current_fps = act_fps

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            t1 = time.time()

            dt = t1 - last_fps_t
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_history.append(inst_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                current_fps = sum(fps_history) / len(fps_history)
            last_fps_t = t1

            if not ret or frame is None or frame.size == 0:
                fail_count += 1
                print(f"[WARNING F{frame_count:05d}] cap.read() returned False / Empty frame (Total read failures: {fail_count})")
                time.sleep(0.033)
                continue

            frame_count += 1

            # Render HUD on raw camera frame
            hud_txt = f"RAW CAMERA {args.camera} | {act_w}x{act_h} | LIVE FPS: {current_fps:.1f} | FRAMES: {frame_count} | FAILS: {fail_count}"
            cv2.putText(frame, hud_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("[INFO] User pressed Quit key. Exiting raw camera loop.")
                break

            if args.max_frames and frame_count >= args.max_frames:
                print(f"[INFO] Reached max frames limit ({args.max_frames}). Exiting.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by KeyboardInterrupt.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_t
    print("\n--------------------------------------------------------------------------")
    print(f"  Raw Camera Summary: {frame_count} frames captured in {total_time:.2f} s ({fail_count} read failures)")
    print(f"  Average Rate      : {(frame_count / total_time):.1f} FPS" if total_time > 0 else "  Average Rate: 0 FPS")
    print("==========================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(run_raw_camera_diagnostic())
