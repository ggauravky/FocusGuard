from __future__ import annotations

import argparse
import time

import cv2

from src.detectors import HaarFeatureDetector
from src.utils.drawing import draw_labeled_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face, eyes, smile, and lips detection with OpenCV"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = HaarFeatureDetector()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.detect(frame)

        for box in result.faces:
            draw_labeled_box(frame, box, "Face", (50, 220, 50))
        for box in result.eyes:
            draw_labeled_box(frame, box, "Eye", (220, 120, 50))
        for box in result.smiles:
            draw_labeled_box(frame, box, "Smile", (50, 200, 255))
        for box in result.lips:
            draw_labeled_box(frame, box, "Lips", (255, 80, 150))

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("FocusGuard - Face Feature Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
