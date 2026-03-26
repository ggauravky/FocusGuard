from __future__ import annotations

import argparse
import time

import cv2
try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

from src.attention import ScreenAttentionMonitor
from src.detectors import HaarFeatureDetector
from src.utils.drawing import draw_labeled_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FocusGuard: face feature detection + continuous screen attention warning"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=3.5,
        help="Seconds of continuous attention before warning/alarm",
    )
    parser.add_argument(
        "--center-ratio",
        type=float,
        default=0.62,
        help="Center window ratio used to decide if face is aligned to screen",
    )
    return parser.parse_args()


def trigger_alarm() -> None:
    if winsound is not None:
        winsound.Beep(1300, 350)
        winsound.Beep(1600, 250)
    else:
        print("\aALERT: Continuous screen attention detected")


def pick_primary_face(
    faces: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def count_eyes_in_face(
    face_box: tuple[int, int, int, int] | None,
    eyes: list[tuple[int, int, int, int]],
) -> int:
    if face_box is None:
        return 0

    x, y, w, h = face_box
    count = 0
    for (ex, ey, ew, eh) in eyes:
        eye_cx = ex + (ew / 2.0)
        eye_cy = ey + (eh / 2.0)
        if x <= eye_cx <= (x + w) and y <= eye_cy <= (y + h):
            count += 1
    return count


def draw_feature_boxes(result, frame) -> None:
    for box in result.faces:
        draw_labeled_box(frame, box, "Face", (50, 220, 50))
    for box in result.eyes:
        draw_labeled_box(frame, box, "Eye", (220, 120, 50))
    for box in result.smiles:
        draw_labeled_box(frame, box, "Smile", (50, 200, 255))
    for box in result.lips:
        draw_labeled_box(frame, box, "Lips", (255, 80, 150))


def draw_attention_overlay(frame, attention_state, threshold_seconds: float) -> None:
    status_color = (60, 220, 80) if attention_state.is_attentive else (70, 150, 255)
    if attention_state.attentive_seconds >= threshold_seconds:
        status_color = (40, 40, 255)

    cv2.putText(
        frame,
        (
            f"Focus timer: {attention_state.attentive_seconds:.1f}s"
            f" / {threshold_seconds:.1f}s"
        ),
        (15, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )
    cv2.putText(
        frame,
        f"State: {attention_state.reason}",
        (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )


def main() -> None:
    args = parse_args()
    detector = HaarFeatureDetector()
    attention_monitor = ScreenAttentionMonitor(
        threshold_seconds=args.attention_threshold,
        center_ratio=args.center_ratio,
    )

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
        draw_feature_boxes(result, frame)

        primary_face = pick_primary_face(result.faces)
        eye_count = count_eyes_in_face(primary_face, result.eyes)

        attention_state = attention_monitor.update(
            now=time.perf_counter(),
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            face_box=primary_face,
            eye_count_in_face=eye_count,
        )

        if attention_state.should_alert:
            trigger_alarm()
        draw_attention_overlay(frame, attention_state, args.attention_threshold)

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

        cv2.imshow("FocusGuard - Screen Attention Warning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
