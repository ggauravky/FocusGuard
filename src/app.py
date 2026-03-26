from __future__ import annotations

import argparse
import time

import cv2
try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

from src.attention import AttentionState, PhoneAttentionMonitor
from src.detectors import HaarFeatureDetector, PhoneDetection, PhoneDetector
from src.utils.drawing import draw_labeled_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FocusGuard: detect phone-looking behavior and alert after sustained 4+ seconds"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument(
        "--phone-threshold",
        type=float,
        default=4.0,
        help="Seconds of continuous phone-looking before warning/alarm",
    )
    parser.add_argument(
        "--phone-confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold for phone detection",
    )
    return parser.parse_args()


def trigger_alarm() -> None:
    if winsound is not None:
        winsound.Beep(1300, 350)
        winsound.Beep(1600, 250)
    else:
        print("\aALERT: Looking at phone for too long")


def pick_primary_face(
    faces: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def pick_primary_phone(
    phone_detections: list[PhoneDetection],
) -> PhoneDetection | None:
    if not phone_detections:
        return None
    return max(phone_detections, key=lambda d: d.confidence)


def draw_feature_boxes(result, frame, phone_detection: PhoneDetection | None) -> None:
    for box in result.faces:
        draw_labeled_box(frame, box, "Face", (50, 220, 50))
    for box in result.eyes:
        draw_labeled_box(frame, box, "Eye", (220, 120, 50))
    for box in result.smiles:
        draw_labeled_box(frame, box, "Smile", (50, 200, 255))
    for box in result.lips:
        draw_labeled_box(frame, box, "Lips", (255, 80, 150))
    if phone_detection is not None:
        draw_labeled_box(
            frame,
            phone_detection.box,
            f"Phone {phone_detection.confidence:.2f}",
            (20, 20, 255),
        )


def draw_attention_overlay(frame, attention_state: AttentionState, threshold_seconds: float) -> None:
    status_color = (60, 220, 80) if attention_state.is_looking_phone else (70, 150, 255)
    if attention_state.looking_seconds >= threshold_seconds:
        status_color = (40, 40, 255)

    cv2.putText(
        frame,
        (
            f"Phone timer: {attention_state.looking_seconds:.1f}s"
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
    face_detector = HaarFeatureDetector()
    phone_detector = PhoneDetector(confidence_threshold=args.phone_confidence)
    attention_monitor = PhoneAttentionMonitor(
        threshold_seconds=args.phone_threshold,
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

        result = face_detector.detect(frame)
        phone_detections = phone_detector.detect(frame)
        primary_phone = pick_primary_phone(phone_detections)
        draw_feature_boxes(result, frame, primary_phone)

        primary_face = pick_primary_face(result.faces)

        attention_state = attention_monitor.update(
            now=time.perf_counter(),
            face_box=primary_face,
            phone_box=primary_phone.box if primary_phone is not None else None,
        )

        if attention_state.should_alert:
            trigger_alarm()
        draw_attention_overlay(frame, attention_state, args.phone_threshold)

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

        cv2.imshow("FocusGuard - Phone Attention Warning", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
