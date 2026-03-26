from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    eyes: list[tuple[int, int, int, int]] = field(default_factory=list)
    smiles: list[tuple[int, int, int, int]] = field(default_factory=list)
    lips: list[tuple[int, int, int, int]] = field(default_factory=list)


class HaarFeatureDetector:
    def __init__(self) -> None:
        haar_dir = Path(cv2.data.haarcascades)

        self.face_cascade = cv2.CascadeClassifier(
            str(haar_dir / "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            str(haar_dir / "haarcascade_eye_tree_eyeglasses.xml")
        )
        self.smile_cascade = cv2.CascadeClassifier(
            str(haar_dir / "haarcascade_smile.xml")
        )

        if self.face_cascade.empty() or self.eye_cascade.empty() or self.smile_cascade.empty():
            raise RuntimeError("Failed to load one or more Haar cascade models.")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        result = DetectionResult()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            result.faces.append((x, y, w, h))
            face_gray = gray[y : y + h, x : x + w]
            face_bgr = frame[y : y + h, x : x + w]

            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(20, 20),
            )
            for (ex, ey, ew, eh) in eyes:
                result.eyes.append((x + ex, y + ey, ew, eh))

            smiles = self.smile_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(25, 25),
            )
            for (sx, sy, sw, sh) in smiles:
                result.smiles.append((x + sx, y + sy, sw, sh))

            lips_box = self._detect_lips_in_face(face_bgr)
            if lips_box is not None:
                lx, ly, lw, lh = lips_box
                result.lips.append((x + lx, y + ly, lw, lh))

        return result

    def _detect_lips_in_face(self, face_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        height, width = face_bgr.shape[:2]
        lower_start = int(height * 0.55)
        lower_face = face_bgr[lower_start:, :]

        if lower_face.size == 0:
            return None

        blurred = cv2.GaussianBlur(lower_face, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 40, 40], dtype=np.uint8)
        upper_red_1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([160, 40, 40], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < (width * height) * 0.002:
            return None

        lx, ly, lw, lh = cv2.boundingRect(contour)
        return (lx, ly + lower_start, lw, lh)
