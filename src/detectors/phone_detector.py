from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np


@dataclass
class PhoneDetection:
    box: tuple[int, int, int, int]
    confidence: float


class PhoneDetector:
    """Detects cell phones using YOLOv4-tiny (COCO) via OpenCV DNN."""

    _COCO_LABEL_PHONE = "cell phone"

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.40,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        model_dir = Path(__file__).resolve().parents[2] / "models" / "yolo"
        model_dir.mkdir(parents=True, exist_ok=True)

        self.cfg_path = model_dir / "yolov4-tiny.cfg"
        self.weights_path = model_dir / "yolov4-tiny.weights"
        self.names_path = model_dir / "coco.names"

        self._ensure_model_files()

        self.labels = self._read_labels(self.names_path)
        if self._COCO_LABEL_PHONE not in self.labels:
            raise RuntimeError("COCO labels do not contain 'cell phone'.")
        self.phone_class_id = self.labels.index(self._COCO_LABEL_PHONE)

        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame: np.ndarray) -> list[PhoneDetection]:
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(416, 416),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)

        boxes, confidences = self._extract_phone_candidates(outputs, width=width, height=height)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        detections: list[PhoneDetection] = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                x, y, w, h = boxes[int(idx)]
                detections.append(
                    PhoneDetection(
                        box=(x, y, w, h),
                        confidence=confidences[int(idx)],
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _extract_phone_candidates(
        self,
        outputs,
        width: int,
        height: int,
    ) -> tuple[list[list[int]], list[float]]:
        boxes: list[list[int]] = []
        confidences: list[float] = []

        for output in outputs:
            for detection in output:
                candidate = self._candidate_from_detection(detection, width=width, height=height)
                if candidate is None:
                    continue
                box, confidence = candidate
                boxes.append(box)
                confidences.append(confidence)

        return boxes, confidences

    def _candidate_from_detection(
        self,
        detection,
        width: int,
        height: int,
    ) -> tuple[list[int], float] | None:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if class_id != self.phone_class_id or confidence < self.confidence_threshold:
            return None

        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        x = max(0, center_x - (w // 2))
        y = max(0, center_y - (h // 2))
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            return None

        return [x, y, w, h], confidence

    def _ensure_model_files(self) -> None:
        files_to_download = [
            (
                self.cfg_path,
                "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            ),
            (
                self.weights_path,
                "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
            ),
            (
                self.names_path,
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            ),
        ]

        for file_path, url in files_to_download:
            if file_path.exists() and file_path.stat().st_size > 0:
                continue
            try:
                urlretrieve(url, str(file_path))
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to download phone detection model files. "
                    "Please check internet access and rerun."
                ) from exc

    @staticmethod
    def _read_labels(names_path: Path) -> list[str]:
        return [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
