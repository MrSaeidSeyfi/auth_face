from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .database import FaceDB
from .embedding import SigLIPEmbedder
from .utils import extract_roi


@dataclass
class DetectionResult:
    label: str
    embedding: np.ndarray
    roi: np.ndarray
    similarity: float
    bbox: Tuple[int, int, int, int]


@dataclass
class FaceRecognitionSettings:
    threshold: float = 0.85
    camera_index: int = 0
    max_faces: int = 3
    detection_confidence: float = 0.5
    display_window: str = "Face Recognition"
    padding: int = 32
    detector_stride: int = 2
    nms_threshold: float = 0.3


class FaceRecognitionApp:
    """
    High-level orchestration of the face recognition pipeline.
    """

    def __init__(
        self,
        db: Optional[FaceDB] = None,
        embedder: Optional[SigLIPEmbedder] = None,
        settings: Optional[FaceRecognitionSettings] = None,
    ) -> None:
        self.db = db or FaceDB()
        self.embedder = embedder or SigLIPEmbedder()
        self.settings = settings or FaceRecognitionSettings()
        self.capture = cv2.VideoCapture(self.settings.camera_index)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.settings.detection_confidence,
        )
        self._faces: Dict[Tuple[int, int, int, int], DetectionResult] = {}
        self._frame_count = 0
        self._last_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []

    def run(self) -> None:
        print("s=save | l=list | d=delete | +/-=threshold | q=quit")
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            self._frame_count += 1
            self._process_frame(frame)
            self._overlay_detections(frame)
            cv2.putText(
                frame,
                f"T:{self.settings.threshold:.2f} DB:{len(self.db.cache)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.settings.display_window, frame)

            key = cv2.waitKey(1) & 0xFF
            if self._handle_input(key):
                break

        self.stop()

    def stop(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()
        self.detector.close()
        self.db.close()

    def _handle_input(self, key: int) -> bool:
        if key in (27, ord("q")):
            return True
        if key == ord("s"):
            self._save_new_face()
        elif key == ord("l"):
            self._print_db_records()
        elif key == ord("d"):
            self._delete_record()
        elif key == ord("+"):
            self.settings.threshold = min(0.95, self.settings.threshold + 0.05)
            print(f"T:{self.settings.threshold:.2f}")
        elif key == ord("-"):
            self.settings.threshold = max(0.50, self.settings.threshold - 0.05)
            print(f"T:{self.settings.threshold:.2f}")
        return False

    def _process_frame(self, frame: np.ndarray) -> None:
        if self._frame_count % self.settings.detector_stride == 0 or not self._last_boxes:
            self._last_boxes = self._detect_faces(frame)
        if not self._last_boxes:
            return

        self._faces.clear()
        for bbox, score in self._last_boxes:
            roi, packed_bbox = extract_roi(frame, bbox)
            if roi.size == 0 or score < self.settings.detection_confidence:
                continue

            embedding = self.embedder.embed(roi)
            name, similarity = self.db.match(embedding, self.settings.threshold)
            top, right, bottom, left = packed_bbox[1], packed_bbox[2], packed_bbox[3], packed_bbox[0]
            rect = (top, right, bottom, left)
            label = name or "Unknown"
            self._faces[rect] = DetectionResult(
                label=label,
                embedding=embedding,
                roi=roi,
                similarity=similarity,
                bbox=rect,
            )

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        if not result.detections:
            return []
        height, width = frame.shape[:2]
        detections = []
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            score = detection.score[0] if detection.score else 0.0
            x1 = max(0, int(bbox.xmin * width))
            y1 = max(0, int(bbox.ymin * height))
            x2 = min(width, int((bbox.xmin + bbox.width) * width))
            y2 = min(height, int((bbox.ymin + bbox.height) * height))
            pad = self.settings.padding
            packed = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(width, x2 + pad),
                min(height, y2 + pad),
            )
            detections.append((packed, score))
        return self._nms(detections)

    def _nms(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if not detections:
            return []
        detections.sort(key=lambda item: item[1], reverse=True)
        kept: List[Tuple[Tuple[int, int, int, int], float]] = []
        for bbox, score in detections:
            if score < self.settings.detection_confidence:
                continue
            if all(self._iou(bbox, prev[0]) <= self.settings.nms_threshold for prev in kept):
                kept.append((bbox, score))
            if len(kept) >= self.settings.max_faces:
                break
        return kept

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union else 0.0

    def _overlay_detections(self, frame: np.ndarray) -> None:
        for (top, right, bottom, left), detection in self._faces.items():
            color = (0, 255, 0) if detection.label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            if detection.label != "Unknown":
                text = f"{detection.label} ({detection.similarity:.2f})"
            else:
                text = "Unknown"

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (left, bottom - 35), (left + text_size[0] + 12, bottom), color, -1)
            cv2.putText(
                frame,
                text,
                (left + 6, bottom - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

    def _save_new_face(self) -> None:
        unknown_face = next((face for face in self._faces.values() if face.label == "Unknown"), None)
        if not unknown_face:
            return

        name = input("Name: ").strip()
        if name:
            self.db.add(name, unknown_face.embedding, unknown_face.roi)

    def _print_db_records(self) -> None:
        faces = self.db.list()
        print(f"\n{'=' * 60}\nFACES ({len(self.db.cache)})\n{'=' * 60}")
        for face in faces:
            print(f"ID:{face.id:3d} | {face.name:20s} | {face.created_at}")
        print("=" * 60)

    def _delete_record(self) -> None:
        try:
            face_id = int(input("ID: "))
        except ValueError:
            return
        self.db.delete(face_id)

