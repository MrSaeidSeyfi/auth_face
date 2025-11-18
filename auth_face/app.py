from __future__ import annotations

import secrets
import time
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
    score: float


@dataclass
class Session:
    token: str
    name: str
    issued_at: float


@dataclass
class FaceRecognitionSettings:
    threshold: float = 0.85
    max_faces: int = 3
    detection_confidence: float = 0.5
    padding: int = 32
    nms_threshold: float = 0.3


class FaceRecognitionApp:
    def __init__(
        self,
        db: Optional[FaceDB] = None,
        embedder: Optional[SigLIPEmbedder] = None,
        settings: Optional[FaceRecognitionSettings] = None,
    ) -> None:
        self.db = db or FaceDB()
        self.embedder = embedder or SigLIPEmbedder()
        self.settings = settings or FaceRecognitionSettings()
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.settings.detection_confidence,
        )
        self.sessions: Dict[str, Session] = {}

    def close(self) -> None:
        self.detector.close()
        self.db.close()

    def authenticate(self, frame: np.ndarray, issue_session: bool = False) -> Tuple[Optional[DetectionResult], Optional[str]]:
        result = self.match(frame)
        if not result:
            return None, None
        token = None
        if issue_session and result.label != "Unknown":
            token = self._create_session(result.label)
        return result, token

    def match(self, frame: np.ndarray) -> Optional[DetectionResult]:
        detections = self._detect_faces(frame)
        if not detections:
            return None
        results = self._build_results(frame, detections)
        if not results:
            return None
        results.sort(key=lambda item: (item.label != "Unknown", item.similarity, item.score), reverse=True)
        return results[0]

    def enroll(self, name: str, frame: np.ndarray) -> DetectionResult:
        name = name.strip()
        if not name:
            raise ValueError("Name required")
        if self.db.name_exists(name):
            raise ValueError("Name already exists")
        detections = self._detect_faces(frame)
        if not detections:
            raise ValueError("No face detected")
        results = self._build_results(frame, detections)
        if not results:
            raise ValueError("No face detected")
        target = max(results, key=lambda item: item.score)
        _, existing_name, duplicate_score = self.db.best_match(target.embedding)
        if duplicate_score >= 0.95 and existing_name:
            raise ValueError(f"Face already enrolled as {existing_name}")
        self.db.add(name, target.embedding, target.roi)
        return DetectionResult(
            label=name,
            embedding=target.embedding,
            roi=target.roi,
            similarity=1.0,
            bbox=target.bbox,
            score=target.score,
        )

    def delete_face(self, face_id: int) -> None:
        self.db.delete(face_id)

    def list_faces(self):
        return self.db.list()

    def validate_session(self, token: str) -> Optional[Session]:
        return self.sessions.get(token)

    def logout(self, token: str) -> bool:
        return self.sessions.pop(token, None) is not None

    def _create_session(self, name: str) -> str:
        token = secrets.token_urlsafe(32)
        self.sessions[token] = Session(token=token, name=name, issued_at=time.time())
        return token

    def _build_results(self, frame: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[DetectionResult]:
        results: List[DetectionResult] = []
        for bbox, score in detections:
            roi, rect = extract_roi(frame, bbox)
            if roi.size == 0 or score < self.settings.detection_confidence:
                continue
            embedding = self.embedder.embed(roi)
            name, similarity = self.db.match(embedding, self.settings.threshold)
            label = name or "Unknown"
            results.append(
                DetectionResult(
                    label=label,
                    embedding=embedding,
                    roi=roi,
                    similarity=similarity,
                    bbox=rect,
                    score=score,
                )
            )
        return results

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
