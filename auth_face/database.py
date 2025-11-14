import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CachedFace:
    name: str
    embedding: np.ndarray


@dataclass
class FaceRecord:
    id: int
    name: str
    created_at: str


class FaceDB:
    """
    Lightweight SQLite-backed face embedding store with an in-memory cache.
    """

    def __init__(self, path: str = "faces.db") -> None:
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self._ensure_schema()
        self.cache: Dict[int, CachedFace] = self._load_cache()

    def _ensure_schema(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                name TEXT,
                embedding BLOB,
                image BLOB,
                created_at TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def _load_cache(self) -> Dict[int, CachedFace]:
        cache: Dict[int, CachedFace] = {}
        query = "SELECT id, name, embedding FROM faces"
        for face_id, name, embedding_blob in self.cursor.execute(query):
            cache[face_id] = CachedFace(name=name, embedding=pickle.loads(embedding_blob))
        return cache

    def add(self, name: str, embedding: np.ndarray, image: np.ndarray) -> int:
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to encode face ROI as JPEG.")

        self.cursor.execute(
            """
            INSERT INTO faces (name, embedding, image, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, pickle.dumps(embedding), buffer.tobytes(), datetime.now()),
        )
        self.conn.commit()

        face_id = self.cursor.lastrowid
        self.cache[face_id] = CachedFace(name=name, embedding=embedding)
        return face_id

    def match(self, embedding: np.ndarray, threshold: float = 0.75) -> Tuple[Optional[str], float]:
        if not self.cache:
            return None, 0.0

        best_id, best_name, best_score = max(
            (
                (face_id, data.name, float(np.dot(embedding, data.embedding)))
                for face_id, data in self.cache.items()
            ),
            key=lambda entry: entry[2],
        )

        if best_score < threshold:
            return None, best_score
        return best_name, best_score

    def list(self) -> List[FaceRecord]:
        query = "SELECT id, name, created_at FROM faces ORDER BY created_at DESC"
        return [FaceRecord(id=row[0], name=row[1], created_at=row[2]) for row in self.cursor.execute(query)]

    def delete(self, face_id: int) -> None:
        self.cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        self.conn.commit()
        self.cache.pop(face_id, None)

    def close(self) -> None:
        self.conn.close()

