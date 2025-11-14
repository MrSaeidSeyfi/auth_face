from typing import Tuple

import cv2
import numpy as np


def extract_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return roi, bbox
    return cv2.resize(roi, (224, 224)), bbox

