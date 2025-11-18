import base64
from typing import Tuple, Union

import cv2
import numpy as np


def extract_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return roi, bbox
    return cv2.resize(roi, (224, 224)), bbox


def load_image(source: Union[str, bytes]) -> np.ndarray:
    data = source
    if isinstance(source, str):
        payload = source.split(",", 1)[1] if ";base64," in source else source
        data = base64.b64decode(payload)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image

