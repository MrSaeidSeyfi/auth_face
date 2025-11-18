from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPEmbedder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        pil_image = self._ensure_pil(image)
        inputs = {
            key: value.to(self.device)
            for key, value in self.processor(images=pil_image, return_tensors="pt").items()
        }

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        embedding = features / features.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    @staticmethod
    def _ensure_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        raise TypeError("Unsupported image type for embedding.")

