import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
from functools import lru_cache


def to_rgb_image(image_source):
    """
    This function converts a file path, PIL image or RGB numpy array into an RGB PIL image
    """
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as opened_image:
            return opened_image.convert("RGB")
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_source).__name__}. "
                    "Use a file path, a PIL image or an RGB numpy array")


class DetectFaceImages:
    def __init__(self, mtcnn_model:MTCNN):
        self.mtcnn = mtcnn_model

    def detect_face(self, image_source):
        """
        This function detects faces and returns the face crop with the number of faces found.
        image_source: file path, PIL image or RGB numpy array
        """
        image = to_rgb_image(image_source)

        boxes, _ = self.mtcnn.detect(image)
        if boxes is None:
            return None, 0

        number_of_faces = len(boxes)
        if number_of_faces > 1:
            return None, number_of_faces

        face = self.mtcnn(image)
        return face, number_of_faces


@lru_cache
def mtcnn():
    """This function loads and cache mtcnn"""
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    return MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.7],
        device=device
        )

if __name__ == "__main__":
    mtcnn()
