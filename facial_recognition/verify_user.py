import asyncio
import numpy as np
from typing import Literal
from facial_recognition.pipeline import FacialVerification
from facial_recognition.verification.inception_resnet_setup import GenerateEmbedding, embedder
from facial_recognition.verification.mtcnn_setup import mtcnn, DetectFaceImages
from facial_recognition.verification.classifier import load_classifier_artificats


async def verify_user(user_name:str, image, color_order:Literal["RGB", "BGR"]) -> dict:
    """
    This function verifies that the person in the image is the user claiming to log in.

    user_name: the name the user typed to log in (case and spaces are ignored)
    image: a file path, a PIL image, or a numpy array
    color_order: channel order of a numpy array image, "BGR" (default, OpenCV webcam frames)
                 or "RGB". Ignored for file paths and PIL images, which are always RGB.
    return dict result
    """
    if color_order not in ("RGB", "BGR"):
        raise ValueError(f"color_order must be 'RGB' or 'BGR', got {color_order!r}")
    if isinstance(image, np.ndarray) and color_order == "BGR":
        image = image[:, :, ::-1].copy()

    verifier = FacialVerification(
        face_image=DetectFaceImages(mtcnn()),
        gen_embedder=GenerateEmbedding(embedder(42)),
        model_artifacts=load_classifier_artificats(),
    )
    return await asyncio.to_thread(verifier.verify_user, user_name, image)


if __name__ == "__main__":
    result = asyncio.run(verify_user(user_name="bode", image="facial_recognition/datasets/test/9.jpeg", color_order="RGB"))
    print(result)
