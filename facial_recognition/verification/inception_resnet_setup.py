import torch
from facenet_pytorch import InceptionResnetV1
from functools import lru_cache

class GenerateEmbedding:
    def __init__(self, embedder:InceptionResnetV1):
        self.embedder = embedder
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

    def generate_embedding(self, face_image):
        """
        This function extract features from face images and generate embedding
        """
        if face_image is None:
            return None
        with torch.no_grad():
            embedding = self.embedder(face_image.unsqueeze(0).to(self.device))[0].cpu().numpy()
        return embedding

@lru_cache
def embedder(SEED:int):
    """
    This function loads pretrained Vggface2
    """
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    return InceptionResnetV1(
        pretrained="vggface2").eval().to(device)
