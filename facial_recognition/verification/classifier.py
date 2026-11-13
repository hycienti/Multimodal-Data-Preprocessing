import joblib
import os
from pathlib import Path
from functools import lru_cache

@lru_cache
def load_classifier_artificats():
    """
    This function loads SVM model
    """
    path_model = Path(__file__).resolve().parent.parent/"model"/"face_verification.joblib"
    model = joblib.load(path_model) 
    return model
