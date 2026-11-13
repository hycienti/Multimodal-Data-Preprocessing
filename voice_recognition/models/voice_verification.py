
import joblib
import numpy as np

class VoiceVerification:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_speaker(self, features):
        return self.model.predict(features)[0]

    def verify(self, features, claimed_identity):
        predicted_identity = self.predict_speaker(features)

        return {
            "claimed_identity": claimed_identity,
            "predicted_identity": predicted_identity,
            "verified": predicted_identity == claimed_identity
        }
