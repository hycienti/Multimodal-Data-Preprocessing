from facial_recognition.verification.inception_resnet_setup import GenerateEmbedding, embedder
from facial_recognition.verification.mtcnn_setup import mtcnn, DetectFaceImages
from facial_recognition.verification.classifier import load_classifier_artificats


class FacialVerification:
    def __init__(self, face_image:DetectFaceImages, gen_embedder:GenerateEmbedding, model_artifacts:dict):
        self.detect_face = face_image
        self.embedder = gen_embedder
        self.classifier = model_artifacts["classifier"]
        self.reference_embeddings = model_artifacts["reference_embeddings"]
        self.threshold = model_artifacts["threshold"]

    def build_result(self, success:bool, message:str, matches=None, person_login=None):
        """This function builds the same result format for every outcome"""
        return {
            "success": success,
            "status": "success" if success else "failed",
            "decision": "granted" if success else "denied",
            "message": message,
            "matches": matches if matches is not None else [],
            "person_login": person_login,
        }

    def get_face_embedding(self, image):
        """This function returns (embedding, None) or (None, reason it failed)"""
        face, number_of_faces = self.detect_face.detect_face(image_source=image)
        if number_of_faces == 0:
            return None, "No face detected. Try again"
        if number_of_faces > 1:
            return None, f"{number_of_faces} faces detected. Only one person should be in front of the camera"

        user_embedding = self.embedder.generate_embedding(face_image=face)
        if user_embedding is None:
            return None, "Face could not be processed. Try again"
        return user_embedding, None

    def get_classifier_guess(self, user_embedding):
        """This function returns the classifier's guess and its probability (information only)"""
        class_probabilities = self.classifier.predict_proba([user_embedding])[0]
        classifier_guess = self.classifier.classes_[class_probabilities.argmax()]
        highest_probability = float(class_probabilities.max())
        return classifier_guess, highest_probability

    def find_matches(self, user_embedding, login_name:str, classifier_guess:str, highest_probability:float):
        """This function returns every registered person whose similarity passes the threshold"""
        matched_people = []
        for registered_name, saved_embedding in self.reference_embeddings.items():
            similarity = float(user_embedding @ saved_embedding)
            if similarity >= self.threshold:
                matched_people.append({
                    "user_attempt_login": login_name,
                    "matched_name": registered_name,
                    "similarity_score": round(similarity, 3),
                    "classifier_guess": classifier_guess,
                    "probability_of_match": f"{highest_probability:.1%}",
                })
        return matched_people

    def decide(self, login_name:str, matched_people:list):
        """This function applies the access rules to the matches"""
        if login_name not in self.reference_embeddings:
            return self.build_result(False, f"User '{login_name}' is not registered", matched_people)
        if len(matched_people) > 1:
            return self.build_result(False, "Face matches more than one registered person. Try again", matched_people)
        if len(matched_people) == 0:
            return self.build_result(False, "No matching user detected. Try again", matched_people)
        if matched_people[0]["matched_name"] != login_name:
            return self.build_result(False, "Face does not match this account. Try again", matched_people)
        return self.build_result(True, "Access granted", matched_people[0],
                                 person_login=login_name.upper())

    def verify_user(self, user_name:str, image):
        """This function verifies that the person in the image is the user claiming to log in"""
        login_name = user_name.strip().lower()

        user_embedding, error_message = self.get_face_embedding(image)
        if error_message:
            return self.build_result(False, error_message)

        classifier_guess, highest_probability = self.get_classifier_guess(user_embedding)
        matched_people = self.find_matches(user_embedding, login_name, classifier_guess, highest_probability)
        return self.decide(login_name, matched_people)


if __name__ == "__main__":
    verifier = FacialVerification(
        face_image=DetectFaceImages(mtcnn()),
        gen_embedder=GenerateEmbedding(embedder(42)),
        model_artifacts=load_classifier_artificats(),
    )
    print(verifier.verify_user("Bode", "facial_recognition/datasets/original/bode/bode_neutral.jpeg"))
