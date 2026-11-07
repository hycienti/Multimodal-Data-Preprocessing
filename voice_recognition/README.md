# Voice Recognition and Speaker Verification

## 1. Overview

This folder contains the voice recognition component developed for the Multimodal Data Preprocessing – Formative 2 assignment.

The component identifies a speaker from audio features using a Support Vector Machine (SVM) classifier. It then compares the predicted speaker identity with the identity claimed by the user to determine whether voice verification succeeds.

The dataset contains recordings from four group members:

- Alain
- Bode
- Elyse
- Hycient

## 2. Project Structure

```text
voice_recognition/
│
├── Voice_Recognition.ipynb
│
├── features/
│   └── audio_features.csv
│
├── models/
│   ├── voice_verification_model.joblib
│   └── voice_verification.py
│
└── README.md
```

## 3. Audio Preprocessing

The voice recognition notebook covers the following steps:

1. Loading the original audio recordings.
2. Visualizing audio waveforms and spectrograms.
3. Cleaning and preprocessing audio recordings.
4. Applying audio augmentation techniques:
   - Pitch shifting
   - Time stretching
5. Extracting audio features.

The extracted features include:

- Mel-Frequency Cepstral Coefficients (MFCCs)
- Spectral roll-off
- Root Mean Square (RMS) energy

The extracted features are saved in `features/audio_features.csv`.

## 4. Model Development

A Support Vector Machine (SVM) classifier was trained to identify speakers using the extracted audio features.

The model uses a preprocessing pipeline that includes feature standardization and classification.

### Model evaluation

| Metric | Result |
|---|---:|
| Accuracy | 91.67% |
| Weighted F1-score | 91.43% |
| Correct predictions | 11 out of 12 |

The model was evaluated using enrollment recordings for training and verification recordings for testing.

**Note:** The dataset is relatively small, and augmented recordings are derived from the original recordings. Therefore, the evaluation results are preliminary and should not be considered a measure of real-world performance.

## 5. Voice Verification

The `models/voice_verification.py` file contains the `VoiceVerification` class.

It provides two methods:

- `predict_speaker(features)`: Predicts the identity of the speaker.
- `verify(features, claimed_identity)`: Compares the predicted identity with the claimed identity.

### Example usage

```python
import pandas as pd
from models.voice_verification import VoiceVerification

MODEL_PATH = "models/voice_verification_model.joblib"

verifier = VoiceVerification(MODEL_PATH)

# Load extracted audio features
features_df = pd.read_csv("features/audio_features.csv")

# Select the 30 numerical feature columns
feature_columns = [
    column for column in features_df.columns
    if column.startswith("MFCC_")
    or column.startswith("Spectral_Rolloff_")
    or column.startswith("RMS_Energy_")
]

sample_features = features_df[feature_columns].iloc[[0]]

# Verify the claimed identity
result = verifier.verify(sample_features, "bode")

print(result)
```

The input must contain the same 30 numerical feature columns used during model training. This interface accepts extracted features, not raw WAV recordings.

### Verification output

The verification method returns a dictionary containing:

- `claimed_identity`
- `predicted_identity`
- `verified`

If the predicted identity matches the claimed identity, `verified` is `True`. Otherwise, it is `False`.

## 6. Demonstration Results

Two verification scenarios were tested:

**Authorized attempt**

- Actual speaker: Bode
- Claimed identity: Bode
- Result: Access granted (`verified: True`)

**Unauthorized attempt**

- Actual speaker: Bode
- Claimed identity: Alain
- Result: Access denied (`verified: False`)

The unauthorized test demonstrates identity mismatch detection. It does not demonstrate detection of completely unknown speakers.

## 7. Requirements

The notebook and model use Python libraries including:

- Python
- NumPy
- Pandas
- Librosa
- Matplotlib
- Scikit-learn
- Joblib
- IPython

Install the required libraries using:

```bash
pip install numpy pandas librosa matplotlib scikit-learn joblib ipython
```

## 8. Integration Notes

The voice recognition component is intended to be integrated into the group's multimodal authentication system.

The integration system should:

1. Receive an audio recording.
2. Preprocess the recording and extract the required 30 numerical features.
3. Pass the extracted features and claimed identity to the `VoiceVerification` class.
4. Use the returned `verified` value to determine whether voice authentication succeeds.

The component currently expects pre-extracted numerical features. Audio preprocessing and feature extraction must therefore be performed before calling the verification methods.

## 9. Limitations

- The model uses a small dataset containing four known speakers.
- Augmented recordings are derived from the original recordings.
- Evaluation results are preliminary.
- The current implementation checks whether the predicted speaker matches the claimed identity; it has not been validated for unknown-speaker detection.
- The model does not independently perform raw audio feature extraction through the verification class.

## 10. Project Deliverables

| File | Description |
|---|---|
| `Voice_Recognition.ipynb` | Complete voice preprocessing, augmentation, feature extraction, training and evaluation notebook |
| `features/audio_features.csv` | Extracted audio feature dataset |
| `models/voice_verification_model.joblib` | Trained SVM model and preprocessing pipeline |
| `models/voice_verification.py` | Python voice verification interface |
| `README.md` | Component documentation |

---

**Project:** Multimodal Data Preprocessing – Formative 2  
**Component:** Voice Recognition and Speaker Verification
