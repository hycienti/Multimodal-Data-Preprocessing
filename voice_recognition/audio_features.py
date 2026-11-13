"""Turn a raw recording into the 30 features the voice model was trained on.

Mirrors the preprocessing in Voice_Recognition.ipynb:
44.1 kHz mono -> trim leading/trailing silence (top_db=35) ->
13 MFCC (mean, std) + spectral roll-off (mean, std) + RMS energy (mean, std).
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SAMPLE_RATE = 44100
N_MFCC = 13
TRIM_TOP_DB = 35

FEATURE_COLUMNS = (
    [f"MFCC_{i}_{stat}" for i in range(1, N_MFCC + 1) for stat in ("Mean", "Std")]
    + ["Spectral_Rolloff_Mean", "Spectral_Rolloff_Std", "RMS_Energy_Mean", "RMS_Energy_Std"]
)


def load_audio(path):
    """Load an audio file as mono float32 at the training sample rate."""
    signal, _ = librosa.load(Path(path), sr=SAMPLE_RATE, mono=True)
    return signal


def trim_silence(signal):
    trimmed, _ = librosa.effects.trim(signal, top_db=TRIM_TOP_DB, frame_length=2048, hop_length=512)
    return trimmed


def extract_features(signal, sample_rate=SAMPLE_RATE):
    """Return a one-row DataFrame with the model's 30 feature columns.

    signal: 1-D float array of mono audio (already trimmed or not; it is trimmed here).
    """
    signal = np.asarray(signal, dtype=np.float32).flatten()
    if sample_rate != SAMPLE_RATE:
        signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    signal = trim_silence(signal)
    if signal.size == 0:
        raise ValueError("Recording is silent after trimming")

    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=signal)

    record = {}
    for i in range(N_MFCC):
        record[f"MFCC_{i + 1}_Mean"] = float(np.mean(mfcc[i]))
        record[f"MFCC_{i + 1}_Std"] = float(np.std(mfcc[i]))
    record["Spectral_Rolloff_Mean"] = float(np.mean(rolloff))
    record["Spectral_Rolloff_Std"] = float(np.std(rolloff))
    record["RMS_Energy_Mean"] = float(np.mean(rms))
    record["RMS_Energy_Std"] = float(np.std(rms))

    return pd.DataFrame([record], columns=FEATURE_COLUMNS)
