"""Audio loading and MFCC feature extraction."""
from pathlib import Path
import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(audio_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a finite, fixed-length MFCC summary vector for an audio file."""
    signal, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
    # Mean and standard deviation over frames create a fixed-length vector.
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite audio features found for: {audio_path}")
    return features.astype(np.float32)
