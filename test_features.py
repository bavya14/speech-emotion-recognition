"""Unit tests for MFCC feature extraction."""
import numpy as np
import soundfile as sf
from src.features import extract_features, N_MFCC


def test_feature_vector_has_expected_shape_and_finite_values(tmp_path):
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, signal, sample_rate)

    features = extract_features(audio_path)
    assert features.shape == (2 * N_MFCC,)
    assert np.isfinite(features).all()
