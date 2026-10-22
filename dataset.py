"""Read a labeled folder-based audio dataset."""
from pathlib import Path
import numpy as np
from .features import extract_features

AUDIO_EXTENSIONS = {".wav"}


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load data_dir/<label>/*.wav and return feature matrix X and labels y."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    features: list[np.ndarray] = []
    labels: list[str] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        audio_files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        for audio_path in audio_files:
            try:
                features.append(extract_features(audio_path))
                labels.append(class_dir.name.lower())
            except Exception as exc:
                print(f"Skipping {audio_path}: {exc}")

    if not features:
        raise ValueError(
            f"No readable WAV files found in {root}. "
            "Put files inside data/raw/<emotion-label>/."
        )

    return np.vstack(features), np.asarray(labels)
