"""Predict an emotion label from a WAV file."""
import argparse
from pathlib import Path
import joblib
from .features import extract_features


def predict(audio_path: str, model_path: str) -> str:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}. Train it first with `python -m src.train`."
        )
    bundle = joblib.load(model_file)
    features = extract_features(audio_path).reshape(1, -1)
    return str(bundle["model"].predict(features)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to a WAV recording")
    parser.add_argument("--model-path", default="models/emotion_model.joblib")
    args = parser.parse_args()
    print(f"Predicted emotion: {predict(args.audio, args.model_path)}")


if __name__ == "__main__":
    main()
