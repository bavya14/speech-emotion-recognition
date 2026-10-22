"""Train and evaluate the speech emotion classifier."""
import argparse
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import load_dataset


def train(data_dir: str, model_path: str, test_size: float = 0.2, random_state: int = 42) -> None:
    X, y = load_dataset(data_dir)
    classes, counts = __import__("numpy").unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Training requires at least two emotion classes.")
    if counts.min() < 2:
        raise ValueError("Each class needs at least two WAV files for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Samples: {len(y)} | Classes: {', '.join(classes)}")
    print(f"Hold-out accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels in sorted order):")
    print(classes)
    print(confusion_matrix(y_test, predictions, labels=classes))

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": classes.tolist()}, destination)
    print(f"\nSaved model to: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Folder with one subfolder per emotion")
    parser.add_argument("--model-path", default="models/emotion_model.joblib", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used for hold-out evaluation")
    args = parser.parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.5")
    train(args.data_dir, args.model_path, args.test_size)


if __name__ == "__main__":
    main()
