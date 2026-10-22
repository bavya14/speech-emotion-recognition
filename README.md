# Speech Emotion Recognition

A beginner-friendly machine-learning project that classifies short audio recordings into emotion labels using **MFCC audio features** and a **scikit-learn classifier**.

> **Important:** This repository is a working project template. It does not include a trained model or audio dataset. Train it with a suitable, legally obtained dataset before using predictions. Emotion labels are estimates from voice patterns—not reliable facts about a person's inner state.

## Features

- Extracts Mel-Frequency Cepstral Coefficients (MFCCs) with Librosa.
- Trains a scikit-learn classifier from a folder-based audio dataset.
- Evaluates a held-out test split and saves a model artifact.
- Predicts an emotion for a WAV audio file.
- Includes an optional Streamlit web interface.
- Keeps data and generated model files out of Git by default.

## Emotion labels

The included dataset loader supports folder names such as `happy`, `sad`, `angry`, `neutral`, `fearful`, `disgust`, and `surprised`. The exact labels depend on your dataset. Use a dataset whose license permits your intended use.

## Requirements

- Python 3.10 or 3.11 recommended
- `pip`
- Audio files in WAV format (mono or stereo; files are converted to mono for feature extraction)

## Quick start

### 1. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add a dataset

Arrange labeled WAV files like this:

```text
data/raw/
├── angry/
│   ├── angry_001.wav
│   └── angry_002.wav
├── happy/
│   ├── happy_001.wav
│   └── happy_002.wav
├── neutral/
└── sad/
```

Each class folder should contain several recordings. For a meaningful evaluation, use enough recordings per class and, where possible, split by speaker so recordings from the same person do not appear in both training and test sets. Dataset folder names become the labels.

The project does **not** bundle audio because datasets may have separate licensing, consent, and redistribution requirements. See [DATASET_GUIDE.md](DATASET_GUIDE.md).

### 4. Train and evaluate

```bash
python -m src.train --data-dir data/raw --model-path models/emotion_model.joblib
```

The script prints evaluation metrics and saves the trained model. You need at least two classes with enough examples for a stratified split.

### 5. Predict from an audio file

```bash
python -m src.predict --model-path models/emotion_model.joblib --audio path/to/recording.wav
```

### 6. Run the optional web app

```bash
streamlit run app.py
```

Upload a WAV file in the browser. The app requires a trained model at `models/emotion_model.joblib`.

## Project structure

```text
speech_emotion_recognition/
├── app.py
├── requirements.txt
├── README.md
├── DATASET_GUIDE.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_features.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
└── models/.gitkeep
```

## How it works

1. **Load audio:** Librosa reads each WAV recording and converts it to mono at a consistent sample rate.
2. **Extract features:** MFCCs summarize the short-term spectral envelope of speech. Their mean and standard deviation over time form a fixed-length feature vector.
3. **Train classifier:** A scikit-learn pipeline scales features and trains an RBF support vector classifier.
4. **Evaluate:** A held-out split reports accuracy and a classification report. This is a baseline; results depend heavily on dataset quality and speaker-independent evaluation.
5. **Predict:** The saved pipeline predicts one label for a new recording.

## Limitations and responsible use

- Vocal expression varies by language, culture, speaker, microphone, environment, and context.
- Labels in emotion datasets are often acted or annotated subjectively.
- This is an educational demonstration, not a medical, hiring, disciplinary, or surveillance tool.
- Obtain consent before recording or processing anyone's voice, and follow dataset terms and local law.
- Do not interpret a prediction as a definitive statement about someone's actual emotional state.

## Troubleshooting

- **No WAV files found:** Confirm that `data/raw/<label>/` contains `.wav` files.
- **Only one class found:** Add at least two emotion folders with enough recordings.
- **Audio decoding error:** Convert the file to a standard PCM WAV using a trusted audio tool.
- **Model not found:** Train the model first or update the model path.
- **Librosa installation issue:** Use the recommended Python version and reinstall dependencies in a clean virtual environment.

## GitHub upload

```bash
git init
git add .
git commit -m "Initial speech emotion recognition project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/speech-emotion-recognition.git
git push -u origin main
```

Do not commit private recordings, API keys, virtual environments, or large datasets. If you want to share a trained model, confirm its size and the dataset's license first.
