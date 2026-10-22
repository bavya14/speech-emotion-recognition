"""Streamlit interface for the trained speech emotion model."""
import tempfile
from pathlib import Path
import joblib
import streamlit as st
from src.features import extract_features

MODEL_PATH = Path("models/emotion_model.joblib")

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload a short WAV recording to estimate its dataset emotion label using MFCC features."
)
st.warning(
    "Educational demo only: a voice-based prediction is uncertain and does not prove "
    "what a person is feeling. Use recordings only with appropriate consent."
)

if not MODEL_PATH.exists():
    st.info("No trained model found. Add a labeled dataset and run `python -m src.train` first.")
    st.stop()

uploaded = st.file_uploader("Choose a WAV audio file", type=["wav"])
if uploaded is not None:
    st.audio(uploaded)
    if st.button("Predict emotion", type="primary"):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = Path(tmp.name)
            bundle = joblib.load(MODEL_PATH)
            features = extract_features(temp_path).reshape(1, -1)
            prediction = bundle["model"].predict(features)[0]
            st.subheader(f"Predicted label: {str(prediction).capitalize()}")
            st.caption("This is a model estimate, not a definitive assessment of a person's emotional state.")
        except Exception as exc:
            st.error(f"Could not process this audio: {exc}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
