from __future__ import annotations

import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover - runtime fallback
    load_model = None


MODEL_PATH = "models/lung_xray_classifier.keras"


def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(target_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def load_model_if_available(model_path: str = MODEL_PATH):
    if load_model is None:
        st.sidebar.warning("TensorFlow is not available in this environment.")
        return None
    if not os.path.exists(model_path):
        st.sidebar.info("Model file not found. Please train a model or update the path.")
        return None

    try:
        return load_model(model_path)
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"Failed to load model: {exc}")
        return None


def main() -> None:
    st.title("Lung X-Ray Pneumonia Detector")
    st.caption("A lightweight classifier for distinguishing normal and pneumonia chest X-rays.")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
    )

    model = load_model_if_available()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("Classify"):
            if model is None:
                st.error("No model available for prediction.")
            else:
                sample = preprocess_image(image)
                prediction = float(model.predict(sample, verbose=0)[0][0])
                label = "Pneumonia detected" if prediction >= 0.5 else "Normal"
                st.write(f"Prediction: {label} (pneumonia probability: {prediction:.3f})")

    else:
        st.info("Upload an X-ray image to begin classification.")


if __name__ == "__main__":
    main()
