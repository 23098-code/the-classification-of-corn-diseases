import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ======================
# CONFIG (ต้องตรงกับตอนเทรน)
# ======================
MODEL_URL = "https://drive.google.com/uc?id=1uU_Oh2dKGaK0C0pym5YMMFKTjQ3FJrwc"
MODEL_PATH = "model_multilabel.h5"

IMG_SIZE = 128
CLASS_NAMES = [
    "blight",
    "common_rust",
    "grey_spot_leaf",
    "healthy"
]

THRESHOLD = 0.5

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # สำคัญมาก!
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    return model


model = load_model()

# ======================
# UI
# ======================
st.title("🌽 Corn Disease Classification (Multi-Label CNN)")
st.write("Upload an image of corn leaf")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# ======================
# PREDICTION
# ======================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ===== preprocessing (เหมือนตอนเทรน)
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===== predict
    predictions = model.predict(img_array)[0]

    st.subheader("Prediction Scores")
    for label, score in zip(CLASS_NAMES, predictions):
        st.write(f"{label}: **{score:.3f}**")

    st.subheader("Detected Diseases")
    detected = False
    for label, score in zip(CLASS_NAMES, predictions):
        if score >= THRESHOLD:
            st.success(f"{label} ({score:.2f})")
            detected = True

    if not detected:
        st.info("No disease detected above threshold")
