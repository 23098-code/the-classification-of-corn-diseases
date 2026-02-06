import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

# ======================
# CONFIG
# ======================
MODEL_URL = "https://drive.google.com/uc?id=1uU_Oh2dKGaK0C0pym5YMMFKTjQ3FJrwc"
MODEL_PATH = "model.keras"

IMG_SIZE = 224
CLASS_NAMES = [
    "blight",
    "common_rust",
    "grey_spot_leaf",
    "healthy"
]

THRESHOLD = 0.5  # สำหรับ multi-label

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()

# ======================
# UI
# ======================
st.title("🌽 Corn Disease Classification (Multi-Label)")
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

    # preprocessing
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    predictions = model.predict(img_array)[0]

    st.subheader("Prediction Results")

    detected = False
    for label, score in zip(CLASS_NAMES, predictions):
        st.write(f"{label}: **{score:.3f}**")
        if score >= THRESHOLD:
            detected = True

    st.subheader("Detected Diseases")
    if detected:
        for label, score in zip(CLASS_NAMES, predictions):
            if score >= THRESHOLD:
                st.success(f"{label} ({score:.2f})")
    else:
        st.info("No disease detected above threshold")
