import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ======================
# CONFIG
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

THRESHOLD = 0.4

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

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
st.write("อัปโหลดรูปใบข้าวโพด แล้วกดปุ่มเพื่อเริ่มการจำแนก")

uploaded_file = st.file_uploader(
    "📤 เลือกรูปภาพ",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    # ปุ่มเริ่มจำแนก
    if st.button("🔍 เริ่มจำแนกโรค"):
        with st.spinner("กำลังวิเคราะห์ภาพ..."):
            # ===== preprocessing
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # ===== predict
            predictions = model.predict(img_array)[0]

        # ===== results
        st.subheader("📊 ค่า Confidence ของแต่ละโรค")
        for label, score in zip(CLASS_NAMES, predictions):
            st.write(f"{label}: **{score:.3f}**")

        st.subheader("✅ ผลการจำแนก (threshold = 0.4)")
        detected = False
        for label, score in zip(CLASS_NAMES, predictions):
            if score >= THRESHOLD:
                st.success(f"{label} ({score:.2f})")
                detected = True

        if not detected:
            st.info("ไม่พบโรคที่มีค่ามากกว่า threshold")

else:
    st.info("⬆️ กรุณาอัปโหลดรูปก่อน")
