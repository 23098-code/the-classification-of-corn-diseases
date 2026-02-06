import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="centered"
)

MODEL_URL = "https://drive.google.com/uc?id=1uU_Oh2dKGaK0C0pym5YMMFKTjQ3FJrwc"
MODEL_PATH = "corn_disease_model.keras"

CLASS_NAMES = [
    "Blight",
    "Common Rust",
    "Grey Spot Leaf",
    "Healthy"
]

IMG_SIZE = 224
THRESHOLD = 0.5

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🌽 Corn Leaf Disease Detection")
st.write("ระบบจำแนกโรคใบข้าวโพดแบบ **Multi-label Classification**")

uploaded_file = st.file_uploader(
    "📷 อัปโหลดรูปใบข้าวโพด",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 วิเคราะห์โรค"):
        with st.spinner("กำลังวิเคราะห์..."):
            predictions = model.predict(img_array)[0]

        st.subheader("📊 ผลการวิเคราะห์")

        detected = False
        for i, score in enumerate(predictions):
            percent = score * 100
            if score >= THRESHOLD:
                detected = True
                st.success(f"✅ {CLASS_NAMES[i]} : {percent:.2f}%")
            else:
                st.write(f"❌ {CLASS_NAMES[i]} : {percent:.2f}%")

        if not detected:
            st.warning("⚠️ ไม่พบโรคที่มีค่าความมั่นใจเกินเกณฑ์")

        # raw scores (สำหรับรายงาน)
        with st.expander("🔬 ดูค่าคะแนนดิบ (Raw scores)"):
            for i, score in enumerate(predictions):
                st.write(f"{CLASS_NAMES[i]} : {score:.4f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "โครงงานการจำแนกโรคใบข้าวโพดด้วยเทคนิค Deep Learning "
    "(Multi-label Image Classification)"
)
