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
THRESHOLD = 0.4

CLASS_NAMES = [
    "blight",
    "common_rust",
    "grey_spot_leaf",
    "healthy"
]

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 กำลังดาวน์โหลดโมเดล..."):
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
st.title("🌽 Corn Disease Classification")
st.write("อัปโหลดรูปหรือเปิดกล้องถ่ายใบข้าวโพด แล้วกดปุ่มเพื่อเริ่มการจำแนก")

# -------- input method
input_method = st.radio(
    "เลือกวิธีใส่ภาพ",
    ["📁 อัปโหลดรูป", "📷 เปิดกล้อง"]
)

image = None

# -------- upload image
if input_method == "📁 อัปโหลดรูป":
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพ",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# -------- camera input
else:
    camera_file = st.camera_input("ถ่ายภาพใบข้าวโพด")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

# ======================
# SHOW IMAGE + PREDICT
# ======================
if image is not None:
    st.image(image, caption="ภาพที่ใช้วิเคราะห์", use_container_width=True)

    if st.button("🔍 เริ่มจำแนกโรค"):
        with st.spinner("🧠 กำลังวิเคราะห์..."):
            # preprocessing
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]

        # -------- show scores
        st.subheader("📊 ค่า Confidence")
        for label, score in zip(CLASS_NAMES, predictions):
            st.write(f"{label}: **{score:.3f}**")

        # -------- threshold result
        st.subheader(f"✅ ผลการจำแนก (threshold = {THRESHOLD})")
        detected = False

        for label, score in zip(CLASS_NAMES, predictions):
            if score >= THRESHOLD:
                st.success(f"{label} ({score:.2f})")
                detected = True

        if not detected:
            st.info("ไม่พบโรคที่มีค่ามากกว่า threshold")

else:
    st.info("⬆️ กรุณาอัปโหลดรูปหรือถ่ายภาพก่อน")
