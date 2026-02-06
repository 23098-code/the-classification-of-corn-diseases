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
st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย Deep Learning")
st.write("อัปโหลดรูปหรือถ่ายภาพใบข้าวโพด แล้วกดปุ่มเพื่อเริ่มการจำแนก")

method = st.radio(
    "เลือกวิธีใส่ภาพ",
    ["📁 อัปโหลดรูป", "📷 เปิดกล้อง"]
)

image = None

if method == "📁 อัปโหลดรูป":
    file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "png", "jpeg"])
    if file is not None:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("ถ่ายภาพใบข้าวโพด")
    if cam is not None:
        image = Image.open(cam).convert("RGB")

# ======================
# SHOW IMAGE
# ======================
if image is not None:
    st.image(image, caption="ภาพที่ใช้ในการจำแนก", use_container_width=True)

    if st.button("🔍 เริ่มจำแนกโรค"):
        with st.spinner("🧠 กำลังวิเคราะห์..."):
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]

        # ----------------------
        # RESULT WITH THRESHOLD
        # ----------------------
        st.subheader(f"✅ ผลการจำแนก")

        found = False
        for name, score in zip(CLASS_NAMES, predictions):
            if score >= THRESHOLD:
                st.success(f"{name} ")
                found = True

        if not found:
            st.info("ไม่พบโรคที่มีค่า confidence สูงกว่า threshold")

else:
    st.info("กรุณาอัปโหลดรูปหรือถ่ายภาพก่อน")


