import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด แล้วกดปุ่มเพื่อวิเคราะห์")

# โหลดโมเดลแบบปลอดภัย
try:
    model = load_model("model.h5")
    st.success("โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("กำลังวิเคราะห์...")
        st.success("ทดสอบสำเร็จ (ยังไม่ใส่โมเดลจริง)")
