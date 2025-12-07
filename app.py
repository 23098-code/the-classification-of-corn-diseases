import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# โหลดโมเดลครั้งเดียว
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

# Class label ต้องแก้ตามโมเดลของเธอ
CLASS_NAMES = ["Leaf Blight", "Healthy", "Rust"]

# ข้อมูลโรค + วิธีดูแลรักษา
TREATMENT = {
    "Leaf Blight": "• ตัดใบที่เป็นโรคออกและเผาทำลาย\n• ใช้สารป้องกันกำจัดเชื้อราเช่น แมนโคเซบ หรือ โพรพิเนบ ตามคำแนะนำ\n• เลือกพันธุ์ที่ต้านทานโรค และเว้นระยะปลูกให้โปร่ง",
    "Rust": "• พ่นสารป้องกันเชื้อราเช่น ไตรไซคลาโซล หรือ โพรพิเนบ\n• หมั่นตรวจแปลงสม่ำเสมอ ลดความชื้นในแปลง\n• ถอนต้นที่รุนแรงออกเพื่อลดการแพร่กระจาย",
    "Healthy": "• พืชแข็งแรง ปกติดี 🎉\n• รดน้ำสม่ำเสมอ ใส่ปุ๋ยตามระยะ\n• ตรวจใบทุกสัปดาห์เพื่อเฝ้าระวังโรค"
}

# ---------------- UI ---------------- #
st.set_page_config(page_title="Corn Disease Classification", layout="centered")

st.markdown("""
<h2 style='text-align:center; color:#2E8B57;'>🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN</h2>
<p style='text-align:center;'>อัปโหลดรูปใบข้าวโพด แล้วกดปุ่มวิเคราะห์เพื่อดูผลการตรวจโรค</p>
<hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 อัปโหลดภาพใบข้าวโพด", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", width=300)

    # ปุ่มประมวลผล
    if st.button("🔍 วิเคราะห์รูปภาพ"):
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = CLASS_NAMES[class_index]

        st.success(f"📌 ผลการวิเคราะห์: **{result}**")

        st.subheader("🛠 วิธีดูแลรักษา / แนะนำการป้องกัน")
        st.info(TREATMENT[result])
else:
    st.warning("กรุณาอัปโหลดรูปภาพก่อน")
