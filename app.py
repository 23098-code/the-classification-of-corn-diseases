import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Corn Leaf Disease Detection", page_icon="🌽", layout="centered")

@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

CLASS_NAMES = ["Blight", "Common Rust", "Healthy", "Gray Leaf Spot"]  # แก้ให้ตรง label ของคุณเอง

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด จากนั้นกดปุ่ม **วิเคราะห์ภาพ** เพื่อแสดงผล")

uploaded_file = st.file_uploader("📤 อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", width=350)

    # ปุ่มกดวิเคราะห์ (ไม่ประมูลอัตโนมัติแล้ว)
    if st.button("🔍 วิเคราะห์ภาพ"):
        img = img.resize((224, 224))   # <----- เปลี่ยนขนาดให้ตรงกับโมเดลของคุณ
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result_index = np.argmax(prediction)
        result = CLASS_NAMES[result_index]

        st.success(f"🩻 ผลการวิเคราะห์: **{result}**")

        # ------------------------------------
        #   การดูแลรักษาตามโรค
        # ------------------------------------
        treatment = {
            "Blight": """
            🔥 วิธีดูแลรักษาโรคใบไหม้ (Blight)
            - ตัดใบที่เป็นโรคออกและเผาทำลาย
            - หลีกเลี่ยงความชื้นสะสมในแปลง
            - ใช้สารป้องกันเชื้อราเช่น Mancozeb, Chlorothalonil
            - ปลูกพันธุ์ที่ทนทานต่อโรค
            """,
            "Common Rust": """
            🍂 วิธีดูแลรักษาโรคราสนิม (Common Rust)
            - ลดความหนาแน่นของต้นเพื่อให้มีการถ่ายเทอากาศ
            - ฉีดพ่นสารกำจัดเชื้อรา เช่น Propiconazole
            - หมั่นตรวจแปลงสม่ำเสมอ
            """,
            "Gray Leaf Spot": """
            ⚠ วิธีดูแลโรคใบจุดเทา (Gray Leaf Spot)
            - ใช้ระบบน้ำหยดแทนการพ่นบนใบ
            - ตัดเศษพืชที่ติดโรคทิ้ง
            - ใช้สารป้องกันเชื้อรา Strobilurins หรือ Triazoles
            """,
            "Healthy": """
            🌱 ใบปกติ ไม่มีโรค
            - ควบคุมน้ำและปุ๋ยเหมาะสม
            - ตรวจเช็คศัตรูพืชเป็นประจำ
            - พิจารณาพ่นป้องกันเชื้อราเป็นระยะเพื่อความปลอดภัย
            """
        }

        st.info(treatment[result])
else:
    st.warning("📌 กรุณาอัปโหลดรูปก่อนค่ะ")
