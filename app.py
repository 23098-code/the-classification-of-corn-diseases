import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==============================
# โหลดโมเดล
# ==============================
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

# ชื่อคลาสตามโมเดลของคุณ
class_names = ["Blight", "Leaf Spot", "Healthy"]   # ← แก้ให้ตรงกับคลาสที่คุณ train

# ==============================
# ฟังก์ชันประมวลผลภาพ
# ==============================
def preprocess(img, target_size=(224,224)): # เปลี่ยนเป็น (1054,404) หากโมเดลต้องการ
    img = img.resize(target_size)  
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# Theme UI Main Page
# ==============================
st.set_page_config(
    page_title="Corn Leaf Disease Classifier",
    page_icon="🌽",
    layout="centered"
)

# CSS ปรับแต่งธีมสีเขียวหรู
st.markdown("""
    <style>
        body {background-color:#e8f5e9;}
        .title {color:#1b5e20; text-align:center; font-size:38px; font-weight:700;}
        .sub {text-align:center; font-size:20px; color:#33691e;}
        .result-box {
            background:#ffffff;
            border-radius:15px;
            padding:20px;
            border:3px solid #4caf50;
        }
        .btn {background:#2e7d32;color:white;font-size:18px;padding:10px;border-radius:8px;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# UI
# ==============================
st.markdown("<p class='title'>🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>อัปโหลดภาพใบข้าวโพด → กดปุ่ม <b>วิเคราะห์ภาพ</b> เพื่อดูผล</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("📤 อัปโหลดรูปภาพ (JPEG/PNG)", type=["jpg","jpeg","png"])

process_btn = st.button("🔍 วิเคราะห์ภาพ", use_container_width=True)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="ภาพที่อัปโหลด", width=350)

    if process_btn:
        img_array = preprocess(img, (224,224))  # ← เปลี่ยนเป็น (1054,404) ถ้าโมเดลต้องการ
        prediction = model.predict(img_array)
        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)*100
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🟩 ผลการวิเคราะห์")

        st.markdown(f"""
        <div class='result-box'>
            <h2 style='color:#2e7d32;'>ผลตรวจ: <b>{result}</b></h2>
            <p>ความมั่นใจของโมเดล: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # ==============================
        # คำแนะนำการดูแลรักษา
        # ==============================
        st.subheader("🌱 แนวทางการดูแลรักษา")

        if result == "Healthy":
            st.success("ใบข้าวโพดปกติดี แนะนำดูแลต่อเนื่อง 🌿")
            st.write("- รดน้ำสม่ำเสมอ ไม่ให้น้ำขัง\n- ตรวจเช็คศัตรูพืชเป็นประจำ\n- ใช้ปุ๋ยตามระยะการเจริญเติบโต")
        elif result == "Blight":
            st.warning("พบโรคใบไหม้ (Blight) 🔥")
            st.write("- ตัดใบที่เป็นโรคทิ้ง เพื่อลดการแพร่กระจาย\n- ใช้สารป้องกันเชื้อรา เช่น แมนโคเซบ/คอปเปอร์ออกซีคลอไรด์\n- เพิ่มการระบายอากาศในแปลงปลูก")
        elif result == "Leaf Spot":
            st.warning("พบโรคใบจุด (Leaf Spot) 🟡")
            st.write("- เก็บเศษซากพืชที่เป็นโรคออกจากแปลง\n- ใช้สารป้องกันกำจัดเชื้อราเป็นระยะ\n- ปลูกหมุนเวียน ลดเชื้อสะสมในดิน")

