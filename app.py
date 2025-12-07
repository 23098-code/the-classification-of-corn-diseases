import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# โหลดโมเดล
@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด จากนั้นกดปุ่ม **วิเคราะห์ภาพ**")

uploaded_file = st.file_uploader("📤 อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])
btn = st.button("🔍 วิเคราะห์ภาพ")  # ต้องกดก่อนถึงจะ predict

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="รูปที่อัปโหลด", use_column_width=True)

if uploaded_file is not None and btn:
    
    # แปลงภาพให้ตรงกับโมเดล (ปรับขนาดเป็น input_size)
    input_size = (224, 224)   # ❗ แก้ให้ตรงกับตอน train ถ้าไม่ใช่ 224x224
    img = img.resize(input_size)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    labels = ["Healthy", "Blight", "Rust", "Leaf Spot"]  # ❗ ระบุ class ตามโมเดลของเธอเอง
    disease = labels[result]

    st.subheader(f"🩺 ผลการวิเคราะห์: **{disease}**")

    # แสดงคำแนะนำรักษาโรค
    tips = {
        "Healthy": "ใบข้าวโพดแข็งแรงดี 🎉 ควรดูแลต่อเนื่อง ให้น้ำเพียงพอ และใส่ปุ๋ยเป็นประจำ",
        "Blight": "❗ แนะนำตัดส่วนที่เป็นโรค เผาทำลาย ลดการแพร่กระจาย ใช้สารป้องกันกำจัดเชื้อรา",
        "Rust": "⚠ มีสนิมใบ ควรใช้สารกำจัดเชื้อรา เช่น mancozeb หรือ chlorothalonil และเพิ่มการถ่ายเทอากาศ",
        "Leaf Spot": "⚠ ใบมีจุดด่าง ควรลดความชื้น ใช้สารป้องกันเชื้อรา และกำจัดใบที่ป่วยออก"
    }

    if disease in tips:
        st.info(f"💡 คำแนะนำเพิ่มเติม:\n{tips[disease]}")
