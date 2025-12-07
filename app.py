import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# โหลดโมเดล
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")  # <--- โหลดผ่าน LFS แล้วใช้ได้เลย

model = load_cnn_model()

# mapping label
classes = ["ใบไหม้ (Leaf Blight)", "ใบจุด (Leaf Spot)", "ปกติ (Healthy)"]

# ============ UI STYLE ============#
st.set_page_config(
    page_title="🌽 ระบบจำแนกโรคใบข้าวโพด",
    page_icon="🌽",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color:#f7fff3; }     
    .title { text-align:center;
             color:#2d6a4f;
             font-size:35px;
             font-weight:700;}
    .sub {text-align:center; font-size:18px; color:#52796f; margin-top:-10px;}
    .footer {text-align:center; font-size:13px; color:#6c757d; margin-top:30px;}
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============#
st.markdown("<h1 class='title'>🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>อัปโหลดภาพใบข้าวโพด แล้ว AI จะทำนายให้ทันที</p>", unsafe_allow_html=True)
st.write("")

uploaded_file = st.file_uploader("📤 เลือกไฟล์รูปภาพใบข้าวโพด", type=["jpg","jpeg","png"])

# ============ Prediction ============#
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_resized = cv2.resize(img, (150,150))
    img_resized = img_resized/255.0
    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index] * 100

    # แสดงภาพ
    st.image(image, caption="🖼 ภาพที่อัปโหลด", use_column_width=True)

    st.success(f"### 🔍 ผลการจำแนก: **{classes[pred_index]}**")
    st.write(f"📊 ความมั่นใจของโมเดล: **{confidence:.2f}%**")

else:
    st.info("⬆ กรุณาอัปโหลดรูปภาพก่อนเริ่มทำนาย")


# Footer
st.markdown("""
<br><p class='footer'>พัฒนาโดยใช้ Convolutional Neural Network (CNN) สำหรับโครงงานวิทยาศาสตร์</p>
""", unsafe_allow_html=True)
