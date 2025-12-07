import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------ Load model ------------------
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

model = load_cnn_model()

# ------------------ UI ------------------
st.set_page_config(page_title="🌽 Corn Disease Classification", layout="centered")

st.markdown("""
<h1 style="text-align:center;color:#228B22;">🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN</h1>
<p style="text-align:center;">อัปโหลดภาพใบข้าวโพดเพื่อให้ระบบวิเคราะห์โรค</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 อัปโหลดรูปภาพใบข้าวโพด", type=["jpg","jpeg","png"])

btn = st.button("🔍 วิเคราะห์โรค")

# ------------------ Predict ------------------
if uploaded_file and btn:
    img = Image.open(uploaded_file).resize((224,224))  # ⚠ เปลี่ยนตาม input model
    st.image(uploaded_file, caption="ภาพที่อัปโหลด", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # ---- label โรคของโมเดล ----
    labels = ["ใบไหม้ (Blight)", "ใบจุด (Leaf Spot)", "ใบสนิม (Rust)", "ปกติ (Healthy)"]
    disease = labels[class_idx]

    st.subheader(f"🩺 ผลการวิเคราะห์: **{disease}**")

    # แสดงวิธีดูแลรักษาตามโรคที่พบ
    st.write("---")
    st.write("📌 **คำแนะนำในการดูแลรักษา**")

    treatment = {
        "ใบไหม้ (Blight)": """
        - กำจัดใบที่เป็นโรคเพื่อลดการแพร่กระจาย  
        - หลีกเลี่ยงการให้น้ำโดนใบโดยตรง  
        - ใช้สารป้องกันกำจัดเชื้อราเช่น แมนโคเซบ หรือ คลอโรทาโลนิล  
        """,
        "ใบจุด (Leaf Spot)": """
        - เว้นระยะปลูกให้โปร่งอากาศถ่ายเท  
        - ฉีดพ่นสารป้องกันเชื้อราเมื่อพบระยะแรก  
        - หมุนเวียนพืชปลูกลดเชื้อสะสม  
        """,
        "ใบสนิม (Rust)": """
        - ใช้เมล็ดพันธุ์ต้านทานโรค  
        - ตัดใบที่เป็นโรคทำลายทิ้ง  
        - ใช้สารกลุ่มสโตรบิลูรินเมื่อระบาดหนัก  
        """,
        "ปกติ (Healthy)": """
        - ใบปกติดี! 🌱  
        - ให้น้ำและปุ๋ยสม่ำเสมอ  
        - ตรวจแปลงเพื่อเฝ้าระวังโรคเป็นประจำ  
        """
    }

    st.success(treatment[disease])

elif uploaded_file and not btn:
    st.info("กดปุ่ม **🔍 วิเคราะห์โรค** เพื่อประมวลผล")
else:
    st.warning("โปรดอัปโหลดรูปภาพก่อน")


