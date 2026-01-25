import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# ==============================
# ตั้งค่าหน้าเว็บ
# ==============================
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f4fff6;
        }
        h1, h2, h3 {
            color: #1b5e20;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด จากนั้นกดปุ่ม **วิเคราะห์ภาพ**")

# ==============================
# โหลดโมเดล
# ==============================
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

try:
    model = load_cnn_model()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ==============================
# ชื่อคลาส (แก้ให้ตรงกับตอน train)
# ==============================
class_names = [
    "Healthy",
    "Corn Blight",
    "Corn Rust"
]

# ==============================
# อัปโหลดรูป
# ==============================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png","jfif","WEBP"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("⏳ กำลังวิเคราะห์ภาพ...")

        # ==============================
        # PREPROCESS (กันพัง 99%)
        # ==============================
        try:
            # ดึงขนาด input จากโมเดล
            input_shape = model.input_shape

            # กรณี (None, 224, 224, 3)
            if len(input_shape) == 4:
                img_size = input_shape[1]
            else:
                st.error("❌ รูปแบบโมเดลไม่รองรับ")
                st.stop()

            # แปลงเป็น RGB เสมอ
            image = image.convert("RGB")

            # resize ให้ตรงกับโมเดล
            image = image.resize((img_size, img_size))

            # แปลงเป็น array
            img_array = np.array(image, dtype=np.float32)

            # normalize
            img_array = img_array / 255.0

            # เพิ่ม batch dimension
            img_array = np.expand_dims(img_array, axis=0)

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการเตรียมภาพ: {e}")
            st.stop()

        # ==============================
        # PREDICT
        # ==============================
        try:
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            st.success(
                f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**"
            )
            st.write(f"📊 ความมั่นใจของโมเดล: **{confidence:.2f}%**")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
            st.stop()

        # ==============================
        # วิธีดูแลรักษา
        # ==============================
        st.subheader("🧑‍🌾 แนวทางการดูแลรักษา")

        care_guide = {
            "Healthy": "ใบข้าวโพดแข็งแรงดี ควรรดน้ำสม่ำเสมอ และใส่ปุ๋ยตามระยะ",
            "Corn Blight": "ควรตัดใบที่เป็นโรค เผาทำลาย และใช้สารป้องกันเชื้อรา",
            "Corn Rust": "หลีกเลี่ยงความชื้นสูง ใช้สารป้องกันโรคพืชตามคำแนะนำ"
        }

        st.write(care_guide[class_names[predicted_class]])
