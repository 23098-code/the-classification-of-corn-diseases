import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")
st.write("ถ่ายภาพหรืออัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

try:
    model = load_model()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ---------------- CLASS NAMES ----------------
class_names = [
    "Blight",
    "Common Rust",
    "Grey Leaf Spot",
    "Healthy"
]

class_names_th = {
    "Blight": "โรคใบไหม้",
    "Common Rust": "โรคราสนิม",
    "Grey Leaf Spot": "โรคใบจุดสีเทา",
    "Healthy": "ใบข้าวโพดสุขภาพดี"
}

care_guide = {
    "Blight": "ตัดใบที่เป็นโรค ลดความชื้น และใช้สารป้องกันเชื้อรา",
    "Common Rust": "หลีกเลี่ยงความชื้นสูง ใช้พันธุ์ต้านทาน และพ่นสารป้องกันรา",
    "Grey Leaf Spot": "ปลูกพืชหมุนเวียน เก็บเศษซากพืช และใช้สารป้องกันโรคพืช",
    "Healthy": "ต้นข้าวโพดสุขภาพดี ดูแลตามปกติ รดน้ำและใส่ปุ๋ยให้เหมาะสม"
}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ⭐ สำคัญมาก
    return img_array

# ---------------- INPUT METHOD ----------------
method = st.radio(
    "📸 เลือกวิธีนำเข้าภาพ",
    ["อัปโหลดภาพ", "ถ่ายภาพจากกล้อง"]
)

uploaded_file = None

if method == "อัปโหลดภาพ":
    uploaded_file = st.file_uploader(
        "อัปโหลดภาพใบข้าวโพด",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )
else:
    uploaded_file = st.camera_input("กดปุ่มเพื่อถ่ายภาพ")

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่ใช้วิเคราะห์", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        try:
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0]

            confidence = float(np.max(prediction))
            predicted_index = int(np.argmax(prediction))
            predicted_class = class_names[predicted_index]

            if confidence < 0.5:
                st.warning("⚠️ ความมั่นใจต่ำกว่า 50% กรุณาถ่ายภาพใหม่ให้เห็นใบชัดเจน")
            else:
                st.success(f"🌱 ผลการวิเคราะห์: {class_names_th[predicted_class]}")
                st.write(f"📊 ความมั่นใจ: {confidence*100:.2f}%")
                st.info(f"🩺 คำแนะนำ: {care_guide[predicted_class]}")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
