import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")
st.write("อัปโหลดหรือถ่ายภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    feature_extractor = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    classifier = tf.keras.models.load_model("model.h5")
    return feature_extractor, classifier

try:
    feature_extractor, model = load_models()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ---------------- CLASS INFO ----------------
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
def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

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
            img = prepare_image(image)

            # 🔑 STEP สำคัญที่สุด
            features = feature_extractor.predict(img)
            features = features.reshape(1, -1)  # (1, 25088)

            prediction = model.predict(features)[0]

            confidence = float(np.max(prediction))
            idx = int(np.argmax(prediction))
            disease = class_names[idx]

            if confidence < 0.5:
                st.warning("⚠️ ความมั่นใจต่ำกว่า 50% กรุณาถ่ายภาพใหม่ให้เห็นใบชัดเจน")
            else:
                st.success(f"🌱 ผลการวิเคราะห์: {class_names_th[disease]}")
                st.write(f"📊 ความมั่นใจ: {confidence*100:.2f}%")
                st.info(f"🩺 คำแนะนำ: {care_guide[disease]}")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
