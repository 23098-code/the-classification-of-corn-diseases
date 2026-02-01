import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")

# =========================
# โหลดโมเดล
# =========================
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

try:
    model = load_cnn_model()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# =========================
# อ่าน input shape จากโมเดลจริง
# =========================
input_shape = model.input_shape  # (None, H, W, C)
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]


# =========================
# คลาส (ต้องตรงลำดับตอน train)
# =========================
class_names = [
    "Blight",
    "Common_Rust",
    "Grey_Leaf_Spot",
    "Healthy"
]

class_names_th = {
    "Blight": "โรคใบไหม้ (Blight)",
    "Common_Rust": "โรคราสนิม (Common Rust)",
    "Grey_Leaf_Spot": "โรคใบจุดสีเทา (Grey Leaf Spot)",
    "Healthy": "ใบข้าวโพดสุขภาพดี"
}

care_guide = {
    "Blight": "ตัดใบป่วยออก ลดความชื้น ใช้สารป้องกันเชื้อรา",
    "Common_Rust": "ตัดใบป่วย เพิ่มการถ่ายเทอากาศ",
    "Grey_Leaf_Spot": "เก็บใบป่วย ทำลายเศษพืช ไม่ปลูกซ้ำ",
    "Healthy": "ต้นแข็งแรง ดูแลตามปกติ ให้น้ำและปุ๋ยเหมาะสม"
}

# =========================
# เตรียมภาพ (สำคัญที่สุด)
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# รับภาพ
# =========================
method = st.radio("เลือกวิธีนำเข้าภาพ", ["📁 อัปโหลดรูป", "📸 ถ่ายภาพ"])

image = None

if method == "📁 อัปโหลดรูป":
    file = st.file_uploader(
        "อัปโหลดภาพ",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )
    if file:
        image = Image.open(file)

else:
    cam = st.camera_input("กดปุ่มถ่ายภาพ")
    if cam:
        image = Image.open(cam)

# =========================
# วิเคราะห์
# =========================
if image is not None:
    st.image(image, caption="ภาพที่ใช้วิเคราะห์", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        with st.spinner("กำลังวิเคราะห์..."):
            try:
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0]

                predicted_index = int(np.argmax(prediction))
                confidence = float(prediction[predicted_index])

                if confidence < 0.5:
                    st.warning("⚠️ รูปภาพไม่ชัดเจน กรุณาถ่ายภาพใหม่")
                    st.stop()

                predicted_class = class_names[predicted_index]

                st.success(
                    f"✅ ผลการวิเคราะห์: **{class_names_th[predicted_class]}**"
                )
            

                st.subheader("🩺 แนวทางการดูแล")
                st.info(care_guide[predicted_class])

            except Exception as e:
                st.error(f"❌ วิเคราะห์ไม่สำเร็จ: {e}")

