import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

# =========================
# STYLE (GREEN THEME)
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: #f4f9f4;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด แล้วกดปุ่ม **วิเคราะห์ภาพ**")

# =========================
# LOAD MODEL
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
# CLASS NAMES (แก้ให้ตรงกับโมเดลคุณ)
# =========================
class_names = [
    "ใบไหม้ (Leaf Blight)",
    "สนิมข้าวโพด (Rust)",
    "ใบจุด (Leaf Spot)",
    "ปกติ (Healthy)"
]

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    # =========================
# PREPROCESS IMAGE (SAFE)
# =========================
input_shape = model.input_shape
img_size = input_shape[1]  # เช่น 224
channels = input_shape[3]  # 3 หรือ 1

image_resized = image.resize((img_size, img_size))

img_array = np.array(image_resized)

# ถ้าโมเดลต้องการ grayscale
if channels == 1:
    if img_array.ndim == 3:
        img_array = img_array[:, :, 0]
    img_array = np.expand_dims(img_array, axis=-1)

# ถ้าเป็น RGBA → RGB
if img_array.ndim == 3 and img_array.shape[-1] == 4:
    img_array = img_array[:, :, :3]

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# =========================
# PREDICT
# =========================
prediction = model.predict(img_array)


        # =========================
        # RESULT
        # =========================
        st.success(f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**")
        st.write(f"📊 ความมั่นใจ: **{confidence:.2f}%**")

        # =========================
        # CARE RECOMMENDATION
        # =========================
        st.subheader("🩺 แนวทางดูแลรักษา")

        if predicted_class == 0:
            st.write("• ใช้สารป้องกันเชื้อรา\n• หลีกเลี่ยงความชื้นสูง\n• กำจัดใบที่เป็นโรค")
        elif predicted_class == 1:
            st.write("• ใช้สารป้องกันสนิม\n• เพิ่มการระบายอากาศในแปลง\n• ไม่ปลูกซ้ำพื้นที่เดิม")
        elif predicted_class == 2:
            st.write("• ฉีดพ่นสารป้องกันเชื้อรา\n• หลีกเลี่ยงการรดน้ำบนใบ\n• ควบคุมความหนาแน่น")
        else:
            st.write("✅ พืชสมบูรณ์ดี ดูแลตามปกติ")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("📌 ระบบต้นแบบเพื่อการศึกษา | Corn Disease Classification with CNN")

