import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =========================
# ตั้งค่าหน้าเว็บ
# =========================
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดภาพใบข้าวโพด แล้วกดปุ่มวิเคราะห์")

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
# ดึง input shape จากโมเดล
# =========================
input_shape = model.input_shape
_, img_height, img_width, img_channels = input_shape

st.info(f"📐 โมเดลต้องการภาพขนาด {img_height}x{img_width} | Channels = {img_channels}")

# =========================
# ชื่อโรค
# ⚠️ ลำดับต้องตรงกับโมเดล
# =========================
class_info = [
    {"en": "Corn Blight", "th": "โรคใบไหม้ข้าวโพด"},
    {"en": "Corn Common Rust", "th": "โรคราสนิมข้าวโพด"},
    {"en": "Corn Gray Leaf Spot", "th": "โรคใบจุดสีเทาข้าวโพด"},
    {"en": "Healthy", "th": "ใบข้าวโพดปกติ"}
]

# =========================
# อัปโหลดภาพ (รองรับ jfif / webp)
# =========================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)

    # แปลงสีให้ตรงกับโมเดล
    if img_channels == 1:
        image_pil = image_pil.convert("L")
    else:
        image_pil = image_pil.convert("RGB")

    st.image(image_pil, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("⏳ กำลังวิเคราะห์...")

        # =========================
        # เตรียมภาพ
        # =========================
        img = image_pil.resize((img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # =========================
        # ทำนาย
        # =========================
        try:
            prediction = model.predict(img_array)
        except Exception as e:
            st.error("❌ โมเดลไม่สามารถประมวลผลภาพนี้ได้")
            st.code(str(e))
            st.stop()

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.subheader("📊 ผลการวิเคราะห์")
        st.success(
            f"🌱 ผลการทำนาย: **{class_info[predicted_class]['th']}**\n\n"
            f"({class_info[predicted_class]['en']})\n\n"
            f"📊 ความมั่นใจ: **{confidence*100:.2f}%**"
        )
