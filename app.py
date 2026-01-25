import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

CONFIDENCE_THRESHOLD = 0.50

# =========================
# CLASS NAMES (ต้องเรียงตรงกับตอน train)
# =========================
class_names = [
    "Blight (โรคใบไหม้)",
    "Common Rust (โรคราสนิม)",
    "Healthy (ใบปกติ)"
]

# =========================
# LOAD MODEL
# =========================
st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")
st.write("อัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

try:
    model = load_model("model.h5")
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# =========================
# GET MODEL INPUT SIZE
# =========================
try:
    _, img_height, img_width, img_channels = model.input_shape
except:
    st.error("❌ ไม่สามารถอ่าน input shape ของโมเดลได้")
    st.stop()

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("⏳ กำลังวิเคราะห์...")

        # -------- PREPROCESS --------
        img = image.resize((img_width, img_height))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------- PREDICT --------
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))

        # -------- RESULT --------
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ โมเดลไม่มั่นใจเพียงพอ ({confidence*100:.2f}%)\n\n"
                "กรุณาถ่ายภาพใหม่:\n"
                "- ใบเดียวชัด ๆ\n"
                "- แสงสว่างพอ\n"
                "- ไม่เบลอ / ไม่ไกล"
            )
        else:
            st.success(f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**")
            st.write(f"📊 ความมั่นใจของโมเดล: **{confidence*100:.2f}%**")

            st.markdown("### 🔎 ความน่าจะเป็นแต่ละคลาส")
            for i, prob in enumerate(prediction[0]):
                st.write(f"- {class_names[i]}: {prob*100:.2f}%")
