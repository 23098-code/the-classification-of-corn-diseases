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
# ⚠️ แก้ชื่อได้ แต่ "จำนวนต้องตรงกับโมเดล"
# =========================
class_names = [
    "Blight (โรคใบไหม้)",
    "Common Rust (โรคราสนิม)",
    "Healthy (ใบปกติ)",
]

# =========================
# UI
# =========================
st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")
st.write("อัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

# =========================
# LOAD MODEL
# =========================
try:
    model = load_model("model.h5")
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# =========================
# CHECK MODEL OUTPUT
# =========================
num_model_classes = model.output_shape[-1]

if num_model_classes != len(class_names):
    st.error(
        f"❌ จำนวนคลาสไม่ตรงกัน\n\n"
        f"- โมเดลทำนายได้: {num_model_classes} คลาส\n"
        f"- class_names มี: {len(class_names)} ชื่อ\n\n"
        f"กรุณาแก้ class_names ให้ตรงกับโมเดล"
    )
    st.stop()

# =========================
# GET INPUT SHAPE
# =========================
_, img_height, img_width, img_channels = model.input_shape

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

# =========================
# PREDICT
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("⏳ กำลังวิเคราะห์...")

        # --- preprocess ---
        img = image.resize((img_width, img_height))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- predict ---
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))

        # --- result ---
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ โมเดลไม่มั่นใจเพียงพอ ({confidence*100:.2f}%)\n\n"
                "แนะนำ:\n"
                "- ถ่ายภาพให้ชัด\n"
                "- ใบเดียวเต็มภาพ\n"
                "- แสงสว่างเพียงพอ"
            )
        else:
            st.success(f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**")
            st.write(f"📊 ความมั่นใจของโมเดล: **{confidence*100:.2f}%**")

            st.markdown("### 🔎 ความน่าจะเป็นแต่ละคลาส")
            for i in range(num_model_classes):
                st.write(f"- {class_names[i]}: {prediction[0][i]*100:.2f}%")
