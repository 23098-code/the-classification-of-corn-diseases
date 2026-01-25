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
# CLASS NAMES (ลำดับตรงกับโมเดล)
# =========================
class_names = [
    "Blight (โรคใบไหม้)",
    "Common Rust (โรคราสนิม)",
    "Grey Leaf Spot (โรคใบจุดสีเทา)",
    "Healthy (ใบปกติ)"
]

# =========================
# UI
# =========================
st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย AI")
st.write("ถ่ายรูปหรืออัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

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
# CHECK CLASS COUNT
# =========================
num_model_classes = model.output_shape[-1]
if num_model_classes != len(class_names):
    st.error(
        f"❌ จำนวนคลาสไม่ตรงกัน\n\n"
        f"- โมเดล: {num_model_classes} คลาส\n"
        f"- class_names: {len(class_names)} ชื่อ"
    )
    st.stop()

# =========================
# INPUT SHAPE
# =========================
_, img_height, img_width, _ = model.input_shape

# =========================
# IMAGE INPUT
# =========================
st.markdown("## 📷 ถ่ายรูปจากกล้อง")
camera_image = st.camera_input("เปิดกล้องเพื่อถ่ายรูปใบข้าวโพด")

st.markdown("## 📤 หรืออัปโหลดรูปภาพ")
uploaded_file = st.file_uploader(
    "รองรับ jpg, jpeg, png, jfif, webp",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

# =========================
# SELECT IMAGE SOURCE
# =========================
image = None

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

# =========================
# PREDICTION
# =========================
if image is not None:
    st.image(image, caption="ภาพที่ใช้วิเคราะห์", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        st.info("⏳ กำลังวิเคราะห์...")

        img = image.resize((img_width, img_height))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ ความมั่นใจต่ำ ({confidence*100:.2f}%)\n\n"
                "คำแนะนำ:\n"
                "- ถ่ายให้เห็นใบเดียวชัด ๆ\n"
                "- แสงสว่างเพียงพอ\n"
                "- ไม่เบลอ ไม่ไกลเกินไป"
            )
        else:
            st.success(f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**")
            st.write(f"📊 ความมั่นใจ: **{confidence*100:.2f}%**")

            st.markdown("### 🔎 ความน่าจะเป็นแต่ละโรค")
            for i in range(num_model_classes):
                st.write(f"- {class_names[i]}: {prediction[0][i]*100:.2f}%")
