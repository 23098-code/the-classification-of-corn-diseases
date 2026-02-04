import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------
# ตั้งค่าหน้าเว็บ
# -----------------------
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพด")
st.write("อัปโหลดภาพใบข้าวโพด แล้วกดปุ่มวิเคราะห์")

# -----------------------
# โหลดโมเดล
# -----------------------
@st.cache_resource
def load_cnn_model():
    return load_model("model.h5")

try:
    model = load_cnn_model()
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# -----------------------
# ชื่อคลาส (ต้องตรงกับตอน train)
# -----------------------
class_names = [
    "Blight (ใบไหม้)",
    "Common Rust (สนิมใบ)",
    "Grey Leaf Spot (จุดเทา)",
    "Healthy (ใบปกติ)"
]

# -----------------------
# อัปโหลดรูป
# -----------------------
uploaded_file = st.file_uploader(
    "📤 อัปโหลดภาพใบข้าวโพด",
    type=["jpg", "jpeg", "png", "jfif", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    if st.button("🔍 วิเคราะห์ภาพ"):
        try:
            # -----------------------
            # เตรียมรูปให้ตรงกับโมเดล
            # -----------------------
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1,128,128,3)

            # -----------------------
            # ทำนาย
            # -----------------------
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # -----------------------
            # แสดงผล
            # -----------------------
            st.success(
                f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**"
            )
            st.info(
                f"📊 ความมั่นใจของโมเดล: **{confidence:.2f}%**"
            )

            # แสดงความน่าจะเป็นทุกคลาส
            st.subheader("🔢 ความน่าจะเป็นแต่ละโรค")
            for i, prob in enumerate(prediction[0]):
                st.write(f"- {class_names[i]}: {prob*100:.2f}%")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
