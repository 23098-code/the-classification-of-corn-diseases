import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------------
# ตั้งค่าหน้าเว็บ
# -------------------------------
st.set_page_config(
    page_title="Corn Disease Classification",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 ระบบจำแนกโรคใบข้าวโพด")
st.write("ถ่ายภาพหรืออัปโหลดภาพใบข้าวโพด ระบบจะครอปใบอัตโนมัติก่อนวิเคราะห์")

# -------------------------------
# โหลดโมเดล
# -------------------------------
try:
    model = load_model("model.h5")
    st.success("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# -------------------------------
# รายชื่อคลาส (เรียงต้องตรงกับตอน train)
# -------------------------------
class_names = [
    "Blight (โรคใบไหม้)",
    "Common Rust (โรคราสนิม)",
    "Grey Leaf Spot (โรคใบจุดสีเทา)",
    "Healthy (ใบสุขภาพดี)"
]

# -------------------------------
# ฟังก์ชันครอปใบอัตโนมัติ
# -------------------------------
def auto_crop_leaf(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # เบลอ + threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # หา contour ที่ใหญ่สุด (สมมติว่าเป็นใบ)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image  # ครอปไม่ได้ ส่งรูปเดิม

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cropped = img[y:y+h, x:x+w]
    return Image.fromarray(cropped)

# -------------------------------
# เตรียมภาพเข้าโมเดล
# -------------------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# เลือกวิธีรับภาพ
# -------------------------------
option = st.radio(
    "เลือกวิธีนำเข้าภาพ",
    ["📤 อัปโหลดภาพ", "📸 ถ่ายภาพจากกล้อง"]
)

uploaded_file = None

if option == "📤 อัปโหลดภาพ":
    uploaded_file = st.file_uploader(
        "อัปโหลดภาพใบข้าวโพด",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )
else:
    uploaded_file = st.camera_input("ถ่ายภาพใบข้าวโพด")

# -------------------------------
# วิเคราะห์ภาพ
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 ภาพต้นฉบับ")
    st.image(image, use_container_width=True)

    cropped_image = auto_crop_leaf(image)

    st.subheader("✂️ ภาพหลังครอปอัตโนมัติ")
    st.image(cropped_image, use_container_width=True)

    if st.button("🔍 วิเคราะห์โรค"):
        try:
            img_array = preprocess_image(cropped_image)
            prediction = model.predict(img_array)[0]

            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)

            # ถ้าความมั่นใจต่ำกว่า 50% ไม่แสดงผล
            if confidence < 0.5:
                st.warning("⚠️ ความมั่นใจต่ำกว่า 50% กรุณาถ่ายภาพใหม่ให้เห็นใบชัดขึ้น")
                st.stop()

            st.success(f"🌱 ผลการทำนาย: **{class_names[predicted_class]}**")
            st.write(f"📊 ความมั่นใจ: **{confidence*100:.2f}%**")

            st.subheader("📈 ความน่าจะเป็นแต่ละโรค")
            for i, prob in enumerate(prediction):
                st.write(f"- {class_names[i]}: {prob*100:.2f}%")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
