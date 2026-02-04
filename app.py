import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ===================== CONFIG =====================
IMG_SIZE = 128
CONF_THRESHOLD = 0.5

class_names = [
    "Blight (ใบไหม้)",
    "Common Rust (สนิมใบ)",
    "Grey Leaf Spot (จุดสีเทา)",
    "Healthy (ใบปกติ)"
]

care_guide = {
    "Blight (ใบไหม้)": "ตัดใบที่เป็นโรคออก ลดความชื้น หลีกเลี่ยงน้ำค้างสะสม",
    "Common Rust (สนิมใบ)": "กำจัดวัชพืชรอบแปลง ใช้พันธุ์ต้านทาน",
    "Grey Leaf Spot (จุดสีเทา)": "หลีกเลี่ยงปลูกซ้ำที่เดิม ปรับระยะปลูกให้โปร่ง",
    "Healthy (ใบปกติ)": "ต้นข้าวโพดสุขภาพดี ดูแลตามปกติ ใส่ปุ๋ยและให้น้ำเหมาะสม"
}

# ===================== LOAD MODEL =====================
model = load_model("model.h5")

# ===================== FUNCTIONS =====================
def aggressive_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img, img  # fallback

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = img[y:y+h, x:x+w]

    return cropped, img


def preprocess_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cropped, _ = aggressive_crop(img)

    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0
    resized = np.expand_dims(resized, axis=0)

    return resized, cropped


# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Corn Disease Classification", page_icon="🌽")
st.title("🌽 ระบบจำแนกโรคใบข้าวโพด")
st.write("ระบบจะครอปใบอัตโนมัติเพื่อลดฉากหลัง")

source = st.radio("เลือกแหล่งภาพ", ["📷 กล้อง", "📁 อัปโหลดไฟล์"])

image = None

if source == "📷 กล้อง":
    camera_img = st.camera_input("ถ่ายภาพใบข้าวโพด")
    if camera_img:
        image = Image.open(camera_img)
else:
    upload = st.file_uploader(
        "อัปโหลดภาพ",
        type=["jpg", "jpeg", "png", "jfif", "webp"]
    )
    if upload:
        image = Image.open(upload)

if image:
    st.subheader("📸 ภาพต้นฉบับ")
    st.image(image, use_container_width=True)

    if st.button("🔍 วิเคราะห์"):
        try:
            img_array, cropped_img = preprocess_image(image)

            # แสดงภาพหลังครอป
            st.subheader("✂️ ภาพหลังครอป (ตัดฉากหลัง)")
            st.image(
                cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            preds = model.predict(img_array)[0]
            best_idx = np.argmax(preds)
            confidence = preds[best_idx]

            if confidence < CONF_THRESHOLD:
                st.warning("⚠️ ความมั่นใจต่ำ กรุณาถ่ายใกล้ขึ้น หรือใช้พื้นหลังเรียบ")
            else:
                label = class_names[best_idx]
                st.success(f"🌱 ผลการทำนาย: {label}")
                st.write(f"📊 ความมั่นใจ: {confidence*100:.2f}%")
                st.info(care_guide[label])

                st.subheader("📊 ความน่าจะเป็นแต่ละคลาส")
                for i, p in enumerate(preds):
                    st.write(f"- {class_names[i]}: {p*100:.2f}%")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
