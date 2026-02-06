import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# โหลดโมเดล
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("corn_disease_multilabel.keras")

model = load_model()

# ชื่อคลาส
classes = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]

st.title("🌽 ระบบตรวจจับโรคใบข้าวโพด")
st.write("อัปโหลดภาพใบข้าวโพดเพื่อวิเคราะห์โรค")

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # เตรียมภาพ
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ทำนาย
    preds = model.predict(img_array)[0]

    st.subheader("ผลการวิเคราะห์")

    threshold = 0.5
    detected = False

    for cls, score in zip(classes, preds):
        if score >= threshold:
            st.success(f"{cls} : {score:.2f}")
            detected = True

    if not detected:
        st.warning("ไม่พบโรคที่มีความมั่นใจเกินเกณฑ์")

