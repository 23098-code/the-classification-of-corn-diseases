import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดรูปใบข้าวโพด แล้วระบบจะช่วยจำแนกโรคให้")


# โหลดโมเดล (โหลดครั้งเดียว)
@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()

# คลาสของโรค (แก้ให้ตรงกับโมเดลของคุณ)
classes = ["Large Leaf Blight", "Rust", "Gray Leaf Spot", "Healthy"]


uploaded = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="ภาพที่อัปโหลด", use_column_width=True)

    img = load_img(uploaded, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    st.subheader(f"🎯 ผลการจำแนก: {classes[class_idx]}")
