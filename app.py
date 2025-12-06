import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model

@st.cache_resource
def load_cnn_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        url = "PUT-YOUR-GDRIVE-DIRECT-LINK-HERE"
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_cnn_model()

st.title("🌽 ระบบจำแนกโรคใบข้าวโพดด้วย CNN")
st.write("อัปโหลดรูปใบข้าวโพด แล้วระบบจะช่วยจำแนกโรคให้")


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



