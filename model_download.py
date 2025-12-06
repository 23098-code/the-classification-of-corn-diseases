import gdown
import os

def download_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
        gdown.download(url, model_path, quiet=False)
