import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model
model = load_model('mask_detector_model.h5')

# UI
st.title("Face Mask Detector")
st.write("Upload a photo to check if the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # RGBA to RGB
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    label = "With Mask" if pred < 0.5 else "Without Mask"
    confidence = (1 - pred) if pred < 0.5 else pred
    st.write(f"### Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}")
