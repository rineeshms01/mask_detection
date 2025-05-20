import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model (make sure mask_detector.h5 is in the same folder)
model = load_model('mask_detector.h5')

# Image size your model expects
IMG_SIZE = 244

st.title("Mask Detector")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((IMG_SIZE, IMG_SIZE,3))
    img_array = np.array(img)

    # If image has alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Predict
    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        st.markdown("### Prediction: **With Mask**")
        st.markdown(f"Confidence: {(1 - pred) * 100:.2f}%")
    else:
        st.markdown("### Prediction: **Without Mask**")
        st.markdown(f"Confidence: {pred * 100:.2f}%")
