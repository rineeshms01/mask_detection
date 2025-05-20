import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import pickle
import os

# File paths
MODEL_PATH = "mask_detector_model.keras"
LB_PATH = "label_binarizer.pickle"

st.title("😷 Mask Detection App")

if not os.path.isfile(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found.")
    st.stop()

if not os.path.isfile(LB_PATH):
    st.error(f"❌ Label binarizer file '{LB_PATH}' not found.")
    st.stop()

# Load model and label binarizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(LB_PATH, "rb") as f:
    lb = pickle.load(f)

# Get labels
labels = list(lb.keys()) if isinstance(lb, dict) else list(lb.classes_)

# Upload image
uploaded_file = st.file_uploader("📤 Upload a face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    label = labels[pred_idx]
    confidence = preds[pred_idx] * 100

    # Display
    if label.lower() == "with_mask":
        st.success(f"✅ **Person is wearing a mask** ({confidence:.2f}% confidence)")
    else:
        st.error(f"❌ **Person is NOT wearing a mask** ({confidence:.2f}% confidence)")
