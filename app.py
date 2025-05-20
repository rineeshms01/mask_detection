import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Page title
st.title("Face Mask Detection")

# Load pre-trained model
@st.cache_resource
def load_trained_model():
    model = load_model("mask_detector_model.h5")  # Update with your actual model filename
    return model

model = load_trained_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image to match model input shape (224x224)
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Display result
    if prediction < 0.5:
        st.markdown("### Prediction: 😷 **With Mask**")
    else:
        st.markdown("### Prediction: 😷 **Without Mask**")

