import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # change path if needed

model = load_model()

# Streamlit UI
st.title("🔍 Human Detection in Thermal Images")
st.write("Upload an image to detect humans using the trained YOLO model.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Run inference
    results = model(img_array)

    # Draw detections
    for r in results:
        im_array = r.plot()  # OpenCV array with boxes + labels

    # Convert BGR (OpenCV) → RGB (Streamlit)
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

    # Display
    st.image(im_rgb, caption="Detection Results", use_container_width=True)
