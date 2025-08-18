import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import av

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("final.pt")  # Change path if needed

model = load_model()

# Define video processor for real-time webcam feed
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # Convert frame to numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = self.model(img)

        # Draw detections
        for r in results:
            im_array = r.plot()  # OpenCV array with boxes + labels

        return av.VideoFrame.from_ndarray(im_array, format="bgr24")

# Streamlit UI
st.title("🔍 Real-Time Human Detection in Thermal Images")
st.write("Detect humans in real-time using your webcam with the trained YOLO model.")

# Webcam streamer
webrtc_streamer(
    key="yolo-webcam",
    video_processor_factory=YoloVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
