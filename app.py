import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
import av

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("final.pt")  # Change path if needed

model = load_model()

# Define WebRTC configuration with STUN/TURN servers
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Public STUN server
            {
                "urls": ["turn:numb.viagenie.ca:3478"],
                "username": "your-email@example.com",  # Replace with your email
                "credential": "your-password"          # Replace with credentials from numb.viagenie.ca
            },  # Alternative public TURN server
        ]
    }
)

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

# Webcam streamer with RTC configuration
webrtc_streamer(
    key="yolo-webcam",
    video_processor_factory=YoloVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=RTC_CONFIG,
)

# Troubleshooting note
st.info(
    "If the webcam fails to connect, ensure your network allows WebRTC traffic (UDP ports) "
    "and that your webcam is not blocked by another application. "
    "You may need to configure a custom TURN server for restrictive networks. "
    "Check the browser console (F12) for WebRTC errors."
)
