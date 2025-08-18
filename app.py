import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
import av

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ensure final.pt is in the same directory or update path

model = load_model()

# Define WebRTC configuration with STUN/TURN servers
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Public STUN server
            {
                "urls": ["turn:numb.viagenie.ca:3478"],
                "username": "your-email@example.com",  # Replace with your email from numb.viagenie.ca
                "credential": "your-password"          # Replace with your password from numb.viagenie.ca
            },  # Public TURN server for testing
        ]
    }
)

# Define video processor for real-time webcam feed
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        try:
            # Convert frame to numpy array (OpenCV format)
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO inference
            results = self.model(img)

            # Draw detections
            im_array = img  # Fallback in case of error
            for r in results:
                im_array = r.plot()  # OpenCV array with boxes + labels

            return av.VideoFrame.from_ndarray(im_array, format="bgr24")
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame  # Return original frame if processing fails

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
    "If the webcam fails to connect, ensure your network allows WebRTC traffic (UDP ports 3478, 19302, 49152–65535). "
    "Check that your webcam is not used by another application. "
    "Register at numb.viagenie.ca for TURN credentials or use a custom TURN server for restrictive networks. "
    "Open browser console (F12) to check for WebRTC errors."
)
