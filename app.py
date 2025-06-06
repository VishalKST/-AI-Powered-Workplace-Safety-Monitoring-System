import streamlit as st
import cv2
import numpy as np
from safety_check import detect_safety

st.set_page_config(page_title="AI Workplace Safety Monitor", layout="wide")
st.title("🦺 AI Workplace Safety Monitoring System")

# Streamlit Sidebar Controls
source = st.sidebar.radio("Select Input Source", ["Webcam", "Sample Video"])
confidence_threshold = st.sidebar.slider("Alert Confidence Threshold", 0.0, 1.0, 0.7)

# Displaying video feed
FRAME_WINDOW = st.image([])

# Webcam or video logic
if source == "Webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('data/sample_video.mp4')

st.sidebar.success("Press 'Stop' on the top to end stream.")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video source not found or ended.")
        break

    # Flip for natural view and resize
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # Process with safety check (dummy for now)
    annotated_frame = detect_safety(frame, confidence_threshold)

    # Convert to RGB for Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()
