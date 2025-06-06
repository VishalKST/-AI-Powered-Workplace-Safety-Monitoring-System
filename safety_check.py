import cv2
import mediapipe as mp
from yolo_detector import detect_ppe

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_safety(frame, threshold):
    # ----------------- Pose Detection ------------------
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        if nose.visibility > 0.5 and nose.y > 0.6:
            cv2.putText(frame, "⚠️ Unsafe Posture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ----------------- YOLOv8 PPE Detection ------------------
    frame, detections = detect_ppe(frame)

    y_pos = 80
    for item, status in detections.items():
        label = f"{item.upper()}: {'✅' if status else '❌'}"
        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.putText(frame, label, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_pos += 30

    return frame
