import cv2
import mediapipe as mp
import csv
from datetime import datetime
import os
import playsound
import time  

bad_posture_start_time = None
bad_posture_detected = False
BAD_POSTURE_ALERT_DELAY = 20  # seconds


# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose and FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Create log file if not exists
log_file = 'posture_logs.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Event'])

# Function to log events
def log_event(event):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event])
    print(f"[{timestamp}] {event}")
    # playsound.playsound("alert.wav")  # Optional: Add alert sound

last_status = ""  # Track last posture/attention status

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    status = "Posture OK"  # Default status

    # Pose detection
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        torso_length = abs(avg_hip_y - avg_shoulder_y)

        if shoulder_diff > 0.05:
            status = "Side Tilt Detected"
            cv2.putText(frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif torso_length < 0.75:
            status = "Slouch Detected"
            cv2.putText(frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Posture OK", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Face detection
        results_face = face_mesh.process(rgb_frame)
        forehead, chin, left_eye_outer, right_eye_outer = None, None, None, None
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                forehead = landmarks[10]
                chin = landmarks[152]
                left_eye_outer = landmarks[33]
                right_eye_outer = landmarks[263]

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Re-calculate angles and differences
        posture_status = "Posture OK"

        if results_pose.pose_landmarks:
            if shoulder_diff > 0.05:
                posture_status = "Side Tilt Detected"
                cv2.putText(frame, posture_status, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif torso_length < 0.75:
                posture_status = "Slouch Detected"
                cv2.putText(frame, posture_status, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "Posture OK", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if forehead and chin:
            vertical_angle = abs(forehead.x - chin.x)
            if vertical_angle > 0.03:
                posture_status = "Head Tilt Detected"
                cv2.putText(frame, posture_status, (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if left_eye_outer and right_eye_outer:
            eye_y_diff = abs(left_eye_outer.y - right_eye_outer.y)
            if eye_y_diff > 0.015:
                posture_status = "Not Looking Straight"
                cv2.putText(frame, posture_status, (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            else:
                cv2.putText(frame, "Looking Straight", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Log only if posture/attention status has changed
        if posture_status != last_status:
            log_event(posture_status)
            last_status = posture_status
            # time.sleep(5)
            # playsound.playsound("alert.wav")  # Optional alert


        # Check if the posture is bad
        if posture_status not in ["Posture OK", "Looking Straight"]:
            if not bad_posture_detected:
                bad_posture_start_time = time.time()
                bad_posture_detected = True
            elif time.time() - bad_posture_start_time >= BAD_POSTURE_ALERT_DELAY:
                playsound.playsound("alert.wav")  # Alert sound after 20s
                bad_posture_start_time = time.time()  # Reset to repeat every 20s
        else:
            bad_posture_detected = False
            bad_posture_start_time = None

        # Log if status changes
        if posture_status != last_status:
            log_event(posture_status)
            last_status = posture_status


    # Show the result
    cv2.imshow("Posture & Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
