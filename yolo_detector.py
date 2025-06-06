from ultralytics import YOLO
import cv2

# Load a YOLOv8 model (you can use custom trained one later)
# This model needs to be trained or fine-tuned on PPE classes
# For now, try pretrained or substitute with your custom one
model = YOLO("yolov8n.pt")  # Replace with 'helmet.pt' if you have a custom model

# You’ll need a model trained on PPE data for better results
TARGET_CLASSES = ['helmet', 'mask', 'gloves']  # Update with real class names from your model

def detect_ppe(frame):
    results = model.predict(frame, conf=0.5, verbose=False)[0]
    
    detected = {'helmet': False, 'mask': False, 'gloves': False}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name.lower() in detected:
            detected[cls_name.lower()] = True

            # Draw box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, cls_name.upper(), (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame, detected
