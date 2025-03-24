from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

human_model = YOLO("models/yolov8n.pt")
weapon_model = YOLO("detect/train/weights/best.pt")

video_path = "poaching videos/Untitled video - Made with Clipchamp[2].mp4"
output_path = "poaching videos/output_with_detections.mp4"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    human_results = human_model.predict(frame)
    weapon_results = weapon_model.predict(frame)

    person_detected = False
    weapon_detected = False
    weapon_boxes = []

    for detection in human_results[0].boxes:
        class_id = int(detection.cls[0])
        confidence = float(detection.conf[0])
        if class_id == 0:
            person_detected = True
            print(f"Person detected with confidence: {confidence:.2f}")
            box = detection.xyxy[0].numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            cv2.putText(frame, f"Human: {confidence:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for detection in weapon_results[0].boxes:
        confidence = float(detection.conf[0])
        weapon_detected = True
        print(f"Weapon detected with confidence: {confidence:.2f}")
        box = detection.xyxy[0].numpy().astype(int)
        weapon_boxes.append((box[0], box[1], box[2], box[3], confidence))

    if person_detected and weapon_detected:
        for (x1, y1, x2, y2, confidence) in weapon_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = "Poacher"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 0, 255), -1)  # Filled box
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text
            
        print("⚠️ Poacher detected! Both human and weapon found.")
    else:
        print("✅ No poacher detected.")

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()