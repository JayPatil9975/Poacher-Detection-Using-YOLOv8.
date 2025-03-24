from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

human_model = YOLO("yolov8n.pt")

weapon_model = YOLO("detect/train/weights/best.pt")

img_path = "iIUaddaBeeYYdJF-800x450-noPad.jpg"
img = cv2.imread(img_path)

human_results = human_model.predict(img_path)
weapon_results = weapon_model.predict(img_path)

person_detected = False
weapon_detected = False

weapon_boxes = []

for detection in human_results[0].boxes:
    class_id = int(detection.cls[0])
    confidence = float(detection.conf[0])
    if class_id == 0:
        person_detected = True
        print(f"Person detected with confidence: {confidence:.2f}")

for detection in weapon_results[0].boxes:
    confidence = float(detection.conf[0])
    weapon_detected = True
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    weapon_boxes.append((x1, y1, x2, y2, confidence))

if person_detected and weapon_detected:
    for (x1, y1, x2, y2, confidence) in weapon_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        label = "Poacher"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 0, 255), -1)  # Filled box
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text
        
    print("⚠️ Poacher detected! Both human and weapon found.")
else:
    print("✅ No poacher detected.")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()