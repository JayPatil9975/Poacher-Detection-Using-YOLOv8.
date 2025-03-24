import cv2
import numpy as np
import librosa
from keras.models import load_model
from ultralytics import YOLO
import sounddevice as sd
import threading

human_model = YOLO("models/yolov8n.pt")
weapon_model = YOLO("detect/train/weights/best.pt")
gunshot_model = load_model('models/best_model.h5')

class_mapping = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

gunshot_detected = False
gunshot_lock = threading.Lock()

def detect_poacher_live():
    cap = cv2.VideoCapture(0)
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
                box = detection.xyxy[0].numpy().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
                cv2.putText(frame, f"Human: {confidence:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for detection in weapon_results[0].boxes:
            confidence = float(detection.conf[0])
            weapon_detected = True
            box = detection.xyxy[0].numpy().astype(int)
            weapon_boxes.append((box[0], box[1], box[2], box[3], confidence))

        if person_detected and weapon_detected:
            for (x1, y1, x2, y2, confidence) in weapon_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = "Poacher"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        with gunshot_lock:
            if gunshot_detected:
                cv2.putText(frame, "Gunshot detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Live Poacher Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_gunshot_live():
    global gunshot_detected

    def audio_callback(indata, frames, time, status):
        global gunshot_detected
        if status:
            print(status)
        audiodata = indata[:, 0]
        mels = np.mean(librosa.feature.melspectrogram(y=audiodata, sr=44100).T, axis=0)
        features = np.array([mels])
        features = features / np.max(features)
        predictions = gunshot_model.predict(features)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        predicted_class = class_mapping[predicted_class_id]
        confidence = predictions[0][predicted_class_id]
        print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
        
        if predicted_class == 'gun_shot':
            print("Gunshot detected!")
            with gunshot_lock:
                gunshot_detected = True

    device_id = sd.default.device[0]

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, device=device_id):
        print("Listening for gunshots... Press Ctrl+C to stop.")
        sd.sleep(1000000)

if __name__ == "__main__":
    video_thread = threading.Thread(target=detect_poacher_live)
    audio_thread = threading.Thread(target=detect_gunshot_live)

    video_thread.start()
    audio_thread.start()

    video_thread.join()
    audio_thread.join()
