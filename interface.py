import streamlit as st
import cv2
import numpy as np
import librosa
from keras.models import load_model
from ultralytics import YOLO
import tempfile
import os

# Load models
human_model = YOLO("models/yolov8n.pt")
weapon_model = YOLO("detect/train/weights/best.pt")
gunshot_model = load_model('models/best_model.h5')

# Class mapping for gunshot detection
class_mapping = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'gun_shot',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

def detect_poacher(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output_with_detections.mp4'
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

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def detect_gunshot(audio_path):
    audiodata, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=audiodata, sr=sample_rate).T, axis=0)
    features = np.array([mels])
    features = features / np.max(features)
    predictions = gunshot_model.predict(features)
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    predicted_class = class_mapping[predicted_class_id]
    confidence = predictions[0][predicted_class_id]
    return predicted_class, confidence, audiodata, sample_rate

st.title("Poacher and Gunshot Detection")

st.header("Video Poacher Detection")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_path = temp_video_file.name
    output_video_path = detect_poacher(temp_video_path)
    st.success("Video processed successfully!")
    st.write("Download the processed video:")
    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Download Video",
            data=file,
            file_name="output_with_detections.mp4",
            mime="video/mp4"
        )

st.header("Audio Gunshot Detection")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(audio_file.read())
        temp_audio_path = temp_audio_file.name
    predicted_class, confidence, audiodata, sample_rate = detect_gunshot(temp_audio_path)
    st.write(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f})")
    if predicted_class == 'gun_shot':
        st.write("Gunshot detected!")
    else:
        st.write("No gunshot detected.")
    st.audio(temp_audio_path)