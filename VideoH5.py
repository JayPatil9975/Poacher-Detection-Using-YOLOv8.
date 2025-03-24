import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/poachingdetectionVER7.h5')

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def draw_bounding_box(frame, box, label):
    (startX, startY, endX, endY) = box
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predicted_class = predict_frame(frame)
        for box, label in predicted_class:
            draw_bounding_box(frame, box, label)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

video_path = 'poaching videos/Untitled video - Made with Clipchamp.mp4'
process_video(video_path)
