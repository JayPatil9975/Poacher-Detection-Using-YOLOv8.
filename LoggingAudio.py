import tensorflow as tf
import numpy as np
import librosa
import os
import sounddevice as sd
import queue
import time
from scipy.signal import butter, lfilter

TARGET_SR = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 5
SAMPLE_RATE = 22050
CONFIDENCE_THRESHOLD = 0.5
BUFFER_SIZE = int(DURATION * SAMPLE_RATE)
CHANNELS = 1
DTYPE = 'float32'

CLASS_THRESHOLDS = {
    "ambient": 0.5,
    "chainsaw": 0.5,
    "gunshot": 0.65
}

last_gunshot_time = 0
MIN_GUNSHOT_INTERVAL = 5
gunshot_counter = 0
REQUIRED_CONSECUTIVE_DETECTIONS = 2

q = queue.Queue()
audio_buffer = np.zeros(BUFFER_SIZE, dtype=DTYPE)

def load_model(model_path):
    return tf.saved_model.load(model_path)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=300, highcut=8000, fs=TARGET_SR, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    q.put(indata.copy())

def update_buffer(new_data):
    global audio_buffer
    new_data = new_data.flatten()
    if len(new_data) >= BUFFER_SIZE:
        audio_buffer = new_data[-BUFFER_SIZE:]
    else:
        shift_size = min(len(new_data), BUFFER_SIZE)
        audio_buffer = np.roll(audio_buffer, -shift_size)
        audio_buffer[-shift_size:] = new_data[-shift_size:]
    return audio_buffer

def validate_gunshot(audio):
    """Simplified validation for gunshot sounds"""
    try:
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        rms = np.sqrt(np.mean(np.square(audio)))
        
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
        spec_sum = np.sum(spec)
        if spec_sum > 0:
            centroid = np.sum(freqs * spec) / spec_sum
        else:
            centroid = 0
            
        peak_indices = []
        for i in range(1, len(audio)-1):
            if audio[i] > audio[i-1] and audio[i] > audio[i+1] and audio[i] > 0.5:
                peak_indices.append(i)
        
        diff = np.diff(np.abs(audio))
        transient_count = np.sum(diff > 0.1)
        
        window_size = 1024
        hop_size = 512
        spec_flux = 0
        
        if len(audio) >= window_size + hop_size:
            for i in range(0, len(audio) - window_size, hop_size):
                window1 = audio[i:i+window_size]
                window2 = audio[i+hop_size:i+hop_size+window_size]
                
                if len(window2) == window_size:
                    spec1 = np.abs(np.fft.rfft(window1))
                    spec2 = np.abs(np.fft.rfft(window2))
                    spec_flux += np.mean(np.abs(spec2 - spec1))
        
        has_energy = rms > 0.05
        has_high_freq = centroid > 1000
        has_peaks = len(peak_indices) > 0
        has_transients = transient_count > 10
        has_flux = spec_flux > 0.1
        
        validation_score = sum([has_energy, has_high_freq, has_peaks, has_transients, has_flux])
        return validation_score >= 3
        
    except Exception as e:
        return True

def preprocess_audio(audio, target_sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    if len(audio) < target_sr * 0.5:
        return None
        
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.T)
    
    audio = bandpass_filter(audio)
    audio = librosa.util.normalize(audio)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=target_sr, 
        n_mels=n_mels, 
        hop_length=hop_length,
        n_fft=2048,
        window='hann'
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < 32:
        mel_spec_db = librosa.util.fix_length(mel_spec_db, size=32, axis=1)
    elif mel_spec_db.shape[1] > 32:
        excess = mel_spec_db.shape[1] - 32
        start = excess // 2
        mel_spec_db = mel_spec_db[:, start:start+32]
    
    mel_spec_db = np.expand_dims(mel_spec_db, axis=[0, -1])
    
    return mel_spec_db.astype(np.float32)

def predict(model, audio, class_labels):
    if audio is None:
        return "insufficient_data", 0.0
        
    infer = model.signatures["serving_default"]
    
    try:
        output = infer(tf.constant(audio))
        probabilities = output[list(output.keys())[0]].numpy()[0]
        
        predicted_class = class_labels[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error: {e}")
        return "error", 0.0

def main():
    global gunshot_counter, last_gunshot_time
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
    
    model_path = "jagawana_v2"
    class_labels = ["ambient", "chainsaw", "gunshot"]
    
    try:
        print("Loading model...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n=== Forest Sound Detection System ===")
    print(f"Listening for sounds...")
    print("Press Ctrl+C to exit")
    print("-----------------------------------")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.5)
        ):
            last_prediction_time = 0
            while True:
                try:
                    audio_chunk = q.get(timeout=1.0)
                    
                    buffer = update_buffer(audio_chunk)
                    
                    current_time = time.time()
                    if current_time - last_prediction_time >= 2.0:
                        last_prediction_time = current_time
                        
                        processed_audio = preprocess_audio(buffer)
                        prediction, confidence = predict(model, processed_audio, class_labels)
                        
                        threshold = CLASS_THRESHOLDS.get(prediction, CONFIDENCE_THRESHOLD)
                        
                        if prediction == "gunshot" and confidence >= threshold:
                            if current_time - last_gunshot_time < MIN_GUNSHOT_INTERVAL:
                                continue
                            
                            if not validate_gunshot(buffer):
                                continue
                            
                            gunshot_counter += 1
                            
                            if gunshot_counter >= REQUIRED_CONSECUTIVE_DETECTIONS:
                                print(f"⚠️  GUNSHOT DETECTED! ({confidence:.2f})")
                                last_gunshot_time = current_time
                                gunshot_counter = 0
                            else:
                                print(f"Possible gunshot ({confidence:.2f}) - Need more confirmation")
                        
                        elif prediction == "chainsaw" and confidence >= threshold:
                            print(f"🪚 CHAINSAW DETECTED! ({confidence:.2f})")
                            gunshot_counter = 0
                        
                        elif prediction == "ambient" and confidence >= threshold:
                            gunshot_counter = 0
                        
                        else:
                            pass
                
                except queue.Empty:
                    pass
                
                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                
                except Exception as e:
                    print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error setting up audio stream: {e}")

if __name__ == "__main__":
    main()