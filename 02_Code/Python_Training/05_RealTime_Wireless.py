import cv2
import torch
import numpy as np
import threading
import time
import requests
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# ==========================================
# CONFIGURATION
# ==========================================
# 1. TELEGRAM SETUP
BOT_TOKEN = "your bot token" 
CHAT_ID = "chat it"  # Your Group ID

# 2. CAMERA SETUP (Static IP Recommended)
# Ensure your phone is streaming at 1280x720 (HD) for best results
IP_CAMERA_URL = "http://your ip/video" 

# 3. MODEL SETUP
MODEL_PATH = r"03_Models\VideoMAE_Model"
CONFIDENCE_THRESHOLD = 0.90  # 90% Confidence
ALERT_COOLDOWN = 15          # Seconds between alerts

# ==========================================
# CLASS: FRESH FRAME (ANTI-LAG)
# ==========================================
class FreshFrame:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        # Optimize buffer for speed
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"Camera Thread Started: {self.url}")

    def update(self):
        while self.running:
            # Read current frame
            ret, frame = self.cap.read()
            
            with self.lock:
                if ret:
                    self.ret = True
                    self.frame = frame
                else:
                    self.ret = False
                    # If signal lost, try to reconnect
                    print("Signal Lost... Reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap.open(self.url)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ==========================================
# HELPER: TELEGRAM ALERT
# ==========================================
is_sending_alert = False

def send_telegram_alert(video_path):
    global is_sending_alert
    try:
        print(f"UPLOADING EVIDENCE: {video_path}...")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
        
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {
                'chat_id': CHAT_ID, 
                'caption': "**VIOLENCE DETECTED**\n Location: Main Hall (Live Feed)\n Status: Investigating"
            }
            requests.post(url, files=files, data=data)
        print("ALERT DELIVERED to Police Control Room.")
    except Exception as e:
        print(f" FAILED TO SEND ALERT: {e}")
    finally:
        is_sending_alert = False

def save_evidence(frames, width, height, fps=20.0):
    filename = f"evidence_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return filename

# ==========================================
# MAIN SYSTEM
# ==========================================
def main():
    global is_sending_alert
    last_alert_time = 0
    
    # 1. Load Model
    print("Loading AI Brain...")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model Loaded on {device.upper()}")

    # 2. Start Camera
    cam = FreshFrame(IP_CAMERA_URL)
    
    # Wait for first frame
    time.sleep(1)
    ret, frame = cam.read()
    if not ret:
        print("CRITICAL ERROR: Could not connect to camera. Check IP.")
        return

    height, width, _ = frame.shape
    
    # Buffers
    frames_buffer = [] 
    violence_buffer = deque(maxlen=100) # Evidence buffer (approx 5 seconds)
    prediction_buffer = deque(maxlen=5) # Smoothing
    
    print("🛡️ SYSTEM ARMED & READY.")

    while True:
        ret, frame = cam.read()
        
        if not ret:
            time.sleep(0.1)
            continue

        # Resize for Model (224x224)
        model_input = cv2.resize(frame, (224, 224))
        
        frames_buffer.append(model_input)
        violence_buffer.append(frame) # Keep high-res for evidence
        
        if len(frames_buffer) > 16:
            frames_buffer.pop(0)

        # Draw UI
        label = "Scanning..."
        color = (0, 255, 0) # Green

        # INFERENCE LOGIC (Run every time buffer is full)
        if len(frames_buffer) == 16:
            # Prepare inputs
            inputs = processor(list(frames_buffer), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits.softmax(dim=1)
                score = probs[0][1].item() # 1 = Fight
            
            prediction_buffer.append(score)
            avg_score = sum(prediction_buffer) / len(prediction_buffer)

            # ALERT LOGIC
            if avg_score > CONFIDENCE_THRESHOLD:
                label = f"VIOLENCE! ({avg_score:.2f})"
                color = (0, 0, 255) # Red
                
                current_time = time.time()
                if not is_sending_alert and (current_time - last_alert_time > ALERT_COOLDOWN):
                    is_sending_alert = True
                    last_alert_time = current_time
                    
                    # Save & Send in Background
                    print("📸 Capturing Evidence...")
                    evidence_path = save_evidence(list(violence_buffer), width, height)
                    t = threading.Thread(target=send_telegram_alert, args=(evidence_path,))
                    t.start()
            else:
                label = f"SAFE ({avg_score:.2f})"

        # Display
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Social Safety - Industrial Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()