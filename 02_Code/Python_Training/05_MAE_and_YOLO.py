import cv2
import torch
import threading
import time
import requests
import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO
import os

os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

# ==========================================
# CONFIGURATION
# ==========================================
BOT_TOKEN = "" #your bot token
CHAT_ID = "" #your chat id 

# UPDATE THIS WITH YOUR MOBILE HOTSPOT IP 
IP_CAMERA_URL = "http://Ip address:8080/video" 

# VIDEOMAE (Violence)
MODEL_PATH = r"03_Models\VideoMAE_Model"
CONFIDENCE_THRESHOLD = 0.90  
VIOLENCE_COOLDOWN = 15          

# YOLO (Crowd Crush / Density)
PROXIMITY_RADIUS = 150     
MIN_CLUSTER_SIZE = 5       
CROWD_COOLDOWN = 30        

# --- NEW: DYNAMIC THRESHOLD LOGIC ---
CROWD_DILUTION_COUNT = 10       # If this many people are in frame, lower the threshold
LOWERED_CONFIDENCE = 0.60       # The new, more sensitive threshold for crowds

# ==========================================
# THREAD SHARED VARIABLES
# ==========================================
shared_violence_score = 0.0
shared_violence_label = "Initializing MAE..."
shared_violence_color = (0, 255, 0)
video_mae_buffer = []
evidence_buffer = deque(maxlen=100)
is_sending_violence = False
is_sending_crowd = False
last_violence_time = 0

# --- NEW: SHARED THRESHOLD ---
shared_dynamic_threshold = CONFIDENCE_THRESHOLD 

# ==========================================
# CLASS: FRESH FRAME 
# ==========================================
class FreshFrame:
    def __init__(self, url):
        self.url = url
        if str(url).isdigit():
            self.cap = cv2.VideoCapture(int(url)) 
        else:
            self.cap = cv2.VideoCapture(self.url) 
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"📡 Camera Thread Started: {self.url}")

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                if ret:
                    self.ret = True
                    self.frame = frame
                else:
                    self.ret = False
                    self.cap.release()
                    time.sleep(1)
                    if not str(self.url).isdigit():
                        self.cap.open(self.url)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# ==========================================
# HELPER: TELEGRAM & EVIDENCE
# ==========================================
def send_telegram_alert(video_path, alert_type, extra_info=""):
    global is_sending_violence, is_sending_crowd
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
        if alert_type == "VIOLENCE":
            caption = "**VIOLENCE DETECTED**\nLocation: Live Feed\nStatus: Immediate Action Required"
        else:
            caption = f"**CROWD CRUSH ALERT**\nLocation: Live Feed\nDetails: {extra_info}"
            
        with open(video_path, 'rb') as video_file:
            requests.post(url, files={'video': video_file}, data={'chat_id': CHAT_ID, 'caption': caption})
        print(f"{alert_type} ALERT DELIVERED.")
    except Exception as e:
        print(f"FAILED TO SEND ALERT: {e}")
    finally:
        if alert_type == "VIOLENCE":
            is_sending_violence = False
        else:
            is_sending_crowd = False

def save_evidence(frames, width, height, fps=20.0):
    filename = f"evidence_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return filename

# ==========================================
# THREAD: ASYNCHRONOUS VIDEOMAE 
# ==========================================
def videomae_worker(processor, model, device, width, height):
    global video_mae_buffer, shared_violence_score, shared_violence_label, shared_violence_color
    global is_sending_violence, last_violence_time, evidence_buffer
    global shared_dynamic_threshold # Pull in the dynamic threshold

    print("VideoMAE Thread Started.")
    prediction_buffer = deque(maxlen=5)

    while True:
        if len(video_mae_buffer) >= 16:
            frames_to_process = video_mae_buffer[:16]
            inputs = processor(list(frames_to_process), return_tensors="pt")
            
            inputs = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model(**inputs)
                probs = outputs.logits.softmax(dim=1)
                score = probs[0][1].item() 
            
            prediction_buffer.append(score)
            avg_score = sum(prediction_buffer) / len(prediction_buffer)

            shared_violence_score = avg_score
            
            # --- MODIFIED: Evaluate against the shifting dynamic threshold ---
            if avg_score > shared_dynamic_threshold:
                shared_violence_label = f"VIOLENCE! ({avg_score:.2f})"
                shared_violence_color = (0, 0, 255) 
                
                current_time = time.time()
                if not is_sending_violence and (current_time - last_violence_time > VIOLENCE_COOLDOWN):
                    is_sending_violence = True
                    last_violence_time = current_time
                    ev_path = save_evidence(list(evidence_buffer), width, height)
                    threading.Thread(target=send_telegram_alert, args=(ev_path, "VIOLENCE")).start()
            else:
                shared_violence_label = f"SAFE ({avg_score:.2f})"
                shared_violence_color = (0, 255, 0) 

            video_mae_buffer.pop(0)
        else:
            time.sleep(0.01)

# ==========================================
# MAIN SYSTEM 
# ==========================================
def main():
    global video_mae_buffer, evidence_buffer, is_sending_crowd
    global shared_dynamic_threshold # Pull in the dynamic threshold
    last_crowd_time = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing System on: {device.upper()}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
    videomae_model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
    videomae_model.eval()
    videomae_model.to(device)
    
    yolo_model = YOLO("yolov8n.pt") 
    yolo_model.to(device) 

    cam = FreshFrame(IP_CAMERA_URL)
    time.sleep(1) 
    ret, frame = cam.read()
    if not ret:
        print("CRITICAL ERROR: Could not connect to camera.")
        return

    height, width, _ = frame.shape
    
    mae_thread = threading.Thread(target=videomae_worker, args=(processor, videomae_model, device, width, height))
    mae_thread.daemon = True
    mae_thread.start()

    print("DUAL-AI SYSTEM ARMED & READY.")

    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.01)
            continue

        evidence_buffer.append(frame.copy())
        
        model_input = cv2.resize(frame, (224, 224))
        if len(video_mae_buffer) < 16:
            video_mae_buffer.append(model_input)
        
        annotated_frame = frame.copy()
        
        # --- YOLO CROWD DENSITY MATH ---
        results = yolo_model(frame, classes=[0], verbose=False) # 0 = person
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # --- NEW: UPDATE DYNAMIC THRESHOLD ---
        total_people_in_frame = len(boxes)
        if total_people_in_frame >= CROWD_DILUTION_COUNT:
            shared_dynamic_threshold = LOWERED_CONFIDENCE
        else:
            shared_dynamic_threshold = CONFIDENCE_THRESHOLD

        dense_people_count = 0
        
        if len(boxes) > 0:
            centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1, y1, x2, y2) in boxes])
            distances = cdist(centers, centers, 'euclidean')
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                neighbors = np.sum(distances[i] < PROXIMITY_RADIUS) - 1 
                
                if neighbors >= MIN_CLUSTER_SIZE:
                    dense_people_count += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

        current_time = time.time()
        if dense_people_count >= MIN_CLUSTER_SIZE:
            if not is_sending_crowd and (current_time - last_crowd_time > CROWD_COOLDOWN):
                is_sending_crowd = True
                last_crowd_time = current_time
                alert_text = f"{dense_people_count} people detected in a high-density crush zone."
                print(f"📸 {alert_text}")
                ev_path = save_evidence(list(evidence_buffer), width, height)
                threading.Thread(target=send_telegram_alert, args=(ev_path, "CROWD", alert_text)).start()

        # --- DASHBOARD OVERLAY ---
        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        
        # --- MODIFIED: Display active threshold to judges ---
        cv2.putText(annotated_frame, f"VIOLENCE: {shared_violence_label} [Active Thresh: {shared_dynamic_threshold:.2f}]", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shared_violence_color, 2)
        
        crowd_color = (0, 0, 255) if dense_people_count >= MIN_CLUSTER_SIZE else (255, 255, 255)
        cv2.putText(annotated_frame, f"CROWD DENSITY: {dense_people_count} People in Danger Zone", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, crowd_color, 2)
        
        cv2.imshow("Dual-AI Security Feed", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()