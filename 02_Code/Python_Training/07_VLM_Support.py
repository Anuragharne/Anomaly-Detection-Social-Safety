import cv2
import torch
import threading
import time
import requests
import numpy as np
import customtkinter as ctk
from google import genai # NEW: Updated Google SDK
from tkinter import filedialog
from PIL import Image
from scipy.spatial.distance import cdist
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO
import os

os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BOT_TOKEN = "" # Your bot token
CHAT_ID = ""   # Your chat id 

# ADD YOUR GEMINI API KEY HERE
GEMINI_API_KEY = ""
# NEW: Initialize the updated Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

IP_CAMERA_URL = "http://:8080/video" 

MODEL_PATH = r"03_Models\VideoMAE_Model"
CONFIDENCE_THRESHOLD = 0.80  
VIOLENCE_COOLDOWN = 30          

PROXIMITY_RADIUS = 80    
MIN_CLUSTER_SIZE = 35       
CROWD_COOLDOWN = 30        

# ==========================================
# 2. GLOBAL STATE & THREAD LOCKS
# ==========================================
gpu_lock = threading.Lock()

# --- Live Feed State ---
live_active = False
live_raw_frame = None
live_pil_image = None
live_status_text = "System Disarmed"

live_mae_buffer = deque(maxlen=16)
live_evidence_buffer = deque(maxlen=60) # Stores tuples of (frame, timestamp)

shared_violence_score = 0.0
shared_violence_label = "SAFE (0.00)"
shared_violence_color = (0, 255, 0)

is_sending_violence = False
is_sending_crowd = False
last_violence_time = 0
last_crowd_time = 0

# --- Upload Feed State ---
upload_active = False
upload_pil_image = None
upload_status_text = "Ready for Analysis"

# ==========================================
# 3. HELPER FUNCTIONS (VLM & DISPATCH)
# ==========================================
def verify_with_gemini(frames_buffer):
    """Extracts 5 frames and asks Gemini for semantic context."""
    try:
        total_frames = len(frames_buffer)
        if total_frames < 5: 
            return False, "Not enough frames for context."
            
        # Extract 5 evenly spaced frames
        indices = [int(i * (total_frames - 1) / 4) for i in range(5)]
        
        pil_images = []
        for idx in indices:
            frame = frames_buffer[idx][0] # Grab frame from (frame, time) tuple
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_frame))
            
        prompt = "Analyze these 5 sequential security camera frames. Is this a genuine violent physical altercation or a dangerous threat? Answer strictly YES or NO on the first line, followed by a one-sentence justification based on body language and context."
        
        # NEW: Updated generation call for the new SDK
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt] + pil_images
        )
        text = response.text.strip().upper()
        
        if text.startswith("YES"):
            return True, text
        return False, text
    except Exception as e:
        print(f"VLM API Error: {e}")
        return False, "API Error"

def capture_and_verify_violence(buffer_copy, width, height):
    """Background thread to handle VLM check and Video Compilation to avoid freezing UI."""
    global is_sending_violence
    
    print("Violence Tripped Local AI. Querying Gemini VLM...")
    is_violent, reason = verify_with_gemini(buffer_copy)
    print(f"Gemini VLM Response: {reason}")
    
    if is_violent:
        # Calculate actual FPS to fix the fast-forward bug
        duration = buffer_copy[-1][1] - buffer_copy[0][1]
        actual_fps = len(buffer_copy) / duration if duration > 0 else 10.0
        safe_fps = max(5.0, min(30.0, actual_fps))
        
        filename = f"evidence_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, safe_fps, (width, height))
        for frame, timestamp in buffer_copy:
            out.write(frame)
        out.release()
        
        # Send Video to Telegram
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
            caption = f"🔴 **VIOLENCE VERIFIED BY VLM**\nLocation: Command Center Feed\nContext: {reason}"
            with open(filename, 'rb') as video_file:
                requests.post(url, files={'video': video_file}, data={'chat_id': CHAT_ID, 'caption': caption})
            print("Video Evidence Dispatched.")
        except Exception as e:
            print(f"Telegram Failed: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                
    is_sending_violence = False

def send_telegram_alert(alert_type, extra_info=""):
    """For standard text alerts like Crowd Crush."""
    global is_sending_crowd
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        text = f"🟡 **CROWD CRUSH ALERT**\nLocation: Command Center Feed\nDetails: {extra_info}"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': text})
    except Exception as e:
        print(f"Alert Failed: {e}")
    finally:
        is_sending_crowd = False

# ==========================================
# 4. LOAD AI MODELS
# ==========================================
print("Initializing AI Core (Hardware Accelerated)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
videomae_model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
videomae_model.eval()
videomae_model.to(device)

yolo_model = YOLO("yolov8n.pt") 
yolo_model.to(device) 
print(f"AI Core Online ({device.upper()})")

# ==========================================
# 5. LIVE PIPELINE (DECOUPLED THREADS)
# ==========================================
def camera_reader_thread():
    global live_active, live_raw_frame
    
    url = int(IP_CAMERA_URL) if str(IP_CAMERA_URL).isdigit() else IP_CAMERA_URL
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while live_active:
        ret, frame = cap.read()
        if ret:
            live_raw_frame = frame
        else:
            time.sleep(0.1) 
            
    cap.release()

def live_processing_thread():
    global live_active, live_raw_frame, live_pil_image, live_status_text
    global is_sending_crowd, last_crowd_time
    global shared_violence_label, shared_violence_color, is_sending_violence, last_violence_time
    global live_evidence_buffer
    
    while live_active:
        if live_raw_frame is None:
            time.sleep(0.01)
            continue
            
        frame = live_raw_frame.copy()
        height, width, _ = frame.shape
        annotated_frame = frame.copy()
        current_time = time.time()
        
        # Add frame to memory buffer with timestamp
        live_evidence_buffer.append((frame.copy(), current_time))
        
        # --- YOLO Spatial (Crowd Crush Only) ---
        with gpu_lock:
            results = yolo_model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
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

        if dense_people_count >= MIN_CLUSTER_SIZE:
            if not is_sending_crowd and (current_time - last_crowd_time > CROWD_COOLDOWN):
                is_sending_crowd = True
                last_crowd_time = current_time
                threading.Thread(target=send_telegram_alert, args=("CROWD", f"{dense_people_count} clustered.")).start()

        # --- VideoMAE Temporal (Violence Detection) ---
        model_input = cv2.resize(frame, (224, 224))
        live_mae_buffer.append(model_input)
        
        if len(live_mae_buffer) == 16:
            inputs = processor(list(live_mae_buffer), return_tensors="pt")
            inputs = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            
            with gpu_lock:
                with torch.inference_mode():
                    outputs = videomae_model(**inputs)
                    probs = outputs.logits.softmax(dim=1)
                    score = probs[0][1].item() 
            
            # Pure Trained Model Trigger (YOLO Decoupled)
            if score > CONFIDENCE_THRESHOLD:
                shared_violence_label = f"VLM ANALYZING ({score:.2f})"
                shared_violence_color = (0, 165, 255) # Orange while VLM checks
                
                if not is_sending_violence and (current_time - last_violence_time > VIOLENCE_COOLDOWN):
                    is_sending_violence = True
                    last_violence_time = current_time
                    buffer_copy = list(live_evidence_buffer)
                    threading.Thread(target=capture_and_verify_violence, args=(buffer_copy, width, height)).start()
            else:
                if not is_sending_violence: # Don't overwrite orange label if currently checking
                    shared_violence_label = f"SAFE ({score:.2f})"
                    shared_violence_color = (0, 255, 0) 

        # UI Drawing
        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"STATUS: {shared_violence_label}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shared_violence_color, 2)
        c_color = (0, 0, 255) if dense_people_count >= MIN_CLUSTER_SIZE else (255, 255, 255)
        cv2.putText(annotated_frame, f"CROWD DENSITY: {dense_people_count} People", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
        
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        h, w = img_rgb.shape[:2]
        if w > 800:
            scale = 800 / w
            img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))

        live_pil_image = Image.fromarray(img_rgb)
        live_status_text = f"SYSTEM ACTIVE | Crowd: {dense_people_count} | Mode: Semantic Gatekeeper"
        
        time.sleep(0.01)

# ==========================================
# 6. UPLOAD PIPELINE (ISOLATED THREAD)
# ==========================================
def upload_processing_thread(filepath):
    global upload_active, upload_pil_image, upload_status_text
    
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    upload_mae_buf = deque(maxlen=16)
    
    local_label = "Scanning..."
    local_color = (0, 255, 0)
    frame_count = 0
    
    while upload_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_count += 1
        annotated_frame = frame.copy()
        
        with gpu_lock:
            results = yolo_model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
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

        model_input = cv2.resize(frame, (224, 224))
        upload_mae_buf.append(model_input)
        
        if len(upload_mae_buf) == 16 and frame_count % 4 == 0:
            inputs = processor(list(upload_mae_buf), return_tensors="pt")
            inputs = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            
            with gpu_lock:
                with torch.inference_mode():
                    outputs = videomae_model(**inputs)
                    probs = outputs.logits.softmax(dim=1)
                    score = probs[0][1].item() 
            
            if score > CONFIDENCE_THRESHOLD:
                local_label = f"VIOLENCE! ({score:.2f})"
                local_color = (0, 0, 255) 
            else:
                local_label = f"SAFE ({score:.2f})"
                local_color = (0, 255, 0) 

        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"STATUS: {local_label}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, local_color, 2)
        c_color = (0, 0, 255) if dense_people_count >= MIN_CLUSTER_SIZE else (255, 255, 255)
        cv2.putText(annotated_frame, f"CROWD DENSITY: {dense_people_count} People", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)

        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        if w > 800:
            scale = 800 / w
            img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))

        upload_pil_image = Image.fromarray(img_rgb)
        upload_status_text = f"ANALYZING | Crowd: {dense_people_count} | Status: {local_label}"
        
        time.sleep(0.03)

    cap.release()
    upload_active = False
    upload_status_text = "Analysis Complete."

# ==========================================
# 7. NATIVE DESKTOP UI (MODERNIZED)
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Social Safety Intelligence Command Center")
app.geometry("1150x800")
app.configure(fg_color="#0d1117") 

FONT_HEADER = ("Segoe UI", 20, "bold")
FONT_STATUS = ("Segoe UI", 16, "bold")
FONT_BUTTON = ("Segoe UI", 14, "bold")

tabview = ctk.CTkTabview(app, width=1050, height=720, corner_radius=15, 
                         fg_color="#161b22", border_width=2, border_color="#30363d")
tabview.pack(padx=20, pady=20, expand=True, fill="both")

tab_live = tabview.add("Live Sentinel Mode")
tab_upload = tabview.add("Forensic Video Analysis")

# --- LIVE UI ---
live_monitor_frame = ctk.CTkFrame(tab_live, corner_radius=10, fg_color="#000000", 
                                  border_width=3, border_color="#1f6feb")
live_monitor_frame.pack(pady=20, padx=20)

live_video_label = ctk.CTkLabel(live_monitor_frame, text="SYSTEM OFFLINE", text_color="#58a6ff", 
                                font=FONT_HEADER, width=800, height=450, fg_color="transparent")
live_video_label.pack(padx=5, pady=5)

live_status_label = ctk.CTkLabel(tab_live, text="Status: Awaiting Command", 
                                 font=FONT_STATUS, text_color="#8b949e")
live_status_label.pack(pady=(0, 15))

live_btn_frame = ctk.CTkFrame(tab_live, fg_color="transparent")
live_btn_frame.pack(pady=5)

def start_live():
    global live_active
    if live_active: return
    live_active = True
    live_status_label.configure(text="Connecting to IP Camera...", text_color="#58a6ff")
    live_monitor_frame.configure(border_color="#238636") 
    
    threading.Thread(target=camera_reader_thread, daemon=True).start()
    threading.Thread(target=live_processing_thread, daemon=True).start()

def stop_live():
    global live_active, live_pil_image
    live_active = False
    live_pil_image = None
    # FIX: Use None instead of "" to prevent CTkImage warnings
    live_video_label.configure(image=None, text="SYSTEM OFFLINE")
    live_status_label.configure(text="System Disarmed", text_color="#8b949e")
    live_monitor_frame.configure(border_color="#1f6feb") 

btn_start_live = ctk.CTkButton(live_btn_frame, text="ARM SYSTEM", font=FONT_BUTTON, 
                               width=160, height=40, corner_radius=8,
                               fg_color="#238636", hover_color="#2ea043",
                               border_width=2, border_color="#3fb950",
                               command=start_live)
btn_start_live.pack(side="left", padx=15)

btn_stop_live = ctk.CTkButton(live_btn_frame, text="DISARM SYSTEM", font=FONT_BUTTON, 
                              width=160, height=40, corner_radius=8,
                              fg_color="#da3633", hover_color="#f85149",
                              border_width=2, border_color="#ff7b72",
                              command=stop_live)
btn_stop_live.pack(side="left", padx=15)

# --- UPLOAD UI ---
upload_monitor_frame = ctk.CTkFrame(tab_upload, corner_radius=10, fg_color="#000000", 
                                    border_width=3, border_color="#8957e5")
upload_monitor_frame.pack(pady=20, padx=20)

upload_video_label = ctk.CTkLabel(upload_monitor_frame, text="NO MEDIA LOADED", text_color="#d2a8ff", 
                                  font=FONT_HEADER, width=800, height=450, fg_color="transparent")
upload_video_label.pack(padx=5, pady=5)

upload_status_label = ctk.CTkLabel(tab_upload, text="Ready for Analysis", 
                                   font=FONT_STATUS, text_color="#8b949e")
upload_status_label.pack(pady=(0, 15))

def select_and_run_video():
    global upload_active
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path: return
    
    upload_active = False 
    time.sleep(0.1)       
    
    upload_active = True
    upload_status_label.configure(text="Initializing File Processing...", text_color="#d2a8ff")
    threading.Thread(target=upload_processing_thread, args=(file_path,), daemon=True).start()

btn_upload = ctk.CTkButton(tab_upload, text="SELECT VIDEO & ANALYZE", font=FONT_BUTTON,
                           width=220, height=45, corner_radius=8,
                           fg_color="#6e40c9", hover_color="#8957e5",
                           border_width=2, border_color="#a371f7",
                           command=select_and_run_video)
btn_upload.pack(pady=5)

def master_ui_loop():
    # FIX: Use High-DPI CTkImage instead of ImageTk.PhotoImage
    if live_active and live_pil_image is not None:
        imgtk = ctk.CTkImage(light_image=live_pil_image, size=(live_pil_image.width, live_pil_image.height))
        live_video_label.imgtk = imgtk
        live_video_label.configure(image=imgtk, text="")
        live_status_label.configure(text=live_status_text)
        
    if upload_active and upload_pil_image is not None:
        imgtk2 = ctk.CTkImage(light_image=upload_pil_image, size=(upload_pil_image.width, upload_pil_image.height))
        upload_video_label.imgtk = imgtk2
        upload_video_label.configure(image=imgtk2, text="")
        upload_status_label.configure(text=upload_status_text)
        
    app.after(33, master_ui_loop)

master_ui_loop()
app.mainloop()