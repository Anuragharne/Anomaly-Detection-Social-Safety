import cv2
import torch
import threading
import time
import requests
import numpy as np
import customtkinter as ctk
import pygetwindow as gw
import win32gui
import win32ui
import win32con
import ctypes
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.spatial.distance import cdist
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO
import os

os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BOT_TOKEN = "" #your bot token
CHAT_ID = "" #your chat id 

MODEL_PATH = r"03_Models\VideoMAE_Model"
CONFIDENCE_THRESHOLD = 0.90  
VIOLENCE_COOLDOWN = 15          

PROXIMITY_RADIUS = 150     
MIN_CLUSTER_SIZE = 5       
CROWD_COOLDOWN = 30        
CROWD_DILUTION_COUNT = 10       
LOWERED_CONFIDENCE = 0.60       

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

selected_window_title = None

shared_violence_score = 0.0
shared_violence_label = "SAFE (0.00)"
shared_violence_color = (0, 255, 0)
shared_dynamic_threshold = CONFIDENCE_THRESHOLD

is_sending_violence = False
is_sending_crowd = False
last_violence_time = 0
last_crowd_time = 0

# --- Upload Feed State ---
upload_active = False
upload_pil_image = None
upload_status_text = "Ready for Analysis"

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def send_telegram_alert(alert_type, extra_info=""):
    global is_sending_violence, is_sending_crowd
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        if alert_type == "VIOLENCE":
            text = "**VIOLENCE DETECTED**\nLocation: Command Center Feed\nStatus: Immediate Action Required"
        else:
            text = f"**CROWD CRUSH ALERT**\nLocation: Command Center Feed\nDetails: {extra_info}"
            
        requests.post(url, data={'chat_id': CHAT_ID, 'text': text})
        print(f"Alert Sent: {alert_type}")
    except Exception as e:
        print(f"Alert Failed: {e}")
    finally:
        if alert_type == "VIOLENCE":
            is_sending_violence = False
        else:
            is_sending_crowd = False

# ==========================================
# 4. OS-LEVEL TRUE WINDOW CAPTURE
# ==========================================
def capture_true_window(window_title):
    """
    Captures the exact window content directly from the Windows Graphics Layer.
    This works even if the window is behind another application (but not minimized).
    """
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        return None
        
    # Get window bounds
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top
    
    if w <= 0 or h <= 0:
        return None
        
    try:
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        
        saveDC.SelectObject(saveBitMap)
        
        # PW_RENDERFULLCONTENT (3) is required to capture Hardware Accelerated apps like Chrome/Edge
        result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img.shape = (h, w, 4)
        
        # Convert BGRA to BGR for OpenCV
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Cleanup memory to prevent memory leaks
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        return frame
    except Exception as e:
        return None

# ==========================================
# 5. LOAD AI MODELS
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
# 6. LIVE PIPELINE (DECOUPLED THREADS)
# ==========================================
def camera_reader_thread():
    """Continuously runs the OS-level capture."""
    global live_active, live_raw_frame, selected_window_title
    
    while live_active:
        if not selected_window_title or selected_window_title == "Select Target Window...":
            time.sleep(0.1)
            continue
            
        frame = capture_true_window(selected_window_title)
        if frame is not None:
            live_raw_frame = frame
            
        time.sleep(0.03) # Cap at ~30 FPS

def live_processing_thread():
    """Handles all heavy AI math in the background, keeping the UI fast."""
    global live_active, live_raw_frame, live_pil_image, live_status_text
    global shared_dynamic_threshold, is_sending_crowd, last_crowd_time
    global shared_violence_label, shared_violence_color, is_sending_violence, last_violence_time
    
    while live_active:
        if live_raw_frame is None:
            time.sleep(0.01)
            continue
            
        frame = live_raw_frame.copy()
        height, width, _ = frame.shape
        annotated_frame = frame.copy()
        
        # 1. Spatial AI (YOLO)
        with gpu_lock:
            results = yolo_model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) >= CROWD_DILUTION_COUNT:
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
                threading.Thread(target=send_telegram_alert, args=("CROWD", f"{dense_people_count} clustered.")).start()

        # 2. Temporal AI (VideoMAE)
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
            
            if score > shared_dynamic_threshold:
                shared_violence_label = f"VIOLENCE! ({score:.2f})"
                shared_violence_color = (0, 0, 255) 
                
                if not is_sending_violence and (current_time - last_violence_time > VIOLENCE_COOLDOWN):
                    is_sending_violence = True
                    last_violence_time = current_time
                    threading.Thread(target=send_telegram_alert, args=("VIOLENCE", "")).start()
            else:
                shared_violence_label = f"SAFE ({score:.2f})"
                shared_violence_color = (0, 255, 0) 

        # 3. Draw UI & Prep for Tkinter
        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"VIOLENCE: {shared_violence_label} [Thresh: {shared_dynamic_threshold:.2f}]", 
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
        live_status_text = f"SYSTEM ACTIVE | Crowd: {dense_people_count} | Status: {shared_violence_label}"
        
        time.sleep(0.01)

# ==========================================
# 7. UPLOAD PIPELINE (ISOLATED THREAD)
# ==========================================
def upload_processing_thread(filepath):
    """Processes uploaded videos smoothly without locking the UI."""
    global upload_active, upload_pil_image, upload_status_text
    
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    upload_mae_buf = deque(maxlen=16)
    
    local_label = "Scanning..."
    local_color = (0, 255, 0)
    local_threshold = CONFIDENCE_THRESHOLD
    frame_count = 0
    
    while upload_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        frame_count += 1
        annotated_frame = frame.copy()
        
        # YOLO Processing
        with gpu_lock:
            results = yolo_model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) >= CROWD_DILUTION_COUNT:
            local_threshold = LOWERED_CONFIDENCE
        else:
            local_threshold = CONFIDENCE_THRESHOLD

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

        # MAE Processing
        model_input = cv2.resize(frame, (224, 224))
        upload_mae_buf.append(model_input)
        
        if len(upload_mae_buf) == 16 and frame_count % 4 == 0:
            inputs = processor(list(upload_mae_buf), return_tensors="pt")
            # FIXED SYNTAX BUG HERE
            inputs = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
            
            with gpu_lock:
                with torch.inference_mode():
                    outputs = videomae_model(**inputs)
                    probs = outputs.logits.softmax(dim=1)
                    score = probs[0][1].item() 
            
            if score > local_threshold:
                local_label = f"VIOLENCE! ({score:.2f})"
                local_color = (0, 0, 255) 
            else:
                local_label = f"SAFE ({score:.2f})"
                local_color = (0, 255, 0) 

        # Draw UI
        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"VIOLENCE: {local_label} [Thresh: {local_threshold:.2f}]", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, local_color, 2)
        c_color = (0, 0, 255) if dense_people_count >= MIN_CLUSTER_SIZE else (255, 255, 255)
        cv2.putText(annotated_frame, f"CROWD DENSITY: {dense_people_count} People", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)

        # Output to Tkinter
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
# 8. NATIVE DESKTOP UI
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Social Safety Intelligence Command Center")
app.geometry("1100x750")

tabview = ctk.CTkTabview(app, width=1050, height=700)
tabview.pack(padx=20, pady=20)
tab_live = tabview.add("Live Sentinel Mode")
tab_upload = tabview.add("Forensic Video Analysis")

# --- UI ELEMENTS: LIVE ---
live_video_label = ctk.CTkLabel(tab_live, text="System Disarmed", width=800, height=450, fg_color="black")
live_video_label.pack(pady=10)

live_status_label = ctk.CTkLabel(tab_live, text="Status: Offline", font=("Arial", 16))
live_status_label.pack(pady=5)

live_btn_frame = ctk.CTkFrame(tab_live)
live_btn_frame.pack(pady=10)

# Window Selector Logic
def get_window_titles():
    # Only return windows that actually have titles to avoid junk data
    return [w.title for w in gw.getAllWindows() if w.title.strip() != ""]

def set_window(choice):
    global selected_window_title
    selected_window_title = choice

def refresh_windows():
    titles = get_window_titles()
    window_selector.configure(values=titles)
    if titles:
        window_selector.set(titles[0])
        set_window(titles[0])

window_selector = ctk.CTkOptionMenu(live_btn_frame, values=["Select Target Window..."], command=set_window, width=350)
window_selector.pack(side="left", padx=10)

btn_refresh = ctk.CTkButton(live_btn_frame, text="Refresh Apps", command=refresh_windows, fg_color="gray", hover_color="darkgray")
btn_refresh.pack(side="left", padx=10)

def start_live():
    global live_active
    if not selected_window_title or selected_window_title == "Select Target Window...":
        live_status_label.configure(text="Error: Please select a window to stream.")
        return
        
    if live_active: return
    live_active = True
    live_status_label.configure(text="Connecting to OS Render Buffer...")
    
    threading.Thread(target=camera_reader_thread, daemon=True).start()
    threading.Thread(target=live_processing_thread, daemon=True).start()

def stop_live():
    global live_active, live_pil_image
    live_active = False
    live_pil_image = None
    live_video_label.configure(image="")
    live_status_label.configure(text="System Disarmed")

btn_start_live = ctk.CTkButton(live_btn_frame, text="Arm System", command=start_live, fg_color="green", hover_color="darkgreen")
btn_start_live.pack(side="left", padx=10)
btn_stop_live = ctk.CTkButton(live_btn_frame, text="Disarm System", command=stop_live, fg_color="red", hover_color="darkred")
btn_stop_live.pack(side="left", padx=10)

# --- UI ELEMENTS: UPLOAD ---
upload_video_label = ctk.CTkLabel(tab_upload, text="No Video Loaded", width=800, height=450, fg_color="black")
upload_video_label.pack(pady=10)

upload_status_label = ctk.CTkLabel(tab_upload, text="Ready for Analysis", font=("Arial", 16))
upload_status_label.pack(pady=5)

def select_and_run_video():
    global upload_active
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path: return
    
    upload_active = False 
    time.sleep(0.1)       
    
    upload_active = True
    threading.Thread(target=upload_processing_thread, args=(file_path,), daemon=True).start()

btn_upload = ctk.CTkButton(tab_upload, text="Select Video & Analyze", command=select_and_run_video)
btn_upload.pack(pady=10)

# Populate dropdown on launch
refresh_windows()

# --- THE MASTER UI LOOP ---
def master_ui_loop():
    """This ONLY updates the screen. It never does AI math, so it never lags."""
    if live_active and live_pil_image is not None:
        imgtk = ImageTk.PhotoImage(image=live_pil_image)
        live_video_label.imgtk = imgtk
        live_video_label.configure(image=imgtk, text="")
        live_status_label.configure(text=live_status_text)
        
    if upload_active and upload_pil_image is not None:
        imgtk2 = ImageTk.PhotoImage(image=upload_pil_image)
        upload_video_label.imgtk = imgtk2
        upload_video_label.configure(image=imgtk2, text="")
        upload_status_label.configure(text=upload_status_text)
        
    app.after(33, master_ui_loop)

master_ui_loop()
app.mainloop()