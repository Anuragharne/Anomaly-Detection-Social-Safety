import cv2
import torch
import threading
import time
import requests
import numpy as np
import customtkinter as ctk
from google import genai
from PIL import Image
from scipy.spatial.distance import cdist
from collections import deque
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from ultralytics import YOLO
import os

os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
BOT_TOKEN = "" # Your bot token
CHAT_ID = ""   # Your chat id 
GEMINI_API_KEY = "" # Your API Key
client = genai.Client(api_key=GEMINI_API_KEY)

IP_CAMERA_URL = "http://192.168.0.150:8080/video" 
MODEL_PATH = r"03_Models\VideoMAE_Model"

# --- DYNAMIC GLOBAL VARIABLES ---
AI_VIOLENCE_ENABLED = True
AI_CROWD_ENABLED = True

DYNAMIC_VIOLENCE_THRESH = 0.80  
DYNAMIC_CROWD_SIZE = 35       
DYNAMIC_PROXIMITY = 80    

VIOLENCE_COOLDOWN = 30          
CROWD_COOLDOWN = 30        

# ==========================================
# 2. STATE MACHINE & THREAD LOCKS
# ==========================================
gpu_lock = threading.Lock()

live_active = False
live_raw_frame = None
live_pil_image = None
latest_frame_id = 0 # Ensures we only process strictly NEW frames

live_mae_buffer = deque(maxlen=16)
live_evidence_buffer = deque(maxlen=60) # Constant 60-frame rolling past memory

last_crowd_time = 0
is_sending_crowd = False

# --- Strict Violence Pipeline State ---
v_pipeline_state = "MONITORING"
last_violence_time = 0
gather_target_time = 0 # Replaces frame counting with strict time locking
staging_buffer = []

# --- UI Sync Variables ---
ui_v_score = 0.0
ui_c_count = 0

# ==========================================
# 3. GUI THREAD-SAFE LOGGING
# ==========================================
def log_to_gui(msg, alert_level="INFO"):
    app.after(0, _update_textbox, msg, alert_level)

def _update_textbox(msg, alert_level):
    vlm_log_box.configure(state="normal")
    time_str = time.strftime("%H:%M:%S")
    
    if alert_level == "CRITICAL":
        formatted_msg = f"[{time_str}] 🔴 {msg}\n\n"
    elif alert_level == "WARNING":
        formatted_msg = f"[{time_str}] 🟡 {msg}\n\n"
    elif alert_level == "SUCCESS":
        formatted_msg = f"[{time_str}] 🟢 {msg}\n\n"
    else:
        formatted_msg = f"[{time_str}] 🔵 {msg}\n\n"
        
    vlm_log_box.insert("end", formatted_msg)
    vlm_log_box.see("end")
    vlm_log_box.configure(state="disabled")

# ==========================================
# 4. HELPER FUNCTIONS (VLM & DISPATCH)
# ==========================================
def verify_with_gemini(frames_buffer):
    """Extracts 5 frames from the extended buffer."""
    try:
        total_frames = len(frames_buffer)
        if total_frames < 5: 
            return False, "Not enough frames for context."
            
        indices = [int(i * (total_frames - 1) / 4) for i in range(5)]
        pil_images = []
        for idx in indices:
            frame = frames_buffer[idx][0]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_frame))
            
        prompt = "Analyze these 5 sequential security camera frames. Is this a genuine violent physical altercation or a dangerous threat? Answer strictly YES or NO on the first line, followed by a one-sentence justification based on body language and context. Ignore violence playing on TV screens or supervised sports."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt] + pil_images
        )
        text = response.text.strip().upper()
        
        if text.startswith("YES"):
            return True, text
        return False, text
    except Exception as e:
        return False, f"API Error: {e}"

def capture_and_verify_violence(buffer_copy, width, height):
    global v_pipeline_state, last_violence_time
    
    log_to_gui("UPLOADING EVIDENCE PACKET TO CLOUD VLM...", "INFO")
    is_violent, reason = verify_with_gemini(buffer_copy)
    log_to_gui(f"SEMANTIC GATEKEEPER:\n{reason}", "INFO")
    
    if is_violent:
        log_to_gui("THREAT VERIFIED. COMPILING FOOTAGE & DISPATCHING...", "CRITICAL")
        duration = buffer_copy[-1][1] - buffer_copy[0][1]
        actual_fps = len(buffer_copy) / duration if duration > 0 else 10.0
        safe_fps = max(5.0, min(30.0, actual_fps))
        
        filename = f"evidence_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, safe_fps, (width, height))
        for frame, timestamp in buffer_copy:
            out.write(frame)
        out.release()
        
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
            caption = f"🔴 **VIOLENCE VERIFIED BY VLM**\nContext: {reason}"
            with open(filename, 'rb') as video_file:
                requests.post(url, files={'video': video_file}, data={'chat_id': CHAT_ID, 'caption': caption})
            log_to_gui("EVIDENCE SECURELY DISPATCHED.", "SUCCESS")
        except Exception as e:
            log_to_gui(f"TELEGRAM DISPATCH FAILED: {e}", "WARNING")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    else:
        log_to_gui("FALSE ALARM INTERCEPTED & DISMISSED.", "SUCCESS")
                
    last_violence_time = time.time()
    v_pipeline_state = "MONITORING"

def send_telegram_alert(alert_type, extra_info=""):
    global is_sending_crowd
    try:
        log_to_gui(f"CROWD CRUSH PROTOCOL ENGAGED: {extra_info}", "WARNING")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        text = f"🟡 **CROWD CRUSH ALERT**\nDetails: {extra_info}"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': text})
    except Exception:
        pass
    finally:
        is_sending_crowd = False

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
    global live_active, live_raw_frame, latest_frame_id
    url = int(IP_CAMERA_URL) if str(IP_CAMERA_URL).isdigit() else IP_CAMERA_URL
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while live_active:
        ret, frame = cap.read()
        if ret:
            live_raw_frame = frame
            latest_frame_id += 1 # Flag that a strictly new frame has arrived
        else:
            time.sleep(0.05) 
    cap.release()

def live_processing_thread():
    global live_active, live_raw_frame, live_pil_image, latest_frame_id
    global is_sending_crowd, last_crowd_time, ui_c_count
    global v_pipeline_state, gather_target_time, staging_buffer, ui_v_score
    
    processed_frame_id = -1
    
    while live_active:
        # Strict Frame Sync: Only run AI if the camera thread has provided a brand new frame
        if live_raw_frame is None or processed_frame_id == latest_frame_id:
            time.sleep(0.01)
            continue
            
        processed_frame_id = latest_frame_id
        frame = live_raw_frame.copy()
        height, width, _ = frame.shape
        annotated_frame = frame.copy()
        current_time = time.time()
        
        live_evidence_buffer.append((frame.copy(), current_time))
        ui_c_count = 0
        
        # --- YOLO Spatial (Crowd Crush) ---
        if AI_CROWD_ENABLED:
            with gpu_lock:
                results = yolo_model(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes) > 0:
                centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1, y1, x2, y2) in boxes])
                distances = cdist(centers, centers, 'euclidean')
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    neighbors = np.sum(distances[i] < DYNAMIC_PROXIMITY) - 1 
                    if neighbors >= DYNAMIC_CROWD_SIZE:
                        ui_c_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

            if ui_c_count >= DYNAMIC_CROWD_SIZE:
                if not is_sending_crowd and (current_time - last_crowd_time > CROWD_COOLDOWN):
                    is_sending_crowd = True
                    last_crowd_time = current_time
                    threading.Thread(target=send_telegram_alert, args=("CROWD", f"{ui_c_count} clustered.")).start()

        # --- VideoMAE Temporal (Violence) ---
        if AI_VIOLENCE_ENABLED:
            model_input = cv2.resize(frame, (224, 224))
            live_mae_buffer.append(model_input)
            
            if len(live_mae_buffer) == 16:
                inputs = processor(list(live_mae_buffer), return_tensors="pt")
                inputs = {k: v.to(device).half() if v.is_floating_point() else v.to(device) for k, v in inputs.items()}
                
                with gpu_lock:
                    with torch.inference_mode():
                        outputs = videomae_model(**inputs)
                        probs = outputs.logits.softmax(dim=1)
                        ui_v_score = probs[0][1].item() 
                
                # Strict Time-Locked State Machine
                if v_pipeline_state == "MONITORING":
                    if ui_v_score > DYNAMIC_VIOLENCE_THRESH:
                        if (current_time - last_violence_time) > VIOLENCE_COOLDOWN:
                            v_pipeline_state = "GATHERING"
                            # Exact 4-Second Post-Event Lock
                            gather_target_time = current_time + 4.0 
                            staging_buffer = list(live_evidence_buffer) 
                            log_to_gui(f"VIOLENCE TRIPWIRE TRIGGERED ({ui_v_score:.2f}). RECORDING 4 SECONDS OF AFTERMATH...", "WARNING")

        if v_pipeline_state == "GATHERING":
            staging_buffer.append((frame.copy(), current_time))
            if current_time >= gather_target_time:
                v_pipeline_state = "ANALYZING"
                buffer_copy = list(staging_buffer)
                threading.Thread(target=capture_and_verify_violence, args=(buffer_copy, width, height)).start()

        # --- UI Overlay Drawing ---
        cv2.rectangle(annotated_frame, (0, 0), (width, 85), (0, 0, 0), -1)
        
        v_status = "DISABLED" if not AI_VIOLENCE_ENABLED else f"ACTIVE ({ui_v_score:.2f})"
        v_color = (0, 255, 0)
        if v_pipeline_state == "GATHERING":
            v_status = "RECORDING AFTERMATH..."
            v_color = (0, 165, 255)
        elif v_pipeline_state == "ANALYZING":
            v_status = "CLOUD ANALYSIS..."
            v_color = (0, 165, 255)
            
        cv2.putText(annotated_frame, f"VIOLENCE CORE: {v_status}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
                    
        c_status = "DISABLED" if not AI_CROWD_ENABLED else f"{ui_c_count} People"
        c_color = (0, 0, 255) if ui_c_count >= DYNAMIC_CROWD_SIZE else (255, 255, 255)
        cv2.putText(annotated_frame, f"CROWD CORE: {c_status}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)
        
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        live_pil_image = Image.fromarray(img_rgb)

# ==========================================
# 7. NEO-MINIMALIST DASHBOARD GUI
# ==========================================
# Refined Professional Palette
BG_BASE = "#0B0F19"       # Deep Navy/Black
BG_CARD = "#151C2C"       # Elevated Card Base
BG_ELEVATED = "#1A2235"   # Interactive Element Base
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"

ACCENT_MAIN = "#3B82F6"   # Cyber Blue
ACCENT_DANGER = "#EF4444" # Material Red
ACCENT_SAFE = "#10B981"   # Neon Green
ACCENT_WARN = "#F59E0B"   # Alert Orange
BORDER_COLOR = "#2E3A59"

ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("Social Safety Intelligence Node")
app.geometry("1450x850")
app.configure(fg_color=BG_BASE) 

app.grid_columnconfigure(0, weight=3) 
app.grid_columnconfigure(1, weight=1, minsize=420) 
app.grid_rowconfigure(0, weight=1)

# --- LEFT PANEL (VIDEO MONITOR) ---
video_panel = ctk.CTkFrame(app, fg_color="transparent")
video_panel.grid(row=0, column=0, sticky="nsew", padx=(25, 10), pady=25)

monitor_frame = ctk.CTkFrame(video_panel, corner_radius=15, fg_color="#000000", border_width=2, border_color=BORDER_COLOR)
monitor_frame.pack(expand=True, fill="both")

live_video_label = ctk.CTkLabel(monitor_frame, text="NO SIGNAL", text_color=TEXT_SECONDARY, font=("Segoe UI", 28, "bold"))
live_video_label.pack(expand=True)

# --- RIGHT PANEL (COMMAND CONSOLE) ---
control_panel = ctk.CTkFrame(app, corner_radius=15, fg_color=BG_BASE)
control_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 25), pady=25)

# Header
header_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
header_frame.pack(fill="x", padx=10, pady=(10, 20))
ctk.CTkLabel(header_frame, text="CONTROL CENTER", font=("Segoe UI", 24, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w")
sys_status_label = ctk.CTkLabel(header_frame, text="SYSTEM OFFLINE", font=("Segoe UI", 13, "bold"), text_color=TEXT_SECONDARY)
sys_status_label.pack(anchor="w")

# -- VIOLENCE AI CARD --
v_card = ctk.CTkFrame(control_panel, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
v_card.pack(fill="x", pady=(0, 15))

def toggle_violence():
    global AI_VIOLENCE_ENABLED
    AI_VIOLENCE_ENABLED = switch_v.get()
    log_to_gui(f"Module: Temporal Violence {'Online' if AI_VIOLENCE_ENABLED else 'Offline'}", "INFO")

switch_v = ctk.CTkSwitch(v_card, text="Temporal Engine", font=("Segoe UI", 15, "bold"), 
                         text_color=TEXT_PRIMARY, progress_color=ACCENT_MAIN, command=toggle_violence)
switch_v.select()
switch_v.pack(pady=15, padx=20, anchor="w")

def update_v_thresh(value):
    global DYNAMIC_VIOLENCE_THRESH
    DYNAMIC_VIOLENCE_THRESH = float(value)
    lbl_v_val.configure(text=f"{DYNAMIC_VIOLENCE_THRESH:.2f}")

v_slider_frame = ctk.CTkFrame(v_card, fg_color="transparent")
v_slider_frame.pack(fill="x", padx=20, pady=(0, 5))
ctk.CTkLabel(v_slider_frame, text="Activation Threshold", text_color=TEXT_SECONDARY, font=("Segoe UI", 12)).pack(side="left")
lbl_v_val = ctk.CTkLabel(v_slider_frame, text="0.80", text_color=ACCENT_MAIN, font=("Segoe UI", 14, "bold"))
lbl_v_val.pack(side="right")
slider_v = ctk.CTkSlider(v_card, from_=0.50, to=0.99, button_color=ACCENT_MAIN, command=update_v_thresh)
slider_v.set(0.80)
slider_v.pack(fill="x", padx=20, pady=(0, 20))

# -- CROWD AI CARD --
c_card = ctk.CTkFrame(control_panel, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
c_card.pack(fill="x", pady=(0, 15))

def toggle_crowd():
    global AI_CROWD_ENABLED
    AI_CROWD_ENABLED = switch_c.get()
    log_to_gui(f"Module: Spatial Crowd {'Online' if AI_CROWD_ENABLED else 'Offline'}", "INFO")

switch_c = ctk.CTkSwitch(c_card, text="Spatial Engine", font=("Segoe UI", 15, "bold"), 
                         text_color=TEXT_PRIMARY, progress_color=ACCENT_MAIN, command=toggle_crowd)
switch_c.select()
switch_c.pack(pady=15, padx=20, anchor="w")

def update_c_size(value):
    global DYNAMIC_CROWD_SIZE
    DYNAMIC_CROWD_SIZE = int(value)
    lbl_c_val.configure(text=f"{DYNAMIC_CROWD_SIZE}")

c_slider_frame1 = ctk.CTkFrame(c_card, fg_color="transparent")
c_slider_frame1.pack(fill="x", padx=20, pady=(0, 5))
ctk.CTkLabel(c_slider_frame1, text="Density Trigger Limit", text_color=TEXT_SECONDARY, font=("Segoe UI", 12)).pack(side="left")
lbl_c_val = ctk.CTkLabel(c_slider_frame1, text="35", text_color=ACCENT_MAIN, font=("Segoe UI", 14, "bold"))
lbl_c_val.pack(side="right")
slider_c = ctk.CTkSlider(c_card, from_=5, to=100, button_color=ACCENT_MAIN, command=update_c_size)
slider_c.set(35)
slider_c.pack(fill="x", padx=20, pady=(0, 15))

def update_c_prox(value):
    global DYNAMIC_PROXIMITY
    DYNAMIC_PROXIMITY = int(value)
    lbl_p_val.configure(text=f"{DYNAMIC_PROXIMITY}px")

c_slider_frame2 = ctk.CTkFrame(c_card, fg_color="transparent")
c_slider_frame2.pack(fill="x", padx=20, pady=(0, 5))
ctk.CTkLabel(c_slider_frame2, text="Pixel Search Radius", text_color=TEXT_SECONDARY, font=("Segoe UI", 12)).pack(side="left")
lbl_p_val = ctk.CTkLabel(c_slider_frame2, text="80px", text_color=ACCENT_MAIN, font=("Segoe UI", 14, "bold"))
lbl_p_val.pack(side="right")
slider_p = ctk.CTkSlider(c_card, from_=20, to=200, button_color=ACCENT_MAIN, command=update_c_prox)
slider_p.set(80)
slider_p.pack(fill="x", padx=20, pady=(0, 20))

# -- VLM NEURAL LOG CARD --
log_card = ctk.CTkFrame(control_panel, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
log_card.pack(fill="both", expand=True, pady=(0, 15))

ctk.CTkLabel(log_card, text="SYSTEM LOG", font=("Segoe UI", 12, "bold"), text_color=TEXT_SECONDARY).pack(anchor="w", padx=15, pady=(10, 0))
vlm_log_box = ctk.CTkTextbox(log_card, fg_color=BG_ELEVATED, text_color=TEXT_PRIMARY, font=("Consolas", 13), wrap="word", corner_radius=8)
vlm_log_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))
vlm_log_box.insert("end", "System Initialized. Ready for deployment.\n\n")
vlm_log_box.configure(state="disabled")

# -- ACTION BUTTONS --
def start_live():
    global live_active
    if live_active: return
    live_active = True
    monitor_frame.configure(border_color=ACCENT_MAIN) 
    sys_status_label.configure(text="STATUS: SECURE MONITORING", text_color=ACCENT_SAFE)
    log_to_gui("CAMERA UPLINK ESTABLISHED. AI ARMED.", "SUCCESS")
    threading.Thread(target=camera_reader_thread, daemon=True).start()
    threading.Thread(target=live_processing_thread, daemon=True).start()

def stop_live():
    global live_active, live_pil_image, v_pipeline_state
    live_active = False
    live_pil_image = None
    v_pipeline_state = "MONITORING"
    live_video_label.configure(image=None, text="NO SIGNAL")
    sys_status_label.configure(text="STATUS: OFFLINE", text_color=TEXT_SECONDARY)
    monitor_frame.configure(border_color=BORDER_COLOR) 
    log_to_gui("AI DISARMED. THREADS TERMINATED.", "WARNING")

btn_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
btn_frame.pack(fill="x")

ctk.CTkButton(btn_frame, text="ARM SYSTEM", font=("Segoe UI", 15, "bold"), 
              fg_color=ACCENT_SAFE, hover_color="#059669", height=45,
              command=start_live).pack(side="left", expand=True, padx=(0, 5))

ctk.CTkButton(btn_frame, text="DISARM", font=("Segoe UI", 15, "bold"), 
              fg_color=ACCENT_DANGER, hover_color="#DC2626", height=45,
              command=stop_live).pack(side="right", expand=True, padx=(5, 0))

# ==========================================
# 8. MASTER UI LOOP
# ==========================================
def master_ui_loop():
    if live_active and live_pil_image is not None:
        panel_w = video_panel.winfo_width() - 40
        panel_h = video_panel.winfo_height() - 40
        if panel_w > 100 and panel_h > 100:
            img_w, img_h = live_pil_image.size
            ratio = min(panel_w/img_w, panel_h/img_h)
            new_size = (int(img_w * ratio), int(img_h * ratio))
            
            imgtk = ctk.CTkImage(light_image=live_pil_image, size=new_size)
            live_video_label.imgtk = imgtk
            live_video_label.configure(image=imgtk, text="")
            
        if v_pipeline_state == "GATHERING":
            sys_status_label.configure(text="STATUS: RECORDING INCIDENT...", text_color=ACCENT_WARN)
        elif v_pipeline_state == "ANALYZING":
            sys_status_label.configure(text="STATUS: VLM VERIFICATION...", text_color=ACCENT_WARN)
        else:
            sys_status_label.configure(text="STATUS: SECURE MONITORING", text_color=ACCENT_SAFE)
        
    app.after(33, master_ui_loop)

master_ui_loop()
app.mainloop()