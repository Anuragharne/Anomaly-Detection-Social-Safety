import cv2
import numpy as np
import torch
import collections
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import time

# --- CONFIGURATION ---
# Point this to the folder where your training finished
MODEL_PATH = r"03_Models\VideoMAE_Model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CRITICAL: The Buffer Size
# We need 64 frames to capture ~2 seconds of action.
# We will sample 16 frames from this buffer (Stride 4).
BUFFER_SIZE = 64 
SAMPLE_STRIDE = 4 

print(f"Initializing VideoMAE System on: {DEVICE.upper()}")

# --- 1. LOAD THE TRAINED BRAIN ---
print("Loading Model & Processor...")
try:
    # Load the EXACT processor used during training
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
    # Load the EXACT model weights
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print(" > Success! Model loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model from {MODEL_PATH}")
    print(f"Error details: {e}")
    print("Did the previous training script finish and save the model?")
    exit()

# --- 2. REAL-TIME LOOP ---
def run_system():
    cap = cv2.VideoCapture(0) # Webcam
    
    # The Rolling Buffer: Stores the last 64 raw frames
    frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
    
    print("-" * 50)
    print(f"System READY. Waiting for buffer to fill ({BUFFER_SIZE} frames)...")
    print("-" * 50)
    
    # Metrics
    fps_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. PRE-PROCESSING (Speed Optimization)
        # Resize frame early to save RAM/CPU (VideoMAE expects 224x224)
        # We resize to slightly larger to allow for "RandomCrop" logic validation
        resized_frame = cv2.resize(frame, (256, 256))
        
        # Convert BGR (OpenCV) to RGB (Transformer expects RGB)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Add to buffer
        frame_buffer.append(rgb_frame)
        
        # 2. INFERENCE (Only when buffer is full)
        status_text = "Buffering..."
        color = (0, 255, 255) # Yellow
        score = 0.0
        label = "Wait"
        
        if len(frame_buffer) == BUFFER_SIZE:
            # A. STRIDED SAMPLING (The Time Dilation Fix)
            # Take indices [0, 4, 8, ..., 60] -> 16 frames
            indices = list(range(0, BUFFER_SIZE, SAMPLE_STRIDE))
            
            # Select frames
            sampled_video = [frame_buffer[i] for i in indices] # List of 16 arrays
            
            # B. PROCESSOR NORMALIZATION
            # This applies the Mean/Std subtraction the model learned
            inputs = processor(list(sampled_video), return_tensors="pt")
            
            # Move to GPU
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # C. PREDICT
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get the score for "Fight" (Index 1)
                fight_score = probabilities[0][1].item()
                nonfight_score = probabilities[0][0].item()
            
            # D. LOGIC
            if fight_score > 0.60: # Threshold
                label = "VIOLENCE"
                score = fight_score
                color = (0, 0, 255) # Red
            else:
                label = "NORMAL"
                score = nonfight_score
                color = (0, 255, 0) # Green

        # 3. VISUALIZATION
        # Draw UI on the ORIGINAL frame (high res)
        # Bar chart background
        cv2.rectangle(frame, (0, 0), (640, 60), (0,0,0), -1)
        
        # Text
        cv2.putText(frame, f"STATUS: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Conf: {score:.2f}", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS Counter
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            # print(f"FPS: {frame_count}") # Debug
            frame_count = 0
            fps_time = time.time()

        cv2.imshow('VideoMAE Security System', frame)
        
        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()