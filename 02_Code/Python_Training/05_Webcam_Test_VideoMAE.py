import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from collections import deque
import time

# --- CONFIGURATION ---
MODEL_PATH = r"03_Models\VideoMAE_Model"    # Your trained model
ORIGINAL_MODEL = "MCG-NJU/videomae-base"    # Fix for missing processor config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN = 16  # VideoMAE needs 16 frames to make a decision
RESIZE_TO = (224, 224) # Standard size for VideoMAE

def main():
    print(f"--- Initializing VideoMAE on {DEVICE} ---")
    
    # 1. Load Model
    print("Loading model... (this takes a few seconds)")
    try:
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
        processor = VideoMAEImageProcessor.from_pretrained(ORIGINAL_MODEL)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Running! Press 'q' to quit ---")

    # Buffer to store the last 16 frames
    frame_buffer = deque(maxlen=CLIP_LEN)
    
    # Variables for prediction text
    current_label = "Buffering..."
    current_conf = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for the model (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)

        # Only predict every 4 frames to keep the video smooth(ish)
        # And only if we have enough frames (16)
        frame_count += 1
        if len(frame_buffer) == CLIP_LEN and frame_count % 4 == 0:
            
            # Prepare inputs
            # The model expects a list of numpy arrays
            inputs = processor(list(frame_buffer), return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Run Prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(-1).item()
            
            # Get Label
            current_label = model.config.id2label[pred_idx]
            current_conf = probs[0][pred_idx].item()
            
            # Print to console (optional)
            # print(f"Detected: {current_label} ({current_conf:.2f})")

        # --- DRAW RESULTS ON SCREEN ---
        # Color: Red for Fight/Harassment, Green for Normal
        if current_label.lower() in ["fight", "harassment"]:
            color = (0, 0, 255) # Red
        elif current_label.lower() == "normal":
            color = (0, 255, 0) # Green
        else:
            color = (255, 255, 0) # Yellow (Buffering)

        # Draw box and text
        cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1) # Black background box
        cv2.putText(frame, f"ACTION: {current_label.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"CONF: {current_conf:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Social Safety - VideoMAE Live Test", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()