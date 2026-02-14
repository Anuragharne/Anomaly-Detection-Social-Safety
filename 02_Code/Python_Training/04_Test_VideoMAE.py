import os
import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# --- CONFIGURATION ---
MODEL_PATH = r"03_Models\VideoMAE_Model"  # Where your TRAINED model is
ORIGINAL_MODEL = "MCG-NJU/videomae-base" # Original model for the processor (Fixes loading error)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_video_clip(video_path, num_frames=16):
    """Reads a video and extracts 16 frames evenly spaced."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Video has 0 frames.")
        return None
    
    # We need exactly 16 frames for VideoMAE
    # If video is short, we might take duplicates, which is fine.
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # Resize strictly to 224x224 to match model expectation (optional but safer)
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            frames.append(frame)
    
    cap.release()
    
    # Pad if we didn't get enough frames (e.g., extremely short video)
    while len(frames) < num_frames:
        frames.append(frames[-1])
        
    return frames

def run_inference(video_path):
    # Remove quotes if user dragged and dropped file
    video_path = video_path.strip('"').strip("'")
    
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return

    print(f"\n--- Testing Video: {video_path} ---")
    
    # 1. Load Model (From YOUR Local Folder)
    print("Loading model...")
    try:
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Processor (From ONLINE Original Base)
    print("Loading processor...")
    try:
        processor = VideoMAEImageProcessor.from_pretrained(ORIGINAL_MODEL)
    except Exception as e:
        print(f"Error loading processor: {e}")
        return

    # 3. Get Video Frames
    print("Processing video...")
    frames = get_video_clip(video_path)
    if frames is None:
        return

    # 4. Prepare Inputs
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. Predict
    print("Running prediction...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = logits.argmax(-1).item()

    # 6. Show Results
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item()
    
    print("-" * 30)
    print(f"RESULT: {predicted_label.upper()}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    print(f"Running on device: {DEVICE}")
    print("Please enter the path to the video file you want to test.")
    print("(Tip: You can drag and drop the file here)")
    
    video_file = input("Video Path: ")
    run_inference(video_file)