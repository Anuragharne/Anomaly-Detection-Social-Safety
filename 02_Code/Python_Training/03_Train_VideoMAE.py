import os
import torch
import numpy as np
import evaluate
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, TrainingArguments, Trainer
from decord import VideoReader, cpu
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_ROOT = r"01_Data"
OUTPUT_DIR = r"03_Models\VideoMAE_Model"
MODEL_CKPT = "MCG-NJU/videomae-base"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 6
LEARNING_RATE = 5e-5

print(f"Initializing Gen-2 Training on: {torch.cuda.get_device_name(0)}")

# --- 1. DATASET CLASS ---
class AdvancedVideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor, split="train"):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.split = split

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Temporal Jitter: Train = Random Speed, Val = Normal Speed
            if self.split == 'train':
                stride = random.choice([2, 4, 8])
            else:
                stride = 4 # Standard speed for testing
                
            required_frames = 16 * stride
            
            if total_frames >= required_frames:
                max_start = total_frames - required_frames
                start_idx = random.randint(0, max_start) if self.split == 'train' else 0
                indices = np.arange(start_idx, start_idx + required_frames, stride)
            else:
                indices = np.resize(np.arange(total_frames), 16)
                
            video = vr.get_batch(indices).asnumpy()
            inputs = self.processor(list(video), return_tensors="pt")
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": torch.tensor(label)
            }
        except Exception as e:
            # print(f"Skipping bad file: {video_path}")
            return {
                "pixel_values": torch.zeros((16, 3, 224, 224)),
                "labels": torch.tensor(label)
            }

# --- 2. PREPARE DATA & AUTO-SPLIT ---
print("Scanning dataset folders...")
all_paths = []
all_labels = []
label2id = {'NonFight': 0, 'Fight': 1}

# Load everything from the 'train' folder
train_folder = os.path.join(DATA_ROOT, 'train')
for label_name, label_id in label2id.items():
    folder = os.path.join(train_folder, label_name)
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.avi')]
        all_paths.extend(files)
        all_labels.extend([label_id] * len(files))

# Also check 'val' folder just in case, and add it to the pool
val_folder = os.path.join(DATA_ROOT, 'val')
for label_name, label_id in label2id.items():
    folder = os.path.join(val_folder, label_name)
    if os.path.exists(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.avi')]
        all_paths.extend(files)
        all_labels.extend([label_id] * len(files))

print(f"Total videos found: {len(all_paths)}")

if len(all_paths) == 0:
    print("CRITICAL ERROR: No videos found! Check 01_Data/train")
    exit()

# SPLIT: 80% Train, 20% Validation
print("Auto-splitting dataset (80% Train / 20% Val)...")
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=42
)

# --- 3. LOAD ENGINE ---
processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_CKPT,
    label2id=label2id,
    id2label={0: 'NonFight', 1: 'Fight'},
    ignore_mismatched_sizes=True
)

train_dataset = AdvancedVideoDataset(train_paths, train_labels, processor, split='train')
val_dataset = AdvancedVideoDataset(val_paths, val_labels, processor, split='val')

print(f" > Final Train Count: {len(train_dataset)}")
print(f" > Final Val Count:   {len(val_dataset)}")

# --- 4. METRICS ---
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# --- 5. TRAINER ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True, 
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- 6. START ---
print("\nStarting Gen-2 Fine-Tuning...")
trainer.train()

print(f"SUCCESS. Model saved to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)