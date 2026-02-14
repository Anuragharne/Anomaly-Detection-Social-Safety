from transformers import VideoMAEImageProcessor
import os

# Point to your model folder
MODEL_DIR = r"03_Models\VideoMAE_Model"

print(f"Fixing model in: {MODEL_DIR}")

# 1. Download the original processor config from Hugging Face
# (This is identical to what we used for training)
print("Downloading missing processor config...")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# 2. Save it into your local folder
print("Saving 'preprocessor_config.json'...")
processor.save_pretrained(MODEL_DIR)

# 3. Verify
if os.path.exists(os.path.join(MODEL_DIR, "preprocessor_config.json")):
    print("SUCCESS: Model is repaired.")
else:
    print("FAILURE: Could not save file.")