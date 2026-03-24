# Social Safety Intelligence Command Center

## Overview
An enterprise-grade, dual-AI surveillance pipeline designed to transition standard CCTV and IP camera networks from passive recording to active, context-aware threat detection. 

Built by Anurag Harne and Anubhav Dubey, this system solves the "Semantic Blindspot" of traditional computer vision. Instead of relying purely on pixel velocity or rigid bounding-box rules, it utilizes a **Cascade Architecture**: a lightweight local AI tripwire triggers a cloud-based Vision-Language Model (VLM) to semantically verify human intent before dispatching alerts.

## 🚀 Key Features
* **Semantic Gatekeeper (VLM Cascade):** Uses **Gemini 2.5 Flash** to analyze 5-frame sequences of anomalous events, filtering out false positives (e.g., sparring in a gym, actors on a screen) by understanding scene context and body language.
* **Temporal Violence Detection:** Employs **VideoMAE** (Masked Autoencoder) fine-tuned on custom datasets to detect the kinetic signatures of physical altercations in real-time.
* **Spatial Crowd Analytics:** Uses **YOLOv8** combined with SciPy Euclidean distance clustering to detect high-density crowd formations and prevent crowd crush events.
* **Zero-Latency UI:** CustomTkinter dashboard operating on decoupled GPU threads, ensuring 30 FPS video playback without blocking inference processes.
* **Automated Dispatch:** Maintains a 60-frame rolling memory buffer. Upon verified threat detection, it compiles an `.mp4` and dispatches it instantly via the Telegram API to security personnel.

## 🛠️ Tech Stack
* **AI & Vision:** PyTorch, Ultralytics (YOLOv8), Hugging Face Transformers (`videomae-base`), OpenCV, SciPy
* **Cloud API:** Google GenAI SDK (`gemini-2.5-flash`)
* **UI & Concurrency:** CustomTkinter, Python `threading`
* **Hardware:** Optimized for NVIDIA RTX 4050 (FP16 half-precision, GPU locking)

## 📂 Project Structure
* `02_Code/Python_Training/`
  * `05_RealTime_Wireless.py` - IP Camera integration module.
  * `06_Native_Dashboard.py` - Offline-first, hardware-accelerated command center.
  * `07_VLM_Support.py` - **[LATEST]** Master pipeline featuring the Gemini 2.5 Semantic Gatekeeper and Telegram dispatch.
* `03_Models/` - Local directory for VideoMAE Hugging Face weights.

## ⚙️ Setup & Execution
1. Install dependencies: `pip install torch torchvision torchaudio opencv-python customtkinter scipy ultralytics transformers google-genai`
2. Insert your specific API credentials in `07_VLM_Support.py`:
   * `BOT_TOKEN` (Telegram)
   * `CHAT_ID` (Telegram)
   * `GEMINI_API_KEY` (Google AI Studio)
3. Run the Command Center:
   ```bash
   python 02_Code/Python_Training/07_VLM_Support.py