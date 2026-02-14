# Social Safety System: Real-Time Violence Detection (VideoMAE)

## Overview
An industrial-grade AI system capable of detecting violence in real-time video streams. 
Unlike traditional "Skeleton-based" approaches, this project utilizes **Vision Transformers (VideoMAE)** to analyze raw pixel data, allowing it to detect subtle aggression, close-quarters combat, and crowd violence.

## Key Features
* **Architecture:** VideoMAE (Masked Autoencoder) fine-tuned on RWF-2000 & HMDB51.
* **Temporal Awareness:** Uses strided sampling to detect both fast punches and slow grappling.
* **Hard Negative Mining:** Specifically trained to distinguish "Hugging" and "Dancing" from actual violence.
* **Real-Time Engine:** Custom inference pipeline with a 64-frame rolling buffer for live webcam analysis.

## Tech Stack
* **Core:** Python 3.10, PyTorch
* **Model:** Hugging Face Transformers (`videomae-base`)
* **Vision:** OpenCV, Decord
* **Hardware:** Optimized for NVIDIA RTX 4050 (Gradient Accumulation enabled)

## Project Structure
* `02_Code/Python_Training/` - Training and Inference scripts.
* `03_Train_VideoMAE.py` - The "Gen-3" training engine with temporal jitter.
* `05_RealTime_VideoMAE.py` - The live demo application.

## Accuracy
* **Validation Accuracy:** 89.06% (Gen-2 Benchmark)
* **Robustness:** Tested against camera shake (RLVS) and occlusion.