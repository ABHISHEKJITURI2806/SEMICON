# AI-Based Restoration of Degraded Images for Semiconductor Inspection
> **KLA Hackathon Challenge — Solution Repository**

![Semiconductor Restoration Pipeline](https://img.shields.io/badge/Task-Semiconductor%20Image%20Restoration-blue)
![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌  Background & Executive Summary

In semiconductor manufacturing, microscopic inspection images are critical for measuring and verifying chip quality across lithography, etching, and wafer fabrication stages. Microscopic defects at sub-nanometer scales can degrade chip yield and lead to catastrophic semiconductor failure. 

However, optical and scanning inspection tools introduce severe degradation:
1. **Speckle Noise**: Pixel-level granularity caused by laser phase interference, pushing intensity values beyond the valid signal range (`< 0.0` or `> 1.0`).
2. **Gaussian Noise / Edge Softening**: Micro-motion and optical diffraction blur fine structure edges and contact hole boundaries.
3. **Spatial Resolution Loss (Super-Resolution)**: Images downsampled from high-resolution ($256 \times 256$) to low-resolution ($128 \times 128$), destroying fine contact features and line-space patterns.

This repository provides an end-to-end AI solution centered around **`SemiconRestorationNet`** — a deep Gated Residual UNet with PixelShuffle Sub-Pixel Convolution that reverses all three degradations simultaneously in real time.

---


## 🔬 Key Technical Innovations

* **Non-Linear Activation Free (NAF) Architecture**: Replaces standard ReLU/GELU activations with `SimpleGate` elementwise channel gating ($X_1 \odot X_2$), preserving fine semiconductor line-space features without clipping micro-intensity details.
* **PixelShuffle 2x Super-Resolution Head**: Reconstructs high-frequency $256 \times 256$ spatial features from $128 \times 128$ inputs without checkerboard artifacts.
* **Hybrid Composite Loss Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{1} + 0.2 \cdot \mathcal{L}_{\text{SSIM}} + 0.1 \cdot \mathcal{L}_{\text{Sobel\_Edge}}$$
  Directly optimizes structural fidelity (SSIM) and edge crispness (Sobel gradients) while penalizing intensity discrepancies.
* **Ultra-Fast Benchmarked Inference**: Inference takes **$< 3.5\text{ ms}$** per image on GPU ($\sim 35.9\text{ ms}$ on CPU), comfortably meeting KLA's strict real-time throughput requirement.

---

## 📁  Repository Structure

```
├── evaluation_script.py      # Main Standalone Evaluation Script (Directory & Single File Input)
├── train.py                  # Full Training & Validation Pipeline
├── web_app.py                # Interactive Web Visualizer Studio (Single & Batch Upload)
├── requirements.txt          # Environment Dependencies
├── README.md                 # Complete Solution Documentation
├── models/
│   └── restoration_model.py  # SemiconRestorationNet Model Architecture
├── utils/
│   ├── dataset.py            # PyTorch Dataset Loader with Augmentations
│   ├── losses.py             # L1 + SSIM + Sobel Edge Composite Loss
│   └── metrics.py            # PSNR, SSIM, and Loss Computations
├── weights/
│   ├── best_model.pth        # Final Trained Model Checkpoint (1.02 MB)
│   └── semicon_restoration_model.pth
└── restored_test_outputs/   # Restored 256x256 .npy outputs generated on test set (400 files)
```

---

## ⚙️ Quick Start & Setup

### Prerequisites
* Python 3.9+
* PyTorch 2.0+ (CUDA supported or CPU)

### Installation
```bash
git clone https://github.com/YourTeam/KLA-Semicon-Restoration.git
cd KLA-Semicon-Restoration
pip install -r requirements.txt
```

---

## 🚀 Inference Execution (Single Image & Batch Options)

The standalone evaluation script `evaluation_script.py` loads the pre-trained model weights (`weights/best_model.pth`) and supports **both single file restoration and full directory batch processing**.

### Option A: Restore a Full Directory (Benchmarking Batch)
```bash
python evaluation_script.py --input_dir dataset/NoisyLR --output_dir restored_test_outputs
```

### Option B: Restore a Single Degraded Image File
```bash
python evaluation_script.py --input_file dataset/NoisyLR/000000.npy --output_dir single_output_test
```
* Supports `.npy` array files as well as `.png` / `.jpg` standard image files.
* Saves both the `.npy` array output and a high-resolution `.png` visual file.

#### CLI Parameters:
* `--input_dir`: Path to directory containing test `.npy` degraded images (default: `dataset/NoisyLR`).
* `--input_file`: Path to a single degraded file (`.npy`, `.png`, `.jpg`).
* `--output_dir`: Path to directory where restored $256 \times 256$ outputs will be written (default: `restored_test_outputs`).
* `--weights`: Path to model checkpoint file (default: `weights/best_model.pth`).

---

## 🌐  Interactive Studio Web App (Live Visualizer)

To run the interactive web dashboard for real-time single image upload and side-by-side visualization:

```bash
python web_app.py
```
Open **`http://127.0.0.1:8050`** in any web browser to:
1. Select any sample from the dataset OR upload a single custom `.npy` / PNG file.
2. View **Degraded Input**, **AI Restored Clean Output**, and **Ground Truth Reference** side-by-side.
3. Monitor live HUD performance metrics: PSNR, SSIM, Latency (ms), and Intensity Range.

---

## 🏋️  Model Training & Reproduction

To reproduce the model training from scratch:

```bash
python train.py
```

### Training Strategy:
* **Batch Size**: 32
* **Patch Size**: 64x64 random LR crops (mapping to 128x128 GT patches) with horizontal/vertical flips and 90-degree rotations.
* **Optimizer**: AdamW ($\text{lr} = 8 \times 10^{-4}$, weight decay $1 \times 10^{-4}$).
* **Learning Rate Schedule**: Cosine Annealing with $\eta_{\text{min}} = 10^{-6}$.

---

## 📊 Model Performance & Benchmarks

| Metric | Degraded Input (Bicubic) | **SemiconRestorationNet (Ours)** | Improvement Net Gain |
| :--- | :---: | :---: | :---: |
| **Peak Signal-to-Noise Ratio (PSNR)** | 24.53 dB | **31.10 dB** | **+6.57 dB** |
| **Structural Similarity (SSIM)** | 0.6084 | **0.8875** | **+0.2791** |
| **Inference Time (H100 GPU)** | — | **< 3.5 ms / image** | **> 280 FPS** |
| **Inference Time (16-Core CPU)** | — | **~ 35.98 ms / image** | **~ 27.8 FPS** |
| **Model Checkpoint Size** | — | **1.02 MB** | **321,793 params** |

---

