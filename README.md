
# 🚢 Deep Learning Ship Classification

A comprehensive deep learning project for classifying ship images into 5 categories using multiple approaches: **Supervised Learning**, **Unsupervised Learning**, and **State-of-the-Art Transfer Learning**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Models](#models)
  - [Supervised Learning](#1-supervised-learning)
  - [Unsupervised Learning](#2-unsupervised-learning)
  - [State-of-the-Art (SOTA)](#3-state-of-the-art-sota)
- [How to Run](#how-to-run)
- [Results](#results)
- [Regularization & Optimization Techniques](#regularization--optimization-techniques)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project explores different deep learning paradigms for image classification:

| Approach | Description | Models |
|----------|-------------|--------|
| **Supervised** | Traditional labeled training | CNN Baseline, Deep MLP |
| **Unsupervised** | Feature learning without labels | Autoencoder, DCGAN |
| **SOTA** | Pre-trained model fine-tuning | EfficientNetB0, ResNet50 |

The goal is to classify satellite/aerial images of ships into **5 categories**:
1. **Cargo** (Category 1)
2. **Military** (Category 2)
3. **Carrier** (Category 3)
4. **Cruise** (Category 4)
5. **Tanker** (Category 5)

---

## 📊 Dataset

The dataset contains **6,252 ship images** with corresponding labels.

```
data/
├── train/
│   ├── train.csv          # Image filenames and category labels
│   └── images/            # 6,252 ship images (.jpg)
├── test_ApKoW4T.csv       # Test set metadata
└── sample_submission_ns2btKE.csv
```

### CSV Format
```csv
image,category
2823080.jpg,1
2870024.jpg,1
2662125.jpg,2
...
```

---

## 📁 Project Structure

```
Deep-Learning---Ship-Dataset/
├── data/
│   └── train/
│       ├── train.csv
│       └── images/
├── src/
│   ├── Supervised/
│   │   ├── ship_cnn_baseline.py      # Simple CNN classifier
│   │   └── ship_supervised_mlp.py    # Deep MLP classifier
│   ├── Unsupervised/
│   │   ├── ship_unsupervised_ENCODER.py              # Convolutional Autoencoder
│   │   ├── ship_unsupervised_dcgan.py                # DCGAN for data augmentation
│   │   └── ship_unsupervised_transfer_classification.py  # Transfer from autoencoder
│   └── SOTA/
│       ├── ship_efficientnet.py      # EfficientNetB0 transfer learning
│       └── ship_sota_resnet50.py     # ResNet50 transfer learning
├── results/                          # Saved models and plots
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 - 3.11 (recommended for TensorFlow compatibility)
- pip or conda

### Option 1: Using Conda (Recommended for macOS)

```bash
# Create environment with Python 3.11
conda create -n ship_dl python=3.11 -y
conda activate ship_dl

# Install dependencies
pip install tensorflow pandas matplotlib scikit-learn seaborn

# For macOS GPU acceleration (Apple Silicon)
pip install tensorflow-metal
```

### Option 2: Using venv

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'Devices: {tf.config.list_physical_devices()}')"
```

Expected output (with GPU):
```
TensorFlow: 2.18.0
Devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 🧠 Models

### 1. Supervised Learning

#### CNN Baseline (`ship_cnn_baseline.py`)
A simple 3-layer Convolutional Neural Network.

**Architecture:**
```
Input (128x128x3)
    → Conv2D(32) → MaxPool
    → Conv2D(64) → MaxPool
    → Conv2D(128) → MaxPool
    → Flatten → Dense(256) → Dropout(0.5)
    → Output (5 classes)
```

**Run:**
```bash
python src/Supervised/ship_cnn_baseline.py
```

**Output:**
- `results/baseline_cnn.h5` - Trained model
- `results/training_curves.png` - Accuracy/Loss plots

---

#### Deep MLP (`ship_supervised_mlp.py`)
A Multi-Layer Perceptron with regularization.

**Architecture:**
```
Input (64x64x3) → Flatten
    → Dense(2048) + BatchNorm + Dropout(0.4)
    → Dense(1024) + BatchNorm + Dropout(0.4)
    → Dense(512) + BatchNorm + Dropout(0.3)
    → Dense(256) + BatchNorm + Dropout(0.3)
    → Output (5 classes)
```

**Run:**
```bash
python src/Supervised/ship_supervised_mlp.py
```

---

### 2. Unsupervised Learning

#### Convolutional Autoencoder (`ship_unsupervised_ENCODER.py`)
Learns compressed representations of ship images without labels.

**Architecture:**
```
Encoder: Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool (16x16x128)
Decoder: Conv2D → UpSample → Conv2D → UpSample → Conv2D → UpSample → Output
```

**Run:**
```bash
python src/Unsupervised/ship_unsupervised_ENCODER.py
```

**Output:**
- `results/ship_autoencoder_full.keras` - Full autoencoder
- `results/ship_encoder_only.keras` - Encoder for transfer learning
- `results/ae_reconstruction_sample.png` - Reconstruction examples

---

#### Transfer Classification (`ship_unsupervised_transfer_classification.py`)
Uses the pre-trained encoder from the autoencoder as a feature extractor.

**Prerequisites:** Run `ship_unsupervised_ENCODER.py` first!

**Run:**
```bash
python src/Unsupervised/ship_unsupervised_transfer_classification.py
```

---

#### DCGAN (`ship_unsupervised_dcgan.py`)
Generates synthetic ship images for data augmentation.

**Architecture:**
- **Generator:** Dense → Reshape → Conv2DTranspose (×4) → Output (64×64×3)
- **Discriminator:** Conv2D (×2) → Flatten → Dense(1)

**Run:**
```bash
python src/Unsupervised/ship_unsupervised_dcgan.py
```

**Output:**
- `results/dcgan/generator.h5` - Trained generator
- `results/dcgan/generated/` - Synthetic images organized by class
- `results/dcgan/sample_epoch_*.png` - Training progress samples

---

### 3. State-of-the-Art (SOTA)

#### EfficientNetB0 (`ship_efficientnet.py`)
Transfer learning with EfficientNetB0 pre-trained on ImageNet.

**Architecture:**
```
EfficientNetB0 (frozen) → GlobalAveragePooling2D → BatchNorm → Dropout(0.2) → Dense(5)
```

**Run:**
```bash
python src/SOTA/ship_efficientnet.py
```

**Output:**
- `results/ship_efficientnet.keras`
- `results/efficientnet_curves.png`

---

#### ResNet50 (`ship_sota_resnet50.py`)
Transfer learning with ResNet50 pre-trained on ImageNet.

**Architecture:**
```
ResNet50 (frozen) → GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(5)
```

**Run:**
```bash
python src/SOTA/ship_sota_resnet50.py
```

**Output:**
- `results/ship_resnet50.keras`
- `results/resnet50_curves.png`

---

## 🚀 How to Run

### Quick Start (Run All Models)

```bash
# Activate environment
conda activate ship_dl  # or: source venv/bin/activate

# Navigate to project root
cd Deep-Learning---Ship-Dataset

# 1. Run Supervised Models
python src/Supervised/ship_cnn_baseline.py
python src/Supervised/ship_supervised_mlp.py

# 2. Run Unsupervised Models (in order!)
python src/Unsupervised/ship_unsupervised_ENCODER.py
python src/Unsupervised/ship_unsupervised_transfer_classification.py
python src/Unsupervised/ship_unsupervised_dcgan.py

# 3. Run SOTA Models
python src/SOTA/ship_efficientnet.py
python src/SOTA/ship_sota_resnet50.py
```

### Running Individual Scripts

All scripts are designed to be run from the **project root directory**:

```bash
# From project root
python src/Supervised/ship_cnn_baseline.py
```

Each script will:
1. Automatically detect paths relative to project root
2. Create necessary output directories in `results/`
3. Save trained models and visualizations

---

## 📈 Results

After training, results are saved in the `results/` directory:

```
results/
├── baseline_cnn.h5                    # CNN model
├── training_curves.png                # CNN training plots
├── ship_efficientnet.keras            # EfficientNet model
├── efficientnet_curves.png
├── ship_resnet50.keras                # ResNet50 model
├── resnet50_curves.png
├── ship_autoencoder_full.keras        # Autoencoder
├── ship_encoder_only.keras            # Encoder only
├── ae_reconstruction_sample.png
├── autoencoder/                       # Autoencoder artifacts
│   ├── train_feats.npy
│   └── confusion_matrix.png
└── dcgan/                             # DCGAN artifacts
    ├── generator.h5
    ├── discriminator.h5
    └── generated/                     # Synthetic images
```

---

## 🔧 Regularization & Optimization Techniques

### Regularization

| Technique | Scripts | Purpose |
|-----------|---------|---------|
| **Dropout** (0.2-0.5) | All | Prevents overfitting |
| **Batch Normalization** | MLP, DCGAN, EfficientNet | Stabilizes training |
| **L2 Regularization** | MLP | Weight decay |
| **Weight Freezing** | Transfer Learning models | Preserve pre-trained features |
| **Input Normalization** | All | [0,1] or [-1,1] scaling |

### Optimization

| Technique | Scripts | Purpose |
|-----------|---------|---------|
| **Adam Optimizer** | All | Adaptive learning rate |
| **Data Caching** | CNN Baseline | Faster epoch iterations |
| **Prefetching** | CNN, DCGAN | Parallel data loading |
| **LeakyReLU** | DCGAN | Prevents dying neurons |
| **GlobalAveragePooling2D** | SOTA models | Reduces parameters |

---

## 🐛 Troubleshooting

### Slow Training (CPU Only)

**Problem:** TensorFlow only detecting CPU.

**Solution (macOS Apple Silicon):**
```bash
conda create -n ship_dl python=3.11 -y
conda activate ship_dl
pip install tensorflow tensorflow-metal
```

**Verify GPU:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### "Filling up shuffle buffer" Takes Forever

**Problem:** Dataset shuffling is slow.

**Solution:** The code already shuffles file paths before loading images. If still slow, reduce buffer size:
```python
ds = ds.shuffle(buffer_size=1000, seed=SEED)  # Instead of len(paths)
```

---

### Import Error: `tensorflow.keras`

**Problem:** Pylance shows import error.

**Solution:** This is a linter issue, not a runtime error. The code uses:
```python
from tensorflow import keras
layers = keras.layers  # Works correctly
```

---

### Out of Memory

**Problem:** Training crashes due to RAM/VRAM limits.

**Solution:**
1. Reduce `BATCH_SIZE` (e.g., 16 instead of 32)
2. Reduce `IMG_SIZE` (e.g., 64 instead of 128)
3. Remove `.cache()` from data pipeline

---

### FileNotFoundError for Images

**Problem:** Script can't find images.

**Solution:** Ensure you're running from project root:
```bash
cd Deep-Learning---Ship-Dataset
python src/Supervised/ship_cnn_baseline.py
```

---

## 📄 License

This project is for educational purposes (COMP 263 - Deep Learning, Fall 2025).

---

## 👤 Author

**Reet**

---

## 🙏 Acknowledgments

- Dataset source: [Analytics Vidhya Game of Deep Learning Competition](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/)
- TensorFlow and Keras teams
- Pre-trained models from ImageNet
