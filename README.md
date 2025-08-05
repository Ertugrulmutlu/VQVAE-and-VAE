# VQVAE-and-VAE

This repository contains a modular and reproducible pipeline to train, inspect, and evaluate a **Variational Autoencoder (VAE)** for compressing passport-style facial images.

> 💡 This is part of a broader project comparing different image compression techniques (see blog post for VQ-VAE comparison).
---

## 🧠 VAE Training Pipeline

---

## 📦 Installation

Before running any code, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧭 Project Overview

```
VAE_Project/
├── main.py                 # Main entry point to train the model
├── test_main.py            # Evaluation and visual comparison on test image
├── Utils/
│   ├── model.py            # VAE architecture
│   ├── train.py            # Training loop
│   ├── data_preprocessor.py # Data loading and augmentation
│   ├── data_inspector.py   # Dataset quality check (e.g. corrupted images)
├── Datas/
│   ├── Subclass            # Train data in here
│   ├── test                # Test data in here
├── vae_model.pt            # Trained with 20 epochs
├── vae_model_50.pt         # Trained with 50 epochs (recommended)
├── requirements.txt
```

> 📂 **Model Checkpoints**:
>
> * `vae_model.pt`: trained with **20 epochs**
> * `vae_model_50.pt`: trained with **50 epochs** (recommended for better reconstruction quality)

---

## 🚀 How to Train

### 1. Prepare Your Dataset

Place your images inside a subfolder under `Datas/`, e.g.,

```
Datas/
├── train/
    └── your_images.jpg/png
```

Each subfolder is treated as a separate class by PyTorch's `ImageFolder`, even though labels are not used.

### 2. Run the Training Pipeline

```bash
python main.py
```

This will:

* Inspect the dataset for corrupted files (optional removal)
* Preprocess all images (resize, convert to RGB, normalize)
* Train a VAE with latent dimension 128
* Save the trained model as `vae_model_50.pt`

---

## 🔍 Evaluate & Visualize

After training, you can evaluate and visualize the compression results:

```bash
python test_main.py
```

This script will:

* Load a test image (`Datas/test/test1.png`)
* Compress and reconstruct it using the trained model
* Save the output as JPEG (`Datas/test/reconstructed_50.jpg`)
* Print the compression ratio
* Display original vs reconstructed side-by-side

---

## 🧱 Code Modules

### `main.py`

Sets up the full training pipeline:

* Loads and preprocesses data
* Initializes VAE
* Trains using `Trainer`

### `test_main.py`

Standalone evaluation and visualization script to test compression results.

### `Utils/model.py`

Contains a clean PyTorch implementation of the VAE model:

* Encoder: 64x64x3 → latent vector
* Decoder: latent vector → 64x64x3 image

### `Utils/train.py`

Modular `Trainer` class for VAE:

* Optimizes VAE using combined MSE + KL Divergence loss
* Supports GPU/CPU

### `Utils/data_preprocessor.py`

Builds the PyTorch dataloader with:

* RGB conversion
* Resizing to 64x64
* Augmentation-ready

### `Utils/data_inspector.py`

Optional tool to check for corrupted images and summarize dataset stats.

---

## 📝 Notes

* Default image resolution: **64x64 RGB**
* Latent space size: **128 dimensions**
* Output reconstructions are saved in **JPEG** with quality 85

---

## 🧠 VQVAE Training Pipeline

## 📦 Installation

Before running any code, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧭 Project Overview

```
CompressionProject/
├── main.py                       # Entry point (VAE or VQ-VAE)
├── test_main.py                 # Evaluation script (128x128)
├── test_main64x64.py            # Evaluation script (64x64)
├── Utils/
│   ├── model.py                 # VAE and VQ-VAE architectures
│   ├── train.py                 # VAETrainer & VQTrainer
│   ├── data_preprocessor.py     # Data loader (128x128)
│   ├── data_preprocessor 64x64.py # Data loader (64x64)
│   ├── data_inspector.py        # Corruption check and stats
├── Datas/
│   ├── Subclass            # Train data in here
│   ├── test                # Test data in here
├── vae_model.pt                 # VAE (20 epochs)
├── vae_model_50.pt              # VAE (50 epochs)
├── vqvae_model.pt               # VQ-VAE 64x64 (20 epochs)
├── vqvae_model_128x128.pt       # VQ-VAE 128x128 (50 epochs)

```

## 📁 Model Checkpoints

* `vqvae_model.pt`: 64x64 VQ-VAE, trained 20 epochs
* `vqvae_model_128x128.pt`: 128x128 VQ-VAE, trained 50 epochs (recommended)

---

### 1. Prepare Your Dataset

Place your facial images in a subfolder inside `Datas/`, for example:

```
Datas/
└── train/
    ├── image1.jpg
    └── image2.png
```

> The folder name doesn't matter. PyTorch treats each subfolder as a class (unused here).

---

### 2. Train a VQ-VAE Model

You can choose between **64x64** and **128x128** training resolutions:

#### 🟢 64x64 Version:

Make sure to use `data_preprocessor 64x64.py`:

```bash
python main.py  # saves model to vqvae_model.pt
```

#### 🔵 128x128 Version:

Uses default `data_preprocessor.py`:

```bash
python main.py  # saves model to vqvae_model_128x128.pt
```

Training outputs include:

* Reconstruction + vector quantization loss
* Progress bar with loss metrics
* Saved model weights in `.pt` format

---

## 🔍 Evaluate & Visualize

### Test script (128x128):

```bash
python test_main.py
```

### Test script (64x64):

```bash
python test_main64x64.py
```

Each script:

* Loads trained VQ-VAE
* Reconstructs a test image from `Datas/test/test1.png`
* Saves result to `Datas/test/reconstructed_vqvae.jpg`
* Displays compression rate and original vs reconstructed side-by-side

---

## 🧱 Code Modules

### `main.py`

Loads `DataPreprocessor`, `VQVAE`, and `VQTrainer` to run full training loop.

### `test_main*.py`

Evaluates model on test image, saves compressed result, prints stats.

### `Utils/model.py`

Defines:

* `VQVAE`: encoder → quantizer → decoder
* `VectorQuantizer`: nearest embedding codebook lookup

### `Utils/train.py`

* `VQTrainer`: training logic using MSE + codebook commitment loss

### `Utils/data_preprocessor*.py`

Image loaders for different resolutions. Includes:

* RGB conversion
* Resize pipeline

### `Utils/data_inspector.py`

Checks for corrupted or unreadable images and prints dataset info.

---

## 📝 Notes

* Input resolution must match training config (64x64 or 128x128)
* Output images saved as JPEG (quality=85)
* Embedding vector size, codebook entries, and commitment cost can be tuned

---

## ⭐️ Support

If you find this useful, feel free to ⭐ the repo and check out the full project series!

---

MIT License | Made with ❤️ by Ertuğrul Mutlu
