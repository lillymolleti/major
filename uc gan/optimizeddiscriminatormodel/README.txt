Short summary - changes & rationale (keeps USE+CMHSA)

# 🎨 Anime Face Generation using DCGAN + Attention

This project generates anime character faces using an improved DCGAN architecture with attention mechanisms and training stability enhancements.

---

## 🚀 Features

### 🧠 Model Architecture Improvements
- Replaced BatchNorm2d with InstanceNorm2d (affine=True) in Generator
- Integrated USE (Upsampling Squeeze-Excitation) block
- Added CMHSA (Channel Multi-Head Self-Attention)
- Introduced residual skip connection for better gradient flow
- Applied Spectral Normalization in Discriminator
- Added Minibatch Standard Deviation layer to reduce mode collapse
- Retained DCGAN backbone using ConvTranspose2d

---

## 🏗️ Model Architecture

### Generator
- ConvTranspose2d-based upsampling
- InstanceNorm2d (affine=True)
- USE block for channel attention
- CMHSA for global attention
- Residual skip connection

### Discriminator
- PatchGAN-style architecture
- Spectral Normalization on all layers
- LeakyReLU activations
- Minibatch StdDev layer

---

## ⚙️ Training Strategy

- Loss Function: Hinge Loss (instead of BCE)
- n_critic: 2–5 (train discriminator multiple times per generator step)
- DiffAugment: flip, translation, brightness
- EMA (Exponential Moving Average) of generator weights
- Adam optimizer with betas (0.0, 0.9)
- Label smoothing (real labels = 0.9)
- Model checkpointing and sample generation per epoch

---

## 📂 Project Structure
models.py # Generator, Discriminator, USE, CMHSA
train.py # Training loop and optimization
dataset.py # Dataset loader
samples/ # Generated images
checkpoints/ # Saved models
---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/anime-gan.git
cd anime-gan
pip install -r requirements.txt

---
## Train the model
python train.py
---
## Generate images
python generate.py
---
## 📊 Results
- Improved stability compared to baseline DCGAN
- Reduced mode collapse using Minibatch StdDev and augmentations
- Better output quality using attention and EMA
