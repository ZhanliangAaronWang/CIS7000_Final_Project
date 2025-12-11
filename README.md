# Adapting SoundStorm for Environmental Sound Generation

**CIS 7000: Advanced Topics in Machine Learning - Final Project**  
*University of Pennsylvania, Fall 2024*

**Team Members:** Aaron Wang, Tripti Tripathi, Kathryn Chen

[![Report](https://img.shields.io/badge/Report-PDF-blue)](link-to-report)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

---

## Overview

This project investigates the adaptation of **SoundStorm**, a state-of-the-art parallel audio generation model originally designed for speech synthesis, to **environmental sound generation**. We conduct comprehensive experiments on ESC-50 to evaluate both generative and discriminative capabilities of the fine-tuned model.

### Key Findings

- **Parallel masked refinement successfully transfers** to non-speech domains, generating temporally coherent audio
- **Systematic failure modes identified**: 67% reduction in zero-crossing rate (high-frequency attenuation), 55% increase in SNR (temporal over-smoothing), 9.5% lower spectral entropy (reduced diversity)
- **Classification performance**: 79.6% accuracy on ESC-50 (vs. 88.5% for task-specific AST baseline)
- **Rigorous evaluation framework**: 8 acoustic metrics, statistical testing, visual analysis

---

## Motivation

While SoundStorm achieves impressive results on speech synthesis with **100× speedup** over autoregressive models, its generalizability to environmental sounds remains unexplored. Environmental sounds present unique challenges:

- **Broader spectral range**: 20 Hz - 20 kHz (vs. speech: 300-3500 Hz)
- **Diverse acoustic phenomena**: Harmonic, impulsive, stochastic, mechanical
- **No linguistic constraints**: Much wider range of valid continuations
- **Rich temporal dynamics**: From sustained drones to millisecond transients

---

## Architecture

### SoundStorm Components

1. **Neural Audio Codec** (Encodec 24 kHz)
   - 8 residual quantizers (RVQ)
   - 1024 codebook size per quantizer
   - Frozen encoder/decoder (pretrained weights)

2. **Bidirectional Conformer Encoder**
   - 2 layers, 512 hidden dimensions
   - 8 attention heads, kernel size 31
   - ~23.5M trainable parameters

3. **Parallel Masked Decoder**
   - MaskGIT-style iterative refinement
   - Cosine unmasking schedule: γ(r) = cos(πr/2)
   - 12 decoding iterations

<p align="center">
  <img src="images/architecture.png" width="70%">
  <br>
  <em>SoundStorm Architecture Overview</em>
</p>

---

## Datasets

### ESC-50
- **2,000** five-second recordings
- **50** environmental sound classes
- **5** high-level categories (animals, natural, human, domestic, urban)
- **5-fold** cross-validation (we use Fold 1: 1,600 train / 400 val)

### UrbanSound8K
- **8,732** recordings of urban sounds
- **10** classes with 10-fold cross-validation
- Used for classification experiments only

---

## Experimental Setup

### Training Configuration

```python
# Audio Generation
batch_size = 8
epochs = 100
optimizer = Adam(lr=1e-4, betas=(0.9, 0.999))
scheduler = OneCycleLR(warmup_ratio=0.3)
loss = CrossEntropy (on masked RVQ tokens)

# Classification
batch_size = 32
epochs = 50
optimizer = Adam(lr=5e-5)
scheduler = CosineAnnealingLR
loss = CrossEntropy
```

### Evaluation Metrics

**Acoustic Metrics (6)**
- Signal-to-Noise Ratio (SNR)
- Zero-Crossing Rate (ZCR)
- RMS Energy
- Spectral Entropy
- Spectral Centroid
- Spectral Rolloff

**Similarity Metrics (2)**
- Spectral Convergence
- Log Spectral Distance

**Statistical Testing**
- Kolmogorov-Smirnov tests (p < 0.05)

---

## Results

### Classification Performance

| Model | ESC-50 | UrbanSound8K |
|-------|--------|--------------|
| AST (baseline) | **88.5%** | **84.2%** |
| SoundStorm (frozen) | 72.3% | 69.8% |
| SoundStorm (fine-tuned) | 79.6% | 75.4% |

**Per-Class Analysis:**
- Strong on temporal structure: clock (92%), dog (87%), keyboard (85%)
- Weak on spectral complexity: helicopter (64%), chainsaw (68%), engine (71%)

### Audio Generation Quality

| Metric | Real Audio | Generated | Difference |
|--------|-----------|-----------|------------|
| SNR (dB) | 14.97 ± 8.10 | 23.18 ± 3.85 | **+54.89%** |
| Zero-Crossing Rate | 0.067 ± 0.075 | 0.022 ± 0.008 | **-66.79%** |
| RMS Energy | 0.105 ± 0.088 | 0.068 ± 0.010 | **-34.92%** |
| Spectral Entropy | 9.85 ± 0.80 | 8.91 ± 0.39 | **-9.49%** |
| Spectral Centroid (Hz) | 3090 ± 1392 | 1392 ± 258 | **-54.95%** |
| Spectral Rolloff (Hz) | 5735 ± 3259 | 3259 ± 729 | **-43.18%** |

**Statistical Significance:**
- All metrics except duration show significant differences (p < 0.05)
- KS statistics > 0.45 for most quality metrics

### Training Dynamics

<p align="center">
  <img src="results/training_curves.png" width="80%">
  <br>
  <em>Training converges smoothly with minimal overfitting (final gap: 0.28)</em>
</p>

---

## Key Findings

### Four Primary Failure Modes

#### 1. **High-Frequency Attenuation** 
- 67% reduction in zero-crossing rate
- 55% reduction in spectral centroid
- **Cause**: Encodec optimized for speech (0-5 kHz), LibriLight training bias

#### 2. **Temporal Over-Smoothing**
- 55% increase in SNR (overly clean signals)
- Missing transients and sharp acoustic features
- **Cause**: Parallel generation averages over multiple continuations

#### 3. **Reduced Spectral Diversity**
- 9.5% lower spectral entropy
- Homogeneous outputs lacking variety
- **Cause**: Limited training data (32 samples/class), conservative generation strategy

#### 4. **Lack of Temporal Evolution**
- Stationary spectrograms vs. dynamic real audio
- Missing long-range dependencies
- **Cause**: Iterative refinement schedule limitations

### Visual Analysis

<p align="center">
  <img src="results/waveform_comparison.png" width="100%">
  <br>
  <em>Real audio (left) shows sharp transients and full spectrum; Generated (right) is smoother with limited high frequencies</em>
</p>

---

## Installation

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.7 (for GPU training)
```

### Setup

```bash
# Clone repository
git clone https://github.com/ZhanliangAaronWang/CIS7000_Final_Project.git
cd CIS7000_Final_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets (ESC-50, UrbanSound8K)
python scripts/download_datasets.py
```

### Required Packages
```
torch>=2.0.0
torchaudio>=2.0.0
encodec>=0.1.1
librosa>=0.10.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## Usage

### Training

```bash
# Fine-tune SoundStorm on ESC-50
python train.py \
  --dataset esc50 \
  --fold 1 \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --output_dir ./checkpoints

# Train classification head
python train_classifier.py \
  --dataset esc50 \
  --checkpoint ./checkpoints/best_model.pt \
  --batch_size 32 \
  --epochs 50
```

### Generation

```bash
# Generate environmental sounds
python generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --num_samples 100 \
  --duration 5.0 \
  --output_dir ./generated_audio

# Generate with specific class conditioning (if implemented)
python generate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --class "dog_bark" \
  --num_samples 10
```

### Evaluation

```bash
# Compute acoustic metrics
python evaluate.py \
  --real_audio ./data/esc50/audio \
  --generated_audio ./generated_audio \
  --metrics all \
  --output_dir ./results

# Run classification evaluation
python evaluate_classifier.py \
  --checkpoint ./checkpoints/classifier.pt \
  --dataset esc50 \
  --fold 1
```

---

## Project Structure

```
CIS7000_Final_Project/
│
├── data/
│   ├── esc50/              # ESC-50 dataset
│   └── urbansound8k/       # UrbanSound8K dataset
│
├── models/
│   ├── soundstorm.py       # SoundStorm implementation
│   ├── conformer.py        # Conformer architecture
│   ├── encodec_wrapper.py  # Encodec codec wrapper
│   └── classifier.py       # Classification head
│
├── scripts/
│   ├── train.py            # Training script
│   ├── generate.py         # Audio generation
│   ├── evaluate.py         # Acoustic metrics evaluation
│   └── download_datasets.py
│
├── utils/
│   ├── audio_metrics.py    # Acoustic metric computation
│   ├── data_loader.py      # Dataset loading utilities
│   └── visualization.py    # Plotting functions
│
├── results/
│   ├── figures/            # Generated plots
│   ├── metrics/            # Evaluation results
│   └── audio_samples/      # Example generations
│
├── checkpoints/            # Saved model weights
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── report.pdf             # Full technical report
```

---
