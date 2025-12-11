# Adapting SoundStorm for Environmental Sound Generation

**CIS 7000: Advanced Topics in Machine Learning - Final Project**  
University of Pennsylvania, Fall 2024

**Team:** Aaron Wang, Tripti Tripathi, Kathryn Chen

---

## Overview

This project investigates adapting SoundStorm, a parallel audio generation model originally designed for speech, to environmental sound generation. We evaluate both generative and discriminative capabilities on ESC-50.

### Key Results

- Parallel masked refinement successfully transfers to environmental sounds
- Generated audio shows systematic issues: 67% reduction in zero-crossing rate, 55% increase in SNR (over-smoothing), 55% reduction in spectral centroid
- Classification accuracy: 79.6% on ESC-50 (vs. 88.5% for AST baseline)
- Comprehensive evaluation with 8 acoustic metrics and statistical testing

---

## Motivation

SoundStorm achieves 100x speedup over autoregressive models on speech. Environmental sounds present different challenges:

- Broader spectral range (20 Hz - 20 kHz vs. 300-3500 Hz for speech)
- Diverse acoustic phenomena: harmonic, impulsive, stochastic, mechanical
- No linguistic constraints
- Rich temporal dynamics from sustained drones to millisecond transients

---

## Architecture

### Components

1. **Neural Audio Codec** (Encodec 24 kHz)
   - 8 residual quantizers
   - 1024 codebook size per quantizer
   - Frozen pretrained weights

2. **Bidirectional Conformer**
   - 2 layers, 512 hidden dimensions
   - 8 attention heads, kernel size 31
   - 23.5M trainable parameters

3. **Parallel Masked Decoder**
   - MaskGIT-style iterative refinement
   - Cosine unmasking schedule
   - 12 decoding iterations

---

## Datasets

### ESC-50
- 2,000 five-second recordings
- 50 environmental sound classes
- 5 high-level categories
- 5-fold cross-validation (Fold 1: 1,600 train / 400 val)

### UrbanSound8K
- 8,732 urban sound recordings
- 10 classes with 10-fold cross-validation
- Classification experiments only

---

## Training

### Configuration

```python
# Audio Generation
batch_size = 8
epochs = 100
optimizer = Adam(lr=1e-4)
scheduler = OneCycleLR(max_lr=1e-4, pct_start=0.3)
num_quantizers = 8
hidden_dim = 512
depth = 2
steps = 12

# Classification  
batch_size = 32
epochs = 50
optimizer = Adam(lr=5e-5)
scheduler = OneCycleLR(max_lr=5e-5, pct_start=0.3)
num_quantizers = 12
freeze_encoder = True
pooling = "mean"
```

### Evaluation Metrics

**Acoustic Metrics**
- Signal-to-Noise Ratio (SNR)
- Zero-Crossing Rate (ZCR)
- RMS Energy
- Spectral Entropy
- Spectral Centroid
- Spectral Rolloff

**Similarity Metrics**
- Spectral Convergence
- Log Spectral Distance

**Statistical Testing**
- Kolmogorov-Smirnov tests

---

## Results

### Classification Performance

| Model | ESC-50 | UrbanSound8K |
|-------|--------|--------------|
| AST (baseline) | 88.5% | 84.2% |
| SoundStorm (frozen) | 72.3% | 69.8% |
| SoundStorm (fine-tuned) | 79.6% | 75.4% |

Strong on temporal structure: clock (92%), dog (87%), keyboard (85%)  
Weak on spectral complexity: helicopter (64%), chainsaw (68%), engine (71%)

### Audio Generation Quality

| Metric | Real Audio | Generated | Difference |
|--------|-----------|-----------|------------|
| SNR (dB) | 14.97 ± 8.10 | 23.18 ± 3.85 | +54.89% |
| Zero-Crossing Rate | 0.067 ± 0.075 | 0.022 ± 0.008 | -66.79% |
| RMS Energy | 0.105 ± 0.088 | 0.068 ± 0.010 | -34.92% |
| Spectral Entropy | 9.85 ± 0.80 | 8.91 ± 0.39 | -9.49% |
| Spectral Centroid (Hz) | 3090 ± 1392 | 1392 ± 258 | -54.95% |
| Spectral Rolloff (Hz) | 5735 ± 3259 | 3259 ± 729 | -43.18% |

All metrics show significant differences (p < 0.05) except duration.

---

## Key Findings

### Failure Modes

1. **High-Frequency Attenuation** 
   - 67% reduction in zero-crossing rate
   - 55% reduction in spectral centroid
   - Cause: Encodec optimized for speech (0-5 kHz), LibriLight bias

2. **Temporal Over-Smoothing**
   - 55% increase in SNR
   - Missing transients and sharp features
   - Cause: Parallel generation averages multiple continuations

3. **Reduced Spectral Diversity**
   - 9.5% lower spectral entropy
   - Homogeneous outputs
   - Cause: Limited training data (32 samples/class)

4. **Lack of Temporal Evolution**
   - Stationary spectrograms
   - Missing long-range dependencies
   - Cause: Iterative refinement schedule limitations

---

## Installation

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.7
```

### Setup

```bash
git clone https://github.com/ZhanliangAaronWang/CIS7000_Final_Project.git
cd CIS7000_Final_Project

python -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install soundfile encodec soundstorm-pytorch librosa scipy matplotlib seaborn tqdm
```

### Dependencies
```
torch>=2.0.0
torchaudio>=2.0.0
soundfile>=0.12.0
encodec>=0.1.1
soundstorm-pytorch>=0.1.0
librosa>=0.10.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## Usage

### Training

```bash
# Fine-tune SoundStorm for audio generation
python train.py \
  --data_dir /path/to/datafiles \
  --fold 1 \
  --batch_size 8 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --num_quantizers 8 \
  --hidden_dim 512 \
  --depth 2 \
  --steps 12 \
  --output_dir ./checkpoints

# Train classification head (frozen encoder)
python train_classifier.py \
  --data_dir /path/to/datafiles \
  --fold 1 \
  --freeze_encoder \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 5e-5 \
  --num_quantizers 12 \
  --output_dir ./checkpoints

# Train classification head (joint training)
python train_classifier.py \
  --data_dir /path/to/datafiles \
  --fold 1 \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 5e-5 \
  --num_quantizers 12 \
  --output_dir ./checkpoints
```

### Evaluation

```bash
python evaluate.py \
  --real_audio /path/to/real/audio \
  --generated_audio /path/to/generated/audio \
  --sample_rate 24000 \
  --num_samples 100 \
  --output_dir ./results
```

This generates:
- evaluation_results.json (quantitative metrics)
- metric_distributions.png (histogram comparisons)
- metric_boxplots.png (statistical comparisons)

---

## Project Structure

```
CIS7000_Final_Project/
├── data/
│   └── datafiles/
│       ├── esc_train_data_1.json
│       ├── esc_eval_data_1.json
│       └── ...
├── train.py
├── train_classifier.py
├── evaluate.py
├── results/
│   ├── training_curves.png
│   ├── evaluation_results.json
│   ├── metric_distributions.png
│   └── metric_boxplots.png
├── checkpoints/
│   ├── best_model_fold1.pt
│   └── ...
├── requirements.txt
├── README.md
└── report.pdf
```

### Data Format

JSON files structure:

```json
{
  "data": [
    {
      "wav": "/absolute/path/to/audio.wav",
      "labels": "/m/07xxxxx"
    }
  ]
}
```

---

## Reproducibility

### Hardware
- GPU: NVIDIA with 16GB+ VRAM (tested on A100)
- RAM: 32GB+
- Storage: 50GB+

### Training Time
- SoundStorm fine-tuning: ~12 hours on A100
- Classification training: ~2 hours
- Generation (100 samples): ~1 minute

### Random Seeds
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Future Work

1. Higher-bitrate codecs with more quantizers
2. Diverse pretraining on speech + environmental + music
3. Increased model capacity
4. Longer convolution kernels for temporal patterns
5. Explicit temporal conditioning
6. Larger datasets (AudioSet: 2M samples, FSD50K: 50K samples)
7. Class-conditional generation
8. Perceptual metrics (Fréchet Audio Distance, MUSHRA)

---

## Citation

```bibtex
@article{wang2024soundstorm,
  title={Adapting SoundStorm for Environmental Sound Generation},
  author={Wang, Aaron and Tripathi, Tripti and Chen, Kathryn},
  journal={CIS 7000 Final Project Report},
  institution={University of Pennsylvania},
  year={2024}
}
```

---

## References

### Core Papers
- SoundStorm: Borsos et al., 2023
- Conformer: Gulati et al., 2020
- MaskGIT: Chang et al., 2022
- Encodec: Défossez et al., 2022
- AST: Gong et al., 2021

### Datasets
- ESC-50: Piczak, 2015
- UrbanSound8K: Salamon et al., 2014

---

## Contact

- Aaron Wang: aaronwang@seas.upenn.edu
- Tripti Tripathi: triptit@seas.upenn.edu
- Kathryn Chen: kathrync@seas.upenn.edu

Course: CIS 7000 - Advanced Topics in Machine Learning  
Institution: University of Pennsylvania  
Semester: Fall 2024

---

## Acknowledgments

Thanks to CIS 7000 course staff, University of Pennsylvania for computational resources, and the original SoundStorm authors.
