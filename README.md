# AI Image Detection under CPU and Time Constraints

An end-to-end, lightweight machine learning pipeline designed to detect AI-generated images under strict CPU-only and wall-clock execution limits. Developed as part of the **Architecture of Machine Learning Systems (AMLS)** course at TU Berlin / BIFOLD.

---

##  Overview

Generative models (Midjourney, DALL-E 3, Stable Diffusion variants) produce synthetic images with remarkable fidelity. However, automated detectors often rely on fragile artifacts or massive neural networks that are impractical to retrain or deploy in resource-constrained, CPU-only environments.

This project delivers a robust, self-contained detection pipeline that:
1. Operates under strict systems constraints to ensure reproducibility and fairness between different systems.
2. Formulates detection as a highly imbalanced binary classification task (real photographs vs. five merged synthetic sources: SD 2.1, SDXL, SD 3, DALL-E 3, Midjourney).
3. Optimizes for maximum AI recall while maintaining a hard False Positive Rate (FPR) $\le 20\%$** on authentic images via automated threshold calibration.

---

## Key Results

| Split | Model | Recall (AI) | FPR (Real) | Precision | F1 | ROC AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation (Clean)** | **SmallCNN (Task 2)** | **0.8098** | **0.1755** | 0.9583 | 0.8778 | **0.8864** |
| Validation (Clean) | Random Forest Baseline | 0.5994 | 0.1649 | 0.9476 | 0.7343 | 0.7912 |
| **Validation (Augmented)** | **Robust SmallCNN (Task 3)** | **0.6073** | **0.1551** | 0.9515 | 0.7414 | **0.7909** |
| Validation (Augmented) | Task 2 Model (Unaugmented) | 0.7364 | 0.2620 *(fails constraint)* | 0.9337 | 0.8234 | 0.8097 |

* **Task 2 Goal Achieved**: Reached $80.98\%$ AI recall at $17.55\%$ FPR (surpassing the $\ge 80\%$ recall and $\le 20\%$ FPR target). Outperformed the 20-feature Random Forest baseline by $+21.0$ pp recall and $+0.095$ AUC.
* **Task 3 Robustness Shift**: Reduced augmented validation FPR from $26.20\%$ down to $15.51\%$ ($-10.7$ pp), bringing the model into full compliance with the $\le 20\%$ FPR requirement under heavy corruptions while exceeding the $\ge 60\%$ recall requirement.

---

## 🛠 System Pipeline & Methods

```
data/train/ ──► [ clean.py ] ──► [ prepare.py ] ──► [ train.py ] ──────────► [ predict.py ]
                (SHA-256 Dedupe)   (96x96 uint8)    (SmallCNN, Balanced)     (task02/predictions.csv)
                                                           │
                                                           ▼
                                               [ train_augmented.py ] ────► [ predict_augmented.py ]
                                               (Corruptions + Dual Calib)    (task03/predictions.csv)
```

### 1. Data Auditing & Deterministic Cleaning (`clean.py`, `prepare.py`)
* **Shortcut Discovery**: Exploratory analysis revealed severe acquisition shortcuts in the training set:
  * *Geometry*: $100\%$ of synthetic images are square ($320\times320$ or $270\times270$, aspect ratio $1.0$), whereas real photos have varying aspect ratios (median $1.333$).
  * *File size*: Median $48.7\text{ kB}$ (real) vs. $26.0\text{ kB}$ (AI).
* **Cleaning Protocol**:
  * Decoded all 29,688 training images using PIL; zero corrupted images found.
  * Hashed image bytes with SHA-256; detected 299 duplicate groups and dropped 312 redundant rows ($98.95\%$ data retained: 4,887 real, 24,489 AI).
  * Rather than exploiting the aspect-ratio shortcut, all images are preprocessed and resized bilinearly to a fixed $96 \times 96 \times 3$ representation and cached as raw `uint8` arrays to eliminate repeated JPEG decode overhead during training.

### 2. Modeling under Time Budgets (`train.py`, `predict.py`)
* **Architecture (`SmallCNN`)**: Custom 7-layer CNN (~435k parameters) with 4 stages, strided convolutions ($96\to 48\to 24\to 12$), BatchNorm, ReLU, Global Average Pooling, and a single logit head (no dropout).
* **Training Strategy**:
  * **Balanced Cyclic Sampling**: Pairs every real training sample with an equal-sized random subset of AI images per cycle to balance gradients and BatchNorm statistics.
  * **Wall-Clock Guard**: Iterates with an internal budget of ~550 seconds, validates every 50 optimizer steps, and saves atomic checkpoints to avoid corrupting model artifacts on script termination.
* **Automated Calibration**: Evaluates quantile-based operating thresholds on `data/calibration/` using a target FPR of $0.155$ to ensure that validation FPR stays safely below $0.20$.

### 3. Realistic Augmentation & Robust Fine-Tuning (`train_augmented.py`)
* Fine-tunes the Task 2 checkpoint using realistic post-processing corruptions (sampled with $p \approx 0.765$):
  * **JPEG Recompression** (quality 15–75)
  * **Gaussian Blur** (radius 1.0–4.5)
  * **Downscale-Upscale Degradation** (scale factor 0.15–0.60)
  * **Additive Gaussian Noise** ($\sigma = 8\text{--}30$)
  *(Spatial flips/rotations/color-jitters were omitted to preserve high-frequency compression signatures.)*
* **Dual-Domain Calibration**: Thresholds are computed separately on clean and corrupted calibration data; the more conservative (higher) threshold is selected.

### 4. Explainability & Failure Diagnostics
* **Gradient Saliency & Occlusion ($8\times8$)**:
  * Heatmaps display **diffuse, edge- and texture-level patterns** across the entire frame rather than highlighting distinct semantic entities.
  * *False Positives*: Strong repetitive textures in real images (e.g., giraffe coat patterns, train paneling) trigger false AI detections.
  * *False Negatives*: Clean, smooth AI images fail to trigger high-frequency artifacts and pass undetected.
* **Generator Disparity**: Recall is highest on **DALL-E 3** ($90.43\%$) and **SDXL** ($88.30\%$), and lowest on **Midjourney** ($64.52\%$). Under post-processing corruptions, **SD 3** suffers the sharpest drop ($-21.1$ pp).

---

## Submission Repository Structure

```text
├── Dockerfile                  # Self-contained CPU Docker environment (< 4 GB)
├── requirements.txt            # Pinned dependencies (CPU PyTorch, NumPy, Pillow, Optuna, etc.)
├── solution/
│   ├── clean.py                # Audits, hashes, deduplicates, and produces cleaning manifest
│   ├── prepare.py              # Resizes retained images to 96x96 binary cache
│   ├── train.py                # Balanced, deadline-driven SmallCNN training (Task 2)
│   ├── predict.py              # Inference script producing task02/predictions.csv
│   ├── train_augmented.py      # Robust fine-tuning with realistic corruptions (Task 3)
│   ├── predict_augmented.py    # Inference script producing task03/predictions.csv
│   └── artifacts/              # Generated models, thresholds, and submission CSVs (runtime)
└── README.md
```

---

## ⚙️ Execution Guide
(The dataset was provided by the professor, if access is needed, feel free to open a request)
### 1. Build Docker Image
```bash
docker build -t amls-ai-detection .
```

### 2. Run Pipeline (Matching Evaluation Harness)
Mount the data folder into `/workspace/solution/data` as read-only:

```bash
# 1. Clean dataset
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python clean.py --timeout_seconds 600

# 2. Prepare pre-resized binary caches
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python prepare.py --timeout_seconds 600

# 3. Train base SmallCNN
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python train.py --timeout_seconds 1800

# 4. Predict Task 2 test holdout
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python predict.py --timeout_seconds 600

# 5. Robust fine-tuning
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python train_augmented.py --timeout_seconds 1800

# 6. Predict Task 3 test holdout
docker run --cpus 8 --network none -v $(pwd)/data:/workspace/solution/data:ro amls-ai-detection \
    python predict_augmented.py --timeout_seconds 600
```

---

## 👥 Authors
* **Riccardo Tellarini**
* **XiangYu Zhang**
* **Thomas Bove**

*Machine Learning Systems Architecture (AMLS), Summer Semester 2026, Technische Universität Berlin.*
