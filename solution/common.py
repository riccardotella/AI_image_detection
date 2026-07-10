# common.py — shared configuration and helpers for all pipeline scripts.

import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 96
SEED = 42
BATCH = 128
EPOCHS = 12  # the training loop is additionally capped by TRAIN_BUDGET_SECONDS
LR = 2e-3
DEVICE = "cpu"
TARGET_FPR = 0.20      # hard constraint verified on data/validation
CAL_TARGET_FPR = 0.15  # calibration quantile: finite-sample margin under 20%
# Assignment limit: training must stay within 5x the Appendix C reference
# runtime (measured 179.7s in the grading-like container -> 5x ~ 898s).
TRAIN_BUDGET_SECONDS = 900

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    """Fix all RNG seeds and force deterministic, CPU-thread-safe execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized; safe to continue


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _image_bytes(x):
    if isinstance(x, bytes):
        return x
    if isinstance(x, dict) and "bytes" in x:
        return x["bytes"]
    raise ValueError(f"Unsupported image cell type: {type(x)}")


def _decode(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_split(name):
    """Load data/<name>/ parquet files.

    Returns (images, labels): images is a uint8 array (N, IMG_SIZE, IMG_SIZE, 3);
    labels is int64 with 0 = real, 1 = ai_generated for labeled splits.
    For the unlabeled predict split the second value is the row_id array instead.
    """
    files = sorted((DATA_DIR / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR / name}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    images = np.stack([_decode(_image_bytes(x)) for x in df["image"]])

    if "source_class" in df.columns:
        labels = (df["source_class"].to_numpy() != 0).astype(np.int64)
    elif "source class" in df.columns:
        labels = (df["source class"].to_numpy() != 0).astype(np.int64)
    elif "row_id" in df.columns:
        labels = df["row_id"].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"No label or row_id column in split '{name}'")

    return images, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Small CNN for binary classification; forward returns one logit per image."""

    def __init__(self, k=32):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2))
            return layers

        self.net = nn.Sequential(
            *block(3, k),
            *block(k, 2 * k),
            *block(2 * k, 4 * k),
            *block(4 * k, 4 * k, pool=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * k, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def score_images(model, images, batch=BATCH):
    """Sigmoid scores for uint8 images (N, H, W, 3); higher = more ai_generated."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].float().div_(255.0)
            scores.append(torch.sigmoid(model(xb)))
    return torch.cat(scores).numpy()


# ---------------------------------------------------------------------------
# Calibration and evaluation
# ---------------------------------------------------------------------------
def calibrate_threshold(scores_real, target_fpr=CAL_TARGET_FPR):
    """Threshold from scores on REAL calibration images.

    Predicting ai_generated when score > threshold keeps the calibration
    false-positive rate at most target_fpr. The default calibrates below the
    20% constraint to leave finite-sample margin on unseen validation data.
    """
    scores_real = np.asarray(scores_real, dtype=np.float64)
    return float(np.quantile(scores_real, 1.0 - target_fpr, method="higher"))


def evaluate(y_true, y_score, threshold):
    """Binary metrics at the given operating threshold (positive = score > threshold)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score > threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
