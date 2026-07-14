"""Shared configuration and utilities for the AMLS image-detection pipeline."""

import csv
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

IMG_SIZE = 96
BATCH = 256
LR = 2e-3
SEED = 42
TARGET_FPR = 0.20
CAL_TARGET_FPR = 0.16
TRAIN_BUDGET_SECONDS = 850

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def image_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict) and "bytes" in value:
        return image_bytes(value["bytes"])
    raise ValueError(f"Unsupported image cell type: {type(value)}")


def decode_image(value):
    with Image.open(io.BytesIO(image_bytes(value))) as image:
        image = image.convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.uint8)


def load_split(name, with_source=False):
    """Load a labeled split or the unlabeled predict split.

    Returns ``(images, labels)`` for labeled data and ``(images, row_ids)`` for
    predict. With ``with_source=True``, labeled data returns source classes too.
    """
    files = sorted((DATA_DIR / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR / name}")

    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    images = np.stack([decode_image(value) for value in frame["image"]])

    label_column = next(
        (column for column in ("source_class", "source class") if column in frame),
        None,
    )
    if label_column is not None:
        source = frame[label_column].to_numpy(dtype=np.int64)
        labels = (source != 0).astype(np.int64)
        return (images, labels, source) if with_source else (images, labels)

    row_column = next(
        (column for column in ("row_id", "row id") if column in frame),
        None,
    )
    if row_column is None:
        raise ValueError(f"No labels or row IDs in split '{name}'")
    if with_source:
        raise ValueError("with_source=True is valid only for labeled splits")
    return images, frame[row_column].to_numpy(dtype=np.int64)


class SmallCNN(nn.Module):
    """Small stride-based CNN designed for CPU training."""

    def __init__(self, channels=32):
        super().__init__()

        def block(in_channels, out_channels, stride):
            return (
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            *block(3, channels, 2),
            *block(channels, 2 * channels, 2),
            *block(2 * channels, 4 * channels, 2),
            *block(4 * channels, 4 * channels, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Identity(),  # preserves the original checkpoint key layout
            nn.Linear(4 * channels, 1),
        )

    def forward(self, inputs):
        return self.net(inputs).squeeze(1)


def score_images(model, images, batch_size=BATCH):
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    scores = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size].float().div_(255.0)
            scores.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(scores).numpy() if scores else np.empty(0, dtype=np.float32)


def calibrate_threshold(real_scores, target_fpr=CAL_TARGET_FPR):
    real_scores = np.asarray(real_scores, dtype=np.float64)
    return float(np.quantile(real_scores, 1.0 - target_fpr, method="higher"))


def evaluate(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def stratified_split(labels, validation_fraction, rng):
    real = rng.permutation(np.flatnonzero(labels == 0))
    ai = rng.permutation(np.flatnonzero(labels == 1))
    n_real = max(1, round(validation_fraction * len(real)))
    n_ai = max(1, round(validation_fraction * len(ai)))
    validation = np.concatenate((real[:n_real], ai[:n_ai]))
    training = np.concatenate((real[n_real:], ai[n_ai:]))
    return rng.permutation(training), rng.permutation(validation)


def balanced_cycle(real_indices, ai_indices, rng):
    sampled_ai = rng.choice(
        ai_indices,
        size=len(real_indices),
        replace=len(ai_indices) < len(real_indices),
    )
    return rng.permutation(np.concatenate((real_indices, sampled_ai)))


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def run_prediction(model_file, threshold_file, task_name):
    set_seed()
    models_dir = ARTIFACTS_DIR / "models"

    model = SmallCNN()
    model.load_state_dict(
        torch.load(models_dir / model_file, map_location="cpu", weights_only=True)
    )
    with (models_dir / threshold_file).open(encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])

    images, row_ids = load_split("predict")
    predictions = (score_images(model, images) > threshold).astype(np.int64)

    output = ARTIFACTS_DIR / task_name / "predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "predicted_label"))
        writer.writerows(zip(row_ids, predictions))

    print(
        f"Wrote {len(predictions)} predictions "
        f"({int(predictions.sum())} AI) -> {output}"
    )
