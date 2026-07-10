# train_augmented.py — Task 3: continue from the Task 2 checkpoint with
# on-the-fly robustness augmentations (JPEG compression, blur, rescale, noise),
# re-calibrate on data/calibration_augmented, evaluate on both validation splits.

import argparse
import io
import json
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

from common import (
    ARTIFACTS_DIR,
    BATCH,
    EPOCHS,
    IMG_SIZE,
    LR,
    SEED,
    TRAIN_BUDGET_SECONDS,
    SmallCNN,
    calibrate_threshold,
    evaluate,
    load_split,
    score_images,
    set_seed,
)

# Time reserved after training for calibration + two validation passes.
MARGIN_SECONDS = 240
AUG_PROB = 0.5  # fraction of training images augmented in each batch


def augment(img, rng):
    """Apply one random robustness transform to a uint8 (H, W, 3) image."""
    choice = int(rng.integers(4))
    if choice == 0:  # JPEG-style compression
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=int(rng.integers(30, 91)))
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    if choice == 1:  # gaussian blur
        radius = float(rng.uniform(0.5, 1.5))
        blurred = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius))
        return np.asarray(blurred, dtype=np.uint8)
    if choice == 2:  # down/up rescale
        side = max(8, int(IMG_SIZE * float(rng.uniform(0.5, 0.9))))
        small = Image.fromarray(img).resize((side, side), Image.BILINEAR)
        return np.asarray(small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
    # mild additive gaussian noise
    noise = rng.normal(0.0, float(rng.uniform(3.0, 10.0)), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_batch(imgs, rng):
    return np.stack([augment(im, rng) if rng.random() < AUG_PROB else im for im in imgs])


def save_checkpoint(model, path):
    # Write-then-rename so a timeout kill cannot leave a corrupt checkpoint.
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()
    # Training loop budget: the stricter of the script timeout (minus the margin
    # reserved for calibration/validation) and the 5x-reference limit.
    deadline = time.perf_counter() + min(
        args.timeout_seconds - MARGIN_SECONDS, TRAIN_BUDGET_SECONDS
    )

    set_seed()

    models_dir = ARTIFACTS_DIR / "models"
    metrics_dir = ARTIFACTS_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model_aug.pt"

    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz")
    X, y = cache["X"], cache["y"]

    # Same internal 90/10 split as train.py (same seed); the internal holdout is
    # augmented once so the best checkpoint is selected for robustness.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_internal = int(0.1 * len(X))
    val_idx, tr_idx = perm[:n_internal], perm[n_internal:]
    val_aug = np.stack([augment(im, rng) for im in X[val_idx]])

    # Continue from the Task 2 checkpoint (required starting point).
    model = SmallCNN()
    model.load_state_dict(torch.load(models_dir / "model.pt", weights_only=True))

    # Fine-tuning from the Task 2 checkpoint: lower LR than training from
    # scratch, decayed with cosine annealing to stabilize the late epochs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR / 4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    n_real = float((y[tr_idx] == 0).sum())
    n_ai = float((y[tr_idx] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_ai))

    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(tr_idx)
        epoch_loss, n_seen, out_of_time = 0.0, 0, False
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            xb_np = augment_batch(X[idx], rng)
            xb = torch.from_numpy(xb_np).permute(0, 3, 1, 2).float().div_(255.0)
            yb = torch.from_numpy(y[idx].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
            if time.perf_counter() > deadline:
                out_of_time = True
                break

        scheduler.step()
        auc = roc_auc_score(y[val_idx], score_images(model, val_aug))
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(model, model_path)
        print(f"epoch {epoch}/{EPOCHS} loss={epoch_loss / n_seen:.4f} "
              f"internal_aug_auc={auc:.4f} best={best_auc:.4f}")
        if out_of_time:
            print("Time budget reached, stopping training early.")
            break

    # Automatic re-calibration on data/calibration_augmented (REAL images only,
    # FPR <= 20%; no manual thresholds).
    model.load_state_dict(torch.load(model_path, weights_only=True))
    cal_X, cal_y = load_split("calibration_augmented")
    threshold = calibrate_threshold(score_images(model, cal_X[cal_y == 0]))
    with open(models_dir / "threshold_aug.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # Verification on BOTH validation splits (never used for training).
    results = {"threshold": threshold}
    for split in ("validation", "validation_augmented"):
        sx, sy = load_split(split)
        results[split] = evaluate(sy, score_images(model, sx), threshold)
    with open(metrics_dir / "task03.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"threshold={threshold:.6f}")
    for split, target in (("validation", 0.8), ("validation_augmented", 0.6)):
        m = results[split]
        fpr_ok = "OK" if m["fpr"] <= 0.20 else "VIOLATED"
        print(f"{split}: recall_ai={m['recall']:.4f} (target >= {target}) "
              f"fpr={m['fpr']:.4f} ({fpr_ok}, constraint <= 0.20) "
              f"roc_auc={m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
