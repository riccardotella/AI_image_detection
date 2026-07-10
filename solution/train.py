# train.py — Task 2: train SmallCNN on the prepared cache, auto-calibrate the
# operating threshold on data/calibration, verify on data/validation.

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import (
    ARTIFACTS_DIR,
    BATCH,
    EPOCHS,
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

# Time reserved after training for calibration, validation and saving,
# so a timeout kill never leaves the pipeline without threshold/metrics.
MARGIN_SECONDS = 180


def save_checkpoint(model, path):
    # Write-then-rename so a timeout kill cannot leave a corrupt model.pt.
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
    model_path = models_dir / "model.pt"

    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz")
    X, y = cache["X"], cache["y"]

    # Internal 90/10 split (train data only) to select the best checkpoint;
    # data/validation is used exclusively for final verification below.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_internal = int(0.1 * len(X))
    val_idx, tr_idx = perm[:n_internal], perm[n_internal:]

    Xt = torch.from_numpy(X).permute(0, 3, 1, 2)  # uint8 (N, 3, H, W)
    yt = torch.from_numpy(y.astype(np.float32))

    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # Class imbalance (~1:5 real:ai): weight the positive class down so both
    # classes contribute equally to the loss.
    n_real = float((y[tr_idx] == 0).sum())
    n_ai = float((y[tr_idx] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_ai))

    best_auc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(tr_idx))
        epoch_loss, n_seen, out_of_time = 0.0, 0, False
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            xb = Xt[idx].float().div_(255.0)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
            if time.perf_counter() > deadline:
                out_of_time = True
                break

        scheduler.step()
        auc = roc_auc_score(y[val_idx], score_images(model, X[val_idx]))
        if auc > best_auc:
            best_auc = auc
            save_checkpoint(model, model_path)
        print(f"epoch {epoch}/{EPOCHS} loss={epoch_loss / n_seen:.4f} "
              f"internal_auc={auc:.4f} best={best_auc:.4f}")
        if out_of_time:
            print("Time budget reached, stopping training early.")
            break

    # Automatic calibration on data/calibration: threshold from REAL images
    # only, targeting FPR <= 20% (no manual thresholds).
    model.load_state_dict(torch.load(model_path, weights_only=True))
    cal_X, cal_y = load_split("calibration")
    threshold = calibrate_threshold(score_images(model, cal_X[cal_y == 0]))
    with open(models_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # Independent verification on data/validation (never used for training).
    val_X, val_y = load_split("validation")
    metrics = evaluate(val_y, score_images(model, val_X), threshold)
    with open(metrics_dir / "task02.json", "w") as f:
        json.dump({"threshold": threshold, **metrics}, f, indent=2)

    status = "OK" if metrics["fpr"] <= 0.20 else "VIOLATED"
    print(f"threshold={threshold:.6f}")
    print(f"validation recall_ai={metrics['recall']:.4f} fpr={metrics['fpr']:.4f} "
          f"({status}, constraint <= 0.20) roc_auc={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
