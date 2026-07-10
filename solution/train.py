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

# Reserve enough time for loading calibration/validation splits, scoring them,
# calibrating the threshold and writing the final artifacts.
MARGIN_SECONDS = 240
INTERNAL_VAL_FRACTION = 0.10
SCHEDULER_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2


def save_checkpoint(model, path):
    """Atomically save a checkpoint so a timeout cannot leave a corrupt file."""
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)


def stratified_internal_split(y, val_fraction, rng):
    """Return train/validation indices while preserving both classes."""
    real_idx = np.flatnonzero(y == 0)
    ai_idx = np.flatnonzero(y == 1)

    real_idx = rng.permutation(real_idx)
    ai_idx = rng.permutation(ai_idx)

    n_real_val = max(1, int(round(val_fraction * len(real_idx))))
    n_ai_val = max(1, int(round(val_fraction * len(ai_idx))))

    val_idx = np.concatenate(
        [real_idx[:n_real_val], ai_idx[:n_ai_val]]
    )
    tr_idx = np.concatenate(
        [real_idx[n_real_val:], ai_idx[n_ai_val:]]
    )

    return rng.permutation(tr_idx), rng.permutation(val_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    if args.timeout_seconds <= MARGIN_SECONDS:
        raise ValueError(
            f"--timeout_seconds must be greater than {MARGIN_SECONDS}"
        )

    set_seed()
    start_time = time.perf_counter()

    # Keep the stricter of:
    # 1) the official script timeout minus the post-training safety margin;
    # 2) the local 5x Appendix-C reference budget configured in common.py.
    training_seconds = min(
        args.timeout_seconds - MARGIN_SECONDS,
        TRAIN_BUDGET_SECONDS,
    )
    deadline = start_time + max(1, training_seconds)

    print("=" * 70)
    print("Starting train.py")
    print(f"Timeout received: {args.timeout_seconds}s")
    print(f"Training budget: {training_seconds}s")
    print(f"Safety margin: {MARGIN_SECONDS}s")
    print("=" * 70)

    models_dir = ARTIFACTS_DIR / "models"
    metrics_dir = ARTIFACTS_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.pt"

    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz", mmap_mode="r")
    X = cache["X"]
    y = np.asarray(cache["y"], dtype=np.int64)


    print(f"Loaded dataset: {len(X)} images")
    print(f"Real images: {(y==0).sum()}")
    print(f"AI images: {(y==1).sum()}")


    # Internal split is used only for checkpoint selection.
    # The official validation split remains untouched until final evaluation.
    rng = np.random.default_rng(SEED)
    tr_idx, val_idx = stratified_internal_split(
        y, INTERNAL_VAL_FRACTION, rng
    )

    real_tr_idx = tr_idx[y[tr_idx] == 0]
    ai_tr_idx = tr_idx[y[tr_idx] == 1]

    if len(real_tr_idx) == 0 or len(ai_tr_idx) == 0:
        raise RuntimeError("Both classes are required in the training split.")
    
    print("\nInternal split")
    print(f"Train images: {len(tr_idx)}")
    print(f"Validation images: {len(val_idx)}")

    print(f"Train real: {len(real_tr_idx)}")
    print(f"Train AI: {len(ai_tr_idx)}")

    # Keep the prepared cache as uint8 in RAM and convert only each mini-batch.
    Xt = torch.from_numpy(np.asarray(X)).permute(0, 3, 1, 2)
    yt = torch.from_numpy(y.astype(np.float32))

    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=SCHEDULER_EPOCHS,
    eta_min=LR * 0.1,
)

    # Each epoch is explicitly balanced, so pos_weight must not be used.
    criterion = nn.BCEWithLogitsLoss()

    # Guarantee that a valid checkpoint exists even if the available timeout is
    # unexpectedly very small.
    save_checkpoint(model, model_path)

    best_auc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        MIN_EPOCH_SECONDS = 160

        remaining = deadline - time.perf_counter()

        if remaining < MIN_EPOCH_SECONDS:
            print(
                f"Only {remaining:.1f}s remaining; "
                "not enough for another complete epoch."
            )
            break

        if time.perf_counter() >= deadline:
            print("Time budget reached before starting the next epoch.")
            break

        # Balanced mini-epoch:
        # use every real image once and sample the same number of AI images.
        ai_sample = rng.choice(
            ai_tr_idx,
            size=len(real_tr_idx),
            replace=len(ai_tr_idx) < len(real_tr_idx),
        )
        epoch_idx = np.concatenate([real_tr_idx, ai_sample])
        order = torch.from_numpy(rng.permutation(epoch_idx))

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )
        print(
            f"Remaining training time: {remaining:.1f}s"
        )
        print(
            f"Balanced epoch size: {len(epoch_idx)} "
            f"({len(real_tr_idx)} real + {len(ai_sample)} ai)"
        )

        model.train()
        epoch_loss = 0.0
        n_seen = 0
        out_of_time = False

        for i in range(0, len(order), BATCH):
            if time.perf_counter() >= deadline:
                out_of_time = True
                break

            idx = order[i:i + BATCH]
            xb = Xt[idx].float().div_(255.0)
            yb = yt[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = len(idx)
            epoch_loss += loss.item() * batch_size
            n_seen += batch_size

            if (i // BATCH) % 20 == 0:
                print(
                    f"batch {i//BATCH:3d} "
                    f"loss={loss.item():.4f}"
                )

        # Do not evaluate a completely empty epoch.
        if n_seen == 0:
            print("Time budget reached before processing a training batch.")
            break

        if out_of_time:
            print(
                f"Epoch {epoch} interrupted after {n_seen}/{len(order)} images; "
                "discarding partial epoch."
            )
            break

        

        internal_scores = score_images(model, np.asarray(X[val_idx]))
        auc = roc_auc_score(y[val_idx], internal_scores)

        if auc > best_auc:
            print(f"New best model: AUC {best_auc:.4f} -> {auc:.4f}")
            best_auc = auc
            epochs_without_improvement = 0
            save_checkpoint(model, model_path)
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

        elapsed = time.perf_counter() - start_time

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch}/{EPOCHS} "
            f"loss={epoch_loss / n_seen:.4f} "
            f"internal_auc={auc:.4f} "
            f"best={best_auc:.4f} "
            f"lr={current_lr:.6f} "
            f"seen={n_seen} "
            f"elapsed={elapsed:.1f}s"
        )
        
        scheduler.step()
        
        if out_of_time:
            print("Time budget reached, stopping training early.")
            break

        print(
            f"Epoch {epoch} completed\n"
            f"   images seen : {n_seen}\n"
            f"   loss        : {epoch_loss/n_seen:.4f}\n"
            f"   internal AUC: {auc:.4f}\n"
            f"   best AUC    : {best_auc:.4f}\n"
            f"   elapsed     : {elapsed:.1f}s"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining finished.")
    print("Loading best checkpoint...")
    print("Calibrating threshold...")

    # Load the best internal-validation checkpoint.
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Calibrate only on REAL calibration images, leaving a safety margin below
    # the hard 20% FPR constraint through CAL_TARGET_FPR in common.py.
    cal_X, cal_y = load_split("calibration")
    cal_real_scores = score_images(model, cal_X[cal_y == 0])
    threshold = calibrate_threshold(cal_real_scores)

    with open(models_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # Final independent verification.
    results = {
        "threshold": threshold,
        "best_internal_auc": best_auc,
    }

    for split in ("validation", "validation_augmented"):
        sx, sy = load_split(split)
        results[split] = evaluate(
            sy,
            score_images(model, sx),
            threshold,
        )

    with open(metrics_dir / "task02.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"threshold={threshold:.6f}")

    m = results["validation"]
    status = "OK" if m["fpr"] <= 0.20 else "VIOLATED"
    print(
        f"validation: recall_ai={m['recall']:.4f} "
        f"fpr={m['fpr']:.4f} "
        f"({status}, constraint <= 0.20) "
        f"roc_auc={m['roc_auc']:.4f}"
    )

    m = results["validation_augmented"]
    print(
        "validation_augmented (reference only): "
        f"recall_ai={m['recall']:.4f} "
        f"fpr={m['fpr']:.4f} "
        f"roc_auc={m['roc_auc']:.4f}"
    )

    total_elapsed = time.perf_counter() - start_time
    print(f"Total train.py elapsed: {total_elapsed:.1f}s")


    print("=" * 70)
    print("FINAL RESULTS")

    print(f"Threshold : {threshold:.4f}")

    print(
        f"Validation:"
        f" recall={results['validation']['recall']:.4f}"
        f" fpr={results['validation']['fpr']:.4f}"
    )

    print(
        f"Validation_aug:"
        f" recall={results['validation_augmented']['recall']:.4f}"
        f" fpr={results['validation_augmented']['fpr']:.4f}"
    )

    print(f"Total runtime: {time.perf_counter()-start_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
