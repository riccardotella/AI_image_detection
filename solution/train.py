# train.py — Task 2: train SmallCNN on the prepared cache, auto-calibrate the
# operating threshold on data/calibration, verify on data/validation.

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import (
    ARTIFACTS_DIR,
    BATCH,
    CAL_TARGET_FPR,
    LR,
    SEED,
    TARGET_FPR,
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

# Use a sufficiently large internal validation set for stable checkpoint
# selection at the operating point FPR <= 0.20.
INTERNAL_VAL_FRACTION = 0.10

# The old code validated once per balanced epoch (~35 batches with BATCH=256).
# We keep a similar frequency, but training is now driven by optimizer steps
# rather than by the requirement to complete a whole epoch.
VALIDATE_EVERY_STEPS = 100

# Step the cosine scheduler after each completed balanced cycle. After the
# eighth cycle, keep the learning rate fixed at eta_min instead of allowing the
# cosine schedule to rise again.
SCHEDULER_CYCLES = 18

# Initial estimate used to reserve enough time for a final internal validation.
# It is updated after every real validation pass.
INITIAL_VALIDATION_ESTIMATE_SECONDS = 45.0
VALIDATION_SAFETY_SECONDS = 5.0


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

    val_idx = np.concatenate([real_idx[:n_real_val], ai_idx[:n_ai_val]])
    tr_idx = np.concatenate([real_idx[n_real_val:], ai_idx[n_ai_val:]])

    return rng.permutation(tr_idx), rng.permutation(val_idx)


def make_balanced_cycle(real_idx, ai_idx, rng):
    """Create one shuffled 50/50 cycle using every real training image once."""
    ai_sample = rng.choice(
        ai_idx,
        size=len(real_idx),
        replace=len(ai_idx) < len(real_idx),
    )
    return rng.permutation(np.concatenate([real_idx, ai_sample]))


def internal_validation(model, images, labels):
    """Evaluate recall at the same conservative FPR used for final calibration."""
    scores = score_images(model, images)
    real_scores = scores[labels == 0]

    # For checkpoint selection, determine the threshold from the REAL examples
    # of the internal holdout, then measure recall on its AI examples.
    threshold = calibrate_threshold(
        real_scores, target_fpr=CAL_TARGET_FPR
    )
    metrics = evaluate(labels, scores, threshold)
    return threshold, metrics


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
    print("Starting train.py — batch-driven training")
    print(f"Timeout received: {args.timeout_seconds}s")
    print(f"Training budget: {training_seconds}s")
    print(f"Safety margin: {MARGIN_SECONDS}s")
    print(f"Batch size: {BATCH}")
    print(f"Internal validation fraction: {INTERNAL_VAL_FRACTION:.2f}")
    print(f"Validate every: {VALIDATE_EVERY_STEPS} optimizer steps")
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
    print(f"Real images: {(y == 0).sum()}")
    print(f"AI images: {(y == 1).sum()}")

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

    balanced_cycle_size = 2 * len(real_tr_idx)
    batches_per_cycle = int(np.ceil(balanced_cycle_size / BATCH))
    print(
        f"Balanced cycle: {balanced_cycle_size} images "
        f"({len(real_tr_idx)} real + {len(real_tr_idx)} ai), "
        f"about {batches_per_cycle} batches"
    )

    # Keep training images as uint8 and convert only each mini-batch.
    Xt = torch.from_numpy(np.asarray(X)).permute(0, 3, 1, 2)
    yt = torch.from_numpy(y.astype(np.float32))

    # Materialize the small internal holdout once, avoiding repeated advanced
    # indexing and copies at every validation pass.
    internal_X = np.ascontiguousarray(X[val_idx])
    internal_y = y[val_idx]

    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_CYCLES,
        eta_min=LR * 0.1,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Always leave a valid checkpoint on disk.
    save_checkpoint(model, model_path)

    best_recall = -1.0
    best_auc = -1.0
    best_fpr = float("inf")
    best_step = 0
    best_internal_threshold = None

    total_steps = 0
    total_images_seen = 0
    completed_cycles = 0
    total_training_time = 0.0
    total_validation_time = 0.0
    validation_estimate = INITIAL_VALIDATION_ESTIMATE_SECONDS
    last_validated_step = 0

    cycle_order = make_balanced_cycle(real_tr_idx, ai_tr_idx, rng)
    cycle_cursor = 0
    cycle_loss_sum = 0.0
    cycle_images_seen = 0

    def run_and_maybe_save_validation(reason):
        nonlocal best_recall, best_auc, best_fpr
        nonlocal best_step, best_internal_threshold
        nonlocal total_validation_time, validation_estimate
        nonlocal last_validated_step

        validation_start = time.perf_counter()
        threshold, metrics = internal_validation(
            model, internal_X, internal_y
        )
        validation_duration = time.perf_counter() - validation_start

        total_validation_time += validation_duration
        # Use a conservative moving estimate for the next reservation.
        validation_estimate = max(
            10.0,
            0.5 * validation_estimate + 0.5 * validation_duration,
        )
        last_validated_step = total_steps

        recall = metrics["recall"]
        auc = metrics["roc_auc"]
        fpr = metrics["fpr"]

        # Primary selection criterion: recall at the same conservative FPR used
        # by the final calibration pipeline. AUC is only the tie-breaker.
        is_better = (
            recall > best_recall + 1e-12
            or (
                abs(recall - best_recall) <= 1e-12
                and auc > best_auc + 1e-12
            )
        )

        print(
            f"\nInternal validation ({reason}) at step {total_steps}: "
            f"recall={recall:.4f} "
            f"fpr={fpr:.4f} "
            f"auc={auc:.4f} "
            f"threshold={threshold:.6f} "
            f"time={validation_duration:.1f}s"
        )

        if is_better:
            print(
                "New best checkpoint: "
                f"recall {best_recall:.4f} -> {recall:.4f}, "
                f"AUC {best_auc:.4f} -> {auc:.4f}"
            )
            best_recall = recall
            best_auc = auc
            best_fpr = fpr
            best_step = total_steps
            best_internal_threshold = threshold
            save_checkpoint(model, model_path)
        else:
            print(
                f"Checkpoint unchanged: best recall={best_recall:.4f}, "
                f"best AUC={best_auc:.4f}, best step={best_step}"
            )

    print("\nStarting optimizer-step loop...")

    while True:
        now = time.perf_counter()
        remaining = deadline - now

        # Stop early enough to validate the current (possibly partial-cycle)
        # model once more before the local training budget expires.
        reserve = validation_estimate + VALIDATION_SAFETY_SECONDS
        if remaining <= reserve:
            print(
                f"\nStopping optimizer loop with {remaining:.1f}s remaining "
                f"to reserve about {reserve:.1f}s for final internal validation."
            )
            break

        # Begin a new balanced cycle only when the previous one is exhausted.
        if cycle_cursor >= len(cycle_order):
            completed_cycles += 1

            if completed_cycles <= SCHEDULER_CYCLES:
                scheduler.step()

            mean_cycle_loss = (
                cycle_loss_sum / cycle_images_seen
                if cycle_images_seen
                else float("nan")
            )
            print(
                f"\nBalanced cycle {completed_cycles} completed: "
                f"loss={mean_cycle_loss:.4f}, "
                f"steps={total_steps}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

            cycle_order = make_balanced_cycle(real_tr_idx, ai_tr_idx, rng)
            cycle_cursor = 0
            cycle_loss_sum = 0.0
            cycle_images_seen = 0

        idx_np = cycle_order[cycle_cursor:cycle_cursor + BATCH]
        cycle_cursor += len(idx_np)
        idx = torch.from_numpy(idx_np)

        batch_start = time.perf_counter()

        xb = Xt[idx].float().div_(255.0)
        yb = yt[idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_duration = time.perf_counter() - batch_start
        total_training_time += batch_duration

        batch_size = len(idx_np)
        total_steps += 1
        total_images_seen += batch_size
        cycle_loss_sum += loss.item() * batch_size
        cycle_images_seen += batch_size

        if total_steps == 1 or total_steps % 20 == 0:
            avg_seconds_per_batch = total_training_time / total_steps
            images_per_second = (
                total_images_seen / total_training_time
                if total_training_time > 0
                else 0.0
            )
            print(
                f"step={total_steps:4d} "
                f"loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"remaining={deadline - time.perf_counter():.1f}s "
                f"sec/batch={avg_seconds_per_batch:.2f} "
                f"img/s={images_per_second:.1f}"
            )

        # Periodic checkpoint selection. Skip it if doing so would consume the
        # time needed for the final validation pass.
        if total_steps % VALIDATE_EVERY_STEPS == 0:
            remaining = deadline - time.perf_counter()
            required = 2.0 * validation_estimate + VALIDATION_SAFETY_SECONDS

            if remaining > required:
                run_and_maybe_save_validation("periodic")
            else:
                print(
                    f"Skipping periodic validation at step {total_steps}: "
                    f"{remaining:.1f}s remain, reserving time for the final pass."
                )

    # Evaluate every optimizer update made since the previous checkpoint,
    # including updates from an unfinished balanced cycle.
    if total_steps > last_validated_step:
        run_and_maybe_save_validation("final")
    elif best_step == 0:
        # Defensive fallback for extremely short custom timeouts.
        run_and_maybe_save_validation("fallback")

    print("\nTraining finished.")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Total images seen: {total_images_seen}")
    print(f"Completed balanced cycles: {completed_cycles}")
    print(f"Training compute time: {total_training_time:.1f}s")
    print(f"Internal validation time: {total_validation_time:.1f}s")
    print(f"Best checkpoint step: {best_step}")
    print(f"Best internal recall: {best_recall:.4f}")
    print(f"Best internal FPR: {best_fpr:.4f}")
    print(f"Best internal AUC: {best_auc:.4f}")

    print("Loading best checkpoint...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    print("Calibrating threshold on data/calibration...")
    cal_X, cal_y = load_split("calibration")
    cal_real_scores = score_images(model, cal_X[cal_y == 0])
    threshold = calibrate_threshold(cal_real_scores)

    with open(models_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # Final independent verification.
    results = {
        "threshold": threshold,
        "best_internal_recall": best_recall,
        "best_internal_fpr": best_fpr,
        "best_internal_auc": best_auc,
        "best_internal_threshold": best_internal_threshold,
        "best_step": best_step,
        "total_optimizer_steps": total_steps,
        "total_images_seen": total_images_seen,
        "completed_balanced_cycles": completed_cycles,
        "training_compute_seconds": total_training_time,
        "internal_validation_seconds": total_validation_time,
    }
# Final independent verification.
    # The oracle variables are diagnostic only and are not used for prediction.
    oracle_threshold = None
    oracle_metrics = None

    for split in ("validation", "validation_augmented"):
        sx, sy = load_split(split)

        # Compute scores once and reuse them.
        split_scores = score_images(model, sx)

        results[split] = evaluate(
            sy,
            split_scores,
            threshold,
        )

        # Diagnostic only:
        # estimate the best operating point at TARGET_FPR directly on validation.
        # Do not save this threshold to threshold.json.
        if split == "validation":
            oracle_threshold = calibrate_threshold(
                split_scores[sy == 0],
                target_fpr=TARGET_FPR,
            )

            oracle_metrics = evaluate(
                sy,
                split_scores,
                oracle_threshold,
            )

    with open(metrics_dir / "task02.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"threshold={threshold:.6f}")

    m = results["validation"]
    status = "OK" if m["fpr"] <= TARGET_FPR else "VIOLATED"
    print(
        f"validation: recall_ai={m['recall']:.4f} "
        f"fpr={m['fpr']:.4f} "
        f"({status}, constraint <= {TARGET_FPR:.2f}) "
        f"roc_auc={m['roc_auc']:.4f}"
    )

    # Print the diagnostic operating point.
    if oracle_metrics is not None:
        print(
            f"validation diagnostic at {TARGET_FPR:.0%} target FPR: "
            f"recall_ai={oracle_metrics['recall']:.4f} "
            f"fpr={oracle_metrics['fpr']:.4f} "
            f"threshold={oracle_threshold:.6f}"
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
    print(f"Threshold: {threshold:.4f}")
    print(f"Best step: {best_step}/{total_steps}")
    print(
        f"Validation: recall={results['validation']['recall']:.4f} "
        f"fpr={results['validation']['fpr']:.4f}"
    )
    print(
        f"Validation_aug: recall={results['validation_augmented']['recall']:.4f} "
        f"fpr={results['validation_augmented']['fpr']:.4f}"
    )
    print(f"Total runtime: {total_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
