# train_augmented.py — Task 3: robust fine-tuning from the Task 2 checkpoint.
# Uses balanced, batch-driven training with realistic PIL-based on-the-fly
# augmentations and calibrates one threshold that protects both clean and
# augmented real images.

import argparse
import io
import json
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from common import (
    ARTIFACTS_DIR,
    BATCH,
    CAL_TARGET_FPR,
    IMG_SIZE,
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

# Reserve time after the optimizer loop for final calibration, evaluation and
# writing artifacts. The stricter local 5x-reference budget remains 850 s.
MARGIN_SECONDS = 240

INTERNAL_VAL_FRACTION = 0.10
VALIDATE_EVERY_STEPS = 100

# Use the conservative 16% target for internal checkpoint selection, while
# the final Task 3 threshold uses 18% based on the calibration sweep. Keeping
# these separate avoids changing Task 2, which still uses CAL_TARGET_FPR.
INTERNAL_SELECTION_FPR = CAL_TARGET_FPR
FINAL_CAL_TARGET_FPR = 0.18

# Fine-tuning starts from the Task 2 checkpoint, so use a lower learning rate.
FINE_TUNE_LR = LR / 4
SCHEDULER_CYCLES = 18

# Keep some clean samples in every batch to avoid catastrophic forgetting.
AUG_PROB = 0.50

INITIAL_VALIDATION_ESTIMATE_SECONDS = 15.0
VALIDATION_SAFETY_SECONDS = 5.0


def save_checkpoint(model, path):
    """Atomically save a checkpoint so a timeout cannot corrupt it."""
    tmp = path.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    tmp.replace(path)


def stratified_internal_split(y, val_fraction, rng):
    """Return stratified train/validation indices."""
    real_idx = rng.permutation(np.flatnonzero(y == 0))
    ai_idx = rng.permutation(np.flatnonzero(y == 1))

    n_real_val = max(1, int(round(val_fraction * len(real_idx))))
    n_ai_val = max(1, int(round(val_fraction * len(ai_idx))))

    val_idx = np.concatenate([real_idx[:n_real_val], ai_idx[:n_ai_val]])
    tr_idx = np.concatenate([real_idx[n_real_val:], ai_idx[n_ai_val:]])
    return rng.permutation(tr_idx), rng.permutation(val_idx)


def make_balanced_cycle(real_idx, ai_idx, rng):
    """Use every real image once and sample the same number of AI images."""
    ai_sample = rng.choice(
        ai_idx,
        size=len(real_idx),
        replace=len(ai_idx) < len(real_idx),
    )
    return rng.permutation(np.concatenate([real_idx, ai_sample]))


def augment_image(img, rng):
    """Apply one realistic robustness transform to a uint8 HWC image."""
    choice = int(rng.integers(4))

    if choice == 0:
        # True JPEG round-trip, matching the original robust baseline.
        buffer = io.BytesIO()
        Image.fromarray(img).save(
            buffer,
            format="JPEG",
            quality=int(rng.integers(30, 91)),
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)

    if choice == 1:
        radius = float(rng.uniform(0.5, 1.5))
        blurred = Image.fromarray(img).filter(
            ImageFilter.GaussianBlur(radius)
        )
        return np.asarray(blurred, dtype=np.uint8)

    if choice == 2:
        side = max(
            8,
            int(IMG_SIZE * float(rng.uniform(0.5, 0.9))),
        )
        pil = Image.fromarray(img)
        small = pil.resize((side, side), Image.BILINEAR)
        restored = small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.asarray(restored, dtype=np.uint8)

    noise = rng.normal(
        0.0,
        float(rng.uniform(3.0, 10.0)),
        img.shape,
    )
    return np.clip(
        img.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)


def augment_uint8_batch(images, rng, probability=AUG_PROB):
    """Apply the realistic transforms independently to selected images."""
    out = np.ascontiguousarray(images).copy()
    active = rng.random(len(out)) < probability

    for i in np.flatnonzero(active):
        out[i] = augment_image(out[i], rng)

    return out


def make_fixed_augmented_view(images, rng, batch_size=BATCH):
    """Create one deterministic realistic augmented view for validation."""
    chunks = []
    for start in range(0, len(images), batch_size):
        chunk = np.ascontiguousarray(images[start:start + batch_size])
        chunks.append(
            augment_uint8_batch(
                chunk,
                rng,
                probability=1.0,
            )
        )
    return np.concatenate(chunks, axis=0)


def robust_internal_validation(model, clean_images, augmented_images, labels):
    """Evaluate one threshold that respects the FPR target on both views."""
    clean_scores = score_images(model, clean_images)
    augmented_scores = score_images(model, augmented_images)

    clean_threshold = calibrate_threshold(
        clean_scores[labels == 0],
        target_fpr=INTERNAL_SELECTION_FPR,
    )
    augmented_threshold = calibrate_threshold(
        augmented_scores[labels == 0],
        target_fpr=INTERNAL_SELECTION_FPR,
    )

    # The larger threshold is the conservative choice: it protects both clean
    # and transformed real images on the internal holdout.
    threshold = max(clean_threshold, augmented_threshold)
    clean_metrics = evaluate(labels, clean_scores, threshold)
    augmented_metrics = evaluate(labels, augmented_scores, threshold)

    return {
        "threshold": threshold,
        "clean_threshold": clean_threshold,
        "augmented_threshold": augmented_threshold,
        "clean": clean_metrics,
        "augmented": augmented_metrics,
    }


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
    training_seconds = min(
        args.timeout_seconds - MARGIN_SECONDS,
        TRAIN_BUDGET_SECONDS,
    )
    deadline = start_time + max(1, training_seconds)

    print("=" * 70)
    print("Starting train_augmented.py — batch-driven robust fine-tuning")
    print(f"Timeout received: {args.timeout_seconds}s")
    print(f"Training budget: {training_seconds}s")
    print(f"Safety margin: {MARGIN_SECONDS}s")
    print(f"Batch size: {BATCH}")
    print(f"Fine-tuning LR: {FINE_TUNE_LR:.6f}")
    print(f"Augmentation probability: {AUG_PROB:.2f}")
    print(f"Internal validation fraction: {INTERNAL_VAL_FRACTION:.2f}")
    print(f"Validate every: {VALIDATE_EVERY_STEPS} optimizer steps")
    print(f"Internal selection FPR target: {INTERNAL_SELECTION_FPR:.2f}")
    print(f"Final calibration FPR target: {FINAL_CAL_TARGET_FPR:.2f}")
    print("=" * 70)

    models_dir = ARTIFACTS_DIR / "models"
    metrics_dir = ARTIFACTS_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    task2_model_path = models_dir / "model.pt"
    model_path = models_dir / "model_aug.pt"
    if not task2_model_path.exists():
        raise FileNotFoundError(
            f"Missing Task 2 checkpoint: {task2_model_path}. Run train.py first."
        )

    cache = np.load(
        ARTIFACTS_DIR / "prepared" / "train.npz",
        mmap_mode="r",
    )
    X = cache["X"]
    y = np.asarray(cache["y"], dtype=np.int64)

    print(f"Loaded dataset: {len(X)} images")
    print(f"Real images: {(y == 0).sum()}")
    print(f"AI images: {(y == 1).sum()}")

    rng = np.random.default_rng(SEED)
    tr_idx, val_idx = stratified_internal_split(
        y,
        INTERNAL_VAL_FRACTION,
        rng,
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

    # Keep the prepared cache as uint8 and convert only selected batches.
    yt = torch.from_numpy(y.astype(np.float32))

    internal_clean_X = np.ascontiguousarray(X[val_idx])
    internal_y = y[val_idx]

    print("Creating fixed augmented internal-validation view...")
    aug_view_start = time.perf_counter()
    internal_augmented_X = make_fixed_augmented_view(
        internal_clean_X,
        np.random.default_rng(SEED + 1),
    )
    print(
        "Fixed augmented view ready in "
        f"{time.perf_counter() - aug_view_start:.1f}s"
    )

    model = SmallCNN()
    model.load_state_dict(
        torch.load(task2_model_path, weights_only=True)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNE_LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_CYCLES,
        eta_min=FINE_TUNE_LR * 0.2,
    )
    criterion = nn.BCEWithLogitsLoss()

    # The Task 2 model is a valid fallback if no robust checkpoint is selected.
    save_checkpoint(model, model_path)

    best_augmented_recall = -1.0
    best_clean_recall = -1.0
    best_mean_auc = -1.0
    best_step = 0
    best_validation = None

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
        nonlocal best_augmented_recall, best_clean_recall, best_mean_auc
        nonlocal best_step, best_validation
        nonlocal total_validation_time, validation_estimate
        nonlocal last_validated_step

        validation_start = time.perf_counter()
        metrics = robust_internal_validation(
            model,
            internal_clean_X,
            internal_augmented_X,
            internal_y,
        )
        validation_duration = time.perf_counter() - validation_start

        total_validation_time += validation_duration
        validation_estimate = max(
            5.0,
            0.5 * validation_estimate + 0.5 * validation_duration,
        )
        last_validated_step = total_steps

        clean = metrics["clean"]
        augmented = metrics["augmented"]
        augmented_recall = augmented["recall"]
        clean_recall = clean["recall"]
        mean_auc = 0.5 * (clean["roc_auc"] + augmented["roc_auc"])

        # Primary objective: robust recall at a threshold that protects both
        # real-image distributions. Preserve clean recall as first tie-breaker.
        is_better = (
            augmented_recall > best_augmented_recall + 1e-12
            or (
                abs(augmented_recall - best_augmented_recall) <= 1e-12
                and clean_recall > best_clean_recall + 1e-12
            )
            or (
                abs(augmented_recall - best_augmented_recall) <= 1e-12
                and abs(clean_recall - best_clean_recall) <= 1e-12
                and mean_auc > best_mean_auc + 1e-12
            )
        )

        print(
            f"\nInternal robust validation ({reason}) at step {total_steps}:\n"
            f"  threshold={metrics['threshold']:.6f} "
            f"(clean={metrics['clean_threshold']:.6f}, "
            f"aug={metrics['augmented_threshold']:.6f})\n"
            f"  clean: recall={clean['recall']:.4f} "
            f"fpr={clean['fpr']:.4f} auc={clean['roc_auc']:.4f}\n"
            f"  aug  : recall={augmented['recall']:.4f} "
            f"fpr={augmented['fpr']:.4f} "
            f"auc={augmented['roc_auc']:.4f}\n"
            f"  validation_time={validation_duration:.1f}s"
        )

        if is_better:
            print(
                "New best robust checkpoint: "
                f"aug recall {best_augmented_recall:.4f} -> "
                f"{augmented_recall:.4f}, "
                f"clean recall {best_clean_recall:.4f} -> "
                f"{clean_recall:.4f}"
            )
            best_augmented_recall = augmented_recall
            best_clean_recall = clean_recall
            best_mean_auc = mean_auc
            best_step = total_steps
            best_validation = metrics
            save_checkpoint(model, model_path)
        else:
            print(
                "Checkpoint unchanged: "
                f"best aug recall={best_augmented_recall:.4f}, "
                f"best clean recall={best_clean_recall:.4f}, "
                f"best step={best_step}"
            )

    print("\nStarting robust optimizer-step loop...")

    while True:
        remaining = deadline - time.perf_counter()
        reserve = validation_estimate + VALIDATION_SAFETY_SECONDS
        if remaining <= reserve:
            print(
                f"\nStopping optimizer loop with {remaining:.1f}s remaining "
                f"to reserve about {reserve:.1f}s for final robust validation."
            )
            break

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

        xb_np = augment_uint8_batch(
            np.ascontiguousarray(X[idx_np]),
            rng,
            probability=AUG_PROB,
        )
        xb = (
            torch.from_numpy(xb_np)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
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
            seconds_per_batch = total_training_time / total_steps
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
                f"sec/batch={seconds_per_batch:.2f} "
                f"img/s={images_per_second:.1f}"
            )

        if total_steps % VALIDATE_EVERY_STEPS == 0:
            remaining = deadline - time.perf_counter()
            required = 2.0 * validation_estimate + VALIDATION_SAFETY_SECONDS
            if remaining > required:
                run_and_maybe_save_validation("periodic")
            else:
                print(
                    f"Skipping periodic validation at step {total_steps}: "
                    f"{remaining:.1f}s remain, reserving final validation."
                )

    if total_steps > last_validated_step:
        run_and_maybe_save_validation("final")
    elif best_step == 0:
        run_and_maybe_save_validation("fallback")

    print("\nRobust fine-tuning finished.")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Total images seen: {total_images_seen}")
    print(f"Completed balanced cycles: {completed_cycles}")
    print(f"Training compute time: {total_training_time:.1f}s")
    print(f"Internal validation time: {total_validation_time:.1f}s")
    print(f"Best checkpoint step: {best_step}")
    print(f"Best internal augmented recall: {best_augmented_recall:.4f}")
    print(f"Best internal clean recall: {best_clean_recall:.4f}")
    print(f"Best internal mean AUC: {best_mean_auc:.4f}")

    print("Loading best robust checkpoint...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Calibrate one conservative threshold using both official calibration
    # distributions. The augmented model must not accuse too many real images
    # in either clean or transformed conditions.
    print("Calibrating threshold on clean and augmented calibration splits...")
    cal_clean_X, cal_clean_y = load_split("calibration")
    cal_aug_X, cal_aug_y = load_split("calibration_augmented")

    cal_clean_scores = score_images(model, cal_clean_X)
    cal_aug_scores = score_images(model, cal_aug_X)

    threshold_clean = calibrate_threshold(
        cal_clean_scores[cal_clean_y == 0],
        target_fpr=FINAL_CAL_TARGET_FPR,
    )
    threshold_augmented = calibrate_threshold(
        cal_aug_scores[cal_aug_y == 0],
        target_fpr=FINAL_CAL_TARGET_FPR,
    )
    threshold = max(threshold_clean, threshold_augmented)

    with open(models_dir / "threshold_aug.json", "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "threshold_clean": threshold_clean,
                "threshold_augmented": threshold_augmented,
                "calibration_target_fpr": FINAL_CAL_TARGET_FPR,
            },
            f,
            indent=2,
        )

    results = {
        "threshold": threshold,
        "threshold_clean": threshold_clean,
        "threshold_augmented": threshold_augmented,
        "best_step": best_step,
        "total_optimizer_steps": total_steps,
        "total_images_seen": total_images_seen,
        "completed_balanced_cycles": completed_cycles,
        "best_internal_augmented_recall": best_augmented_recall,
        "best_internal_clean_recall": best_clean_recall,
        "best_internal_mean_auc": best_mean_auc,
        "training_compute_seconds": total_training_time,
        "internal_validation_seconds": total_validation_time,
        "internal_best": best_validation,
    }

    validation_scores = {}
    validation_labels = {}
    for split in ("validation", "validation_augmented"):
        sx, sy = load_split(split)
        scores = score_images(model, sx)
        validation_scores[split] = scores
        validation_labels[split] = sy
        results[split] = evaluate(sy, scores, threshold)

    # Diagnostic only: threshold chosen directly on validation to show the
    # model's potential at the exact 20% FPR limit. It is never saved for use
    # by predict_augmented.py.
    oracle_clean = calibrate_threshold(
        validation_scores["validation"][validation_labels["validation"] == 0],
        target_fpr=TARGET_FPR,
    )
    oracle_augmented = calibrate_threshold(
        validation_scores["validation_augmented"][
            validation_labels["validation_augmented"] == 0
        ],
        target_fpr=TARGET_FPR,
    )
    oracle_threshold = max(oracle_clean, oracle_augmented)
    results["validation_oracle_diagnostic"] = {
        "threshold": oracle_threshold,
        "threshold_clean": oracle_clean,
        "threshold_augmented": oracle_augmented,
        "validation": evaluate(
            validation_labels["validation"],
            validation_scores["validation"],
            oracle_threshold,
        ),
        "validation_augmented": evaluate(
            validation_labels["validation_augmented"],
            validation_scores["validation_augmented"],
            oracle_threshold,
        ),
    }

    with open(metrics_dir / "task03.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"threshold={threshold:.6f} "
        f"(clean={threshold_clean:.6f}, "
        f"augmented={threshold_augmented:.6f})"
    )

    for split, target in (("validation", 0.8),
                          ("validation_augmented", 0.6)):
        m = results[split]
        fpr_ok = "OK" if m["fpr"] <= TARGET_FPR else "VIOLATED"
        print(
            f"{split}: recall_ai={m['recall']:.4f} "
            f"(target >= {target:.1f}) "
            f"fpr={m['fpr']:.4f} "
            f"({fpr_ok}, constraint <= {TARGET_FPR:.2f}) "
            f"roc_auc={m['roc_auc']:.4f}"
        )

    oracle = results["validation_oracle_diagnostic"]
    print(
        f"validation diagnostic at exact {TARGET_FPR:.0%} target on both views:"
    )
    for split in ("validation", "validation_augmented"):
        m = oracle[split]
        print(
            f"  {split}: recall={m['recall']:.4f} "
            f"fpr={m['fpr']:.4f}"
        )

    total_elapsed = time.perf_counter() - start_time
    print(f"Total train_augmented.py elapsed: {total_elapsed:.1f}s")
    print("=" * 70)
    print("FINAL TASK 3 RESULTS")
    print(f"Threshold: {threshold:.4f}")
    print(f"Best step: {best_step}/{total_steps}")
    print(
        f"Validation: recall={results['validation']['recall']:.4f} "
        f"fpr={results['validation']['fpr']:.4f}"
    )
    print(
        "Validation_aug: "
        f"recall={results['validation_augmented']['recall']:.4f} "
        f"fpr={results['validation_augmented']['fpr']:.4f}"
    )
    print(f"Total runtime: {total_elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
