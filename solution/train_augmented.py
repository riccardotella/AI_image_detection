"""Task 3: robust fine-tuning with realistic image transformations."""

import argparse
import io
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
    balanced_cycle,
    calibrate_threshold,
    evaluate,
    load_split,
    save_json,
    save_model,
    score_images,
    set_seed,
    stratified_split,
)

MARGIN_SECONDS = 240
VALIDATION_FRACTION = 0.10
VALIDATE_EVERY = 100
SCHEDULER_CYCLES = 18
AUGMENTATION_PROBABILITY = 0.50
FINAL_CALIBRATION_FPR = 0.18
FINE_TUNE_LR = LR / 4
INITIAL_VALIDATION_SECONDS = 15.0


def augment_image(image, rng):
    """Apply JPEG, blur, rescaling or Gaussian noise."""
    choice = int(rng.integers(4))

    if choice == 0:
        buffer = io.BytesIO()
        Image.fromarray(image).save(
            buffer, format="JPEG", quality=int(rng.integers(30, 91))
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)

    if choice == 1:
        blurred = Image.fromarray(image).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5)))
        )
        return np.asarray(blurred, dtype=np.uint8)

    if choice == 2:
        side = max(8, int(IMG_SIZE * float(rng.uniform(0.5, 0.9))))
        pil = Image.fromarray(image)
        small = pil.resize((side, side), Image.Resampling.BILINEAR)
        restored = small.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(restored, dtype=np.uint8)

    noise = rng.normal(0.0, float(rng.uniform(3.0, 10.0)), image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_batch(images, rng, probability=AUGMENTATION_PROBABILITY):
    output = np.ascontiguousarray(images).copy()
    for index in np.flatnonzero(rng.random(len(output)) < probability):
        output[index] = augment_image(output[index], rng)
    return output


def fixed_augmented_view(images):
    rng = np.random.default_rng(SEED + 1)
    return np.concatenate(
        [
            augment_batch(images[start:start + BATCH], rng, probability=1.0)
            for start in range(0, len(images), BATCH)
        ]
    )


def validate(model, clean_images, augmented_images, labels):
    clean_scores = score_images(model, clean_images)
    augmented_scores = score_images(model, augmented_images)
    clean_threshold = calibrate_threshold(
        clean_scores[labels == 0], CAL_TARGET_FPR
    )
    augmented_threshold = calibrate_threshold(
        augmented_scores[labels == 0], CAL_TARGET_FPR
    )
    threshold = max(clean_threshold, augmented_threshold)
    return {
        "threshold": threshold,
        "clean": evaluate(labels, clean_scores, threshold),
        "augmented": evaluate(labels, augmented_scores, threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout_seconds <= MARGIN_SECONDS:
        raise ValueError(f"timeout must exceed {MARGIN_SECONDS}s")

    set_seed()
    started = time.perf_counter()
    budget = min(args.timeout_seconds - MARGIN_SECONDS, TRAIN_BUDGET_SECONDS)
    deadline = started + budget

    task2_path = ARTIFACTS_DIR / "models" / "model.pt"
    model_path = ARTIFACTS_DIR / "models" / "model_aug.pt"
    threshold_path = ARTIFACTS_DIR / "models" / "threshold_aug.json"
    metrics_path = ARTIFACTS_DIR / "metrics" / "task03.json"
    if not task2_path.exists():
        raise FileNotFoundError(f"Run train.py first: {task2_path} is missing")

    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz", mmap_mode="r")
    images = cache["X"]
    labels = np.asarray(cache["y"], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    train_indices, validation_indices = stratified_split(
        labels, VALIDATION_FRACTION, rng
    )
    real_indices = train_indices[labels[train_indices] == 0]
    ai_indices = train_indices[labels[train_indices] == 1]
    if not len(real_indices) or not len(ai_indices):
        raise RuntimeError("Both classes are required")

    tensor_labels = torch.from_numpy(labels.astype(np.float32))
    clean_validation = np.ascontiguousarray(images[validation_indices])
    validation_labels = labels[validation_indices]
    augmented_validation = fixed_augmented_view(clean_validation)

    model = SmallCNN()
    model.load_state_dict(
        torch.load(task2_path, map_location="cpu", weights_only=True)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_CYCLES, eta_min=FINE_TUNE_LR * 0.2
    )
    criterion = nn.BCEWithLogitsLoss()
    save_model(model, model_path)

    best = {
        "augmented_recall": -1.0,
        "clean_recall": -1.0,
        "mean_auc": -1.0,
        "step": 0,
    }
    steps = images_seen = completed_cycles = 0
    training_time = validation_time = 0.0
    validation_estimate = INITIAL_VALIDATION_SECONDS
    last_validated_step = 0

    order = balanced_cycle(real_indices, ai_indices, rng)
    cursor = 0

    def run_validation(reason):
        nonlocal validation_time, validation_estimate, last_validated_step, best

        start = time.perf_counter()
        metrics = validate(
            model, clean_validation, augmented_validation, validation_labels
        )
        duration = time.perf_counter() - start
        validation_time += duration
        validation_estimate = max(5.0, 0.5 * (validation_estimate + duration))
        last_validated_step = steps

        clean = metrics["clean"]
        augmented = metrics["augmented"]
        mean_auc = 0.5 * (clean["roc_auc"] + augmented["roc_auc"])
        better = (
            augmented["recall"] > best["augmented_recall"] + 1e-12
            or (
                abs(augmented["recall"] - best["augmented_recall"]) <= 1e-12
                and clean["recall"] > best["clean_recall"] + 1e-12
            )
            or (
                abs(augmented["recall"] - best["augmented_recall"]) <= 1e-12
                and abs(clean["recall"] - best["clean_recall"]) <= 1e-12
                and mean_auc > best["mean_auc"] + 1e-12
            )
        )
        if better:
            best = {
                "augmented_recall": augmented["recall"],
                "clean_recall": clean["recall"],
                "mean_auc": mean_auc,
                "step": steps,
            }
            save_model(model, model_path)

        print(
            f"validation {reason} step={steps}: "
            f"clean_recall={clean['recall']:.4f} clean_fpr={clean['fpr']:.4f} "
            f"aug_recall={augmented['recall']:.4f} aug_fpr={augmented['fpr']:.4f} "
            f"{'saved' if better else 'unchanged'}"
        )

    print(
        f"Task 3: {len(images)} images, batch={BATCH}, budget={budget}s, "
        f"augmentation_probability={AUGMENTATION_PROBABILITY:.2f}"
    )

    while deadline - time.perf_counter() > validation_estimate + 5.0:
        if cursor >= len(order):
            completed_cycles += 1
            if completed_cycles <= SCHEDULER_CYCLES:
                scheduler.step()
            order = balanced_cycle(real_indices, ai_indices, rng)
            cursor = 0

        batch_indices = order[cursor:cursor + BATCH]
        cursor += len(batch_indices)
        index_tensor = torch.from_numpy(batch_indices)

        batch_start = time.perf_counter()
        augmented = augment_batch(
            np.ascontiguousarray(images[batch_indices]), rng
        )
        batch_images = (
            torch.from_numpy(augmented)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        batch_labels = tensor_labels[index_tensor]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch_images), batch_labels)
        loss.backward()
        optimizer.step()
        training_time += time.perf_counter() - batch_start

        steps += 1
        images_seen += len(batch_indices)
        if steps == 1 or steps % 100 == 0:
            rate = images_seen / training_time if training_time else 0.0
            print(
                f"step={steps} loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} img/s={rate:.1f}"
            )

        if steps % VALIDATE_EVERY == 0:
            remaining = deadline - time.perf_counter()
            if remaining > 2 * validation_estimate + 5.0:
                run_validation("periodic")

    if steps > last_validated_step or best["step"] == 0:
        run_validation("final")

    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    clean_calibration, clean_labels = load_split("calibration")
    augmented_calibration, augmented_labels = load_split("calibration_augmented")
    clean_scores = score_images(model, clean_calibration)
    augmented_scores = score_images(model, augmented_calibration)

    clean_threshold = calibrate_threshold(
        clean_scores[clean_labels == 0], FINAL_CALIBRATION_FPR
    )
    augmented_threshold = calibrate_threshold(
        augmented_scores[augmented_labels == 0], FINAL_CALIBRATION_FPR
    )
    threshold = max(clean_threshold, augmented_threshold)
    save_json(
        {
            "threshold": threshold,
            "threshold_clean": clean_threshold,
            "threshold_augmented": augmented_threshold,
            "target_fpr": FINAL_CALIBRATION_FPR,
        },
        threshold_path,
    )

    results = {
        "threshold": threshold,
        "threshold_clean": clean_threshold,
        "threshold_augmented": augmented_threshold,
        "training": {
            "steps": steps,
            "images_seen": images_seen,
            "completed_balanced_cycles": completed_cycles,
            "best_step": best["step"],
            "best_internal_augmented_recall": best["augmented_recall"],
            "best_internal_clean_recall": best["clean_recall"],
            "best_internal_mean_auc": best["mean_auc"],
            "training_seconds": training_time,
            "internal_validation_seconds": validation_time,
        },
    }
    for split in ("validation", "validation_augmented"):
        split_images, split_labels = load_split(split)
        results[split] = evaluate(
            split_labels, score_images(model, split_images), threshold
        )

    results["elapsed_seconds"] = time.perf_counter() - started
    save_json(results, metrics_path)

    for split in ("validation", "validation_augmented"):
        metrics = results[split]
        print(
            f"{split}: recall={metrics['recall']:.4f} "
            f"fpr={metrics['fpr']:.4f} auc={metrics['roc_auc']:.4f}"
        )
    if any(results[split]["fpr"] > TARGET_FPR for split in results if split.startswith("validation")):
        print(f"WARNING: an FPR exceeds {TARGET_FPR:.2f}")
    print(
        f"Task 3 complete: best_step={best['step']}/{steps}, "
        f"threshold={threshold:.6f}, elapsed={results['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
