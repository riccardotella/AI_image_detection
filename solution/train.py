"""Task 2: deadline-driven training, calibration and validation."""

import argparse
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
INITIAL_VALIDATION_SECONDS = 45.0


def validate(model, images, labels):
    scores = score_images(model, images)
    threshold = calibrate_threshold(scores[labels == 0], CAL_TARGET_FPR)
    return threshold, evaluate(labels, scores, threshold)


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

    model_path = ARTIFACTS_DIR / "models" / "model.pt"
    threshold_path = ARTIFACTS_DIR / "models" / "threshold.json"
    metrics_path = ARTIFACTS_DIR / "metrics" / "task02.json"

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

    tensor_images = torch.from_numpy(np.asarray(images)).permute(0, 3, 1, 2)
    tensor_labels = torch.from_numpy(labels.astype(np.float32))
    validation_images = np.ascontiguousarray(images[validation_indices])
    validation_labels = labels[validation_indices]

    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SCHEDULER_CYCLES, eta_min=LR * 0.1
    )
    criterion = nn.BCEWithLogitsLoss()
    save_model(model, model_path)

    best = {"recall": -1.0, "auc": -1.0, "fpr": None, "step": 0}
    steps = images_seen = completed_cycles = 0
    training_time = validation_time = 0.0
    validation_estimate = INITIAL_VALIDATION_SECONDS
    last_validated_step = 0

    order = balanced_cycle(real_indices, ai_indices, rng)
    cursor = 0

    def run_validation(reason):
        nonlocal validation_time, validation_estimate, last_validated_step, best

        start = time.perf_counter()
        threshold, metrics = validate(model, validation_images, validation_labels)
        duration = time.perf_counter() - start
        validation_time += duration
        validation_estimate = max(10.0, 0.5 * (validation_estimate + duration))
        last_validated_step = steps

        better = metrics["recall"] > best["recall"] + 1e-12 or (
            abs(metrics["recall"] - best["recall"]) <= 1e-12
            and metrics["roc_auc"] > best["auc"] + 1e-12
        )
        if better:
            best = {
                "recall": metrics["recall"],
                "auc": metrics["roc_auc"],
                "fpr": metrics["fpr"],
                "step": steps,
            }
            save_model(model, model_path)

        print(
            f"validation {reason} step={steps}: recall={metrics['recall']:.4f} "
            f"fpr={metrics['fpr']:.4f} auc={metrics['roc_auc']:.4f} "
            f"{'saved' if better else 'unchanged'}"
        )

    print(
        f"Task 2: {len(images)} images, batch={BATCH}, budget={budget}s, "
        f"train={len(train_indices)}, internal_validation={len(validation_indices)}"
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
        batch_images = tensor_images[index_tensor].float().div_(255.0)
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
    calibration_images, calibration_labels = load_split("calibration")
    real_scores = score_images(
        model, calibration_images[calibration_labels == 0]
    )
    threshold = calibrate_threshold(real_scores)
    save_json({"threshold": threshold}, threshold_path)

    results = {
        "threshold": threshold,
        "training": {
            "steps": steps,
            "images_seen": images_seen,
            "completed_balanced_cycles": completed_cycles,
            "best_step": best["step"],
            "best_internal_recall": best["recall"],
            "best_internal_fpr": best["fpr"],
            "best_internal_auc": best["auc"],
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
    if results["validation"]["fpr"] > TARGET_FPR:
        print(f"WARNING: validation FPR exceeds {TARGET_FPR:.2f}")
    print(
        f"Task 2 complete: best_step={best['step']}/{steps}, "
        f"threshold={threshold:.6f}, elapsed={results['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
