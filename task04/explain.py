# task04/explain.py — Task 1.4 explainability for the final Task 3 model.
#
# Place this file at:
#   <repository>/task04/explain.py
#
# Run after the official pipeline has produced model_aug.pt and
# threshold_aug.json:
#
#   docker run --rm \
#     -v <repository>:/workspace/repo \
#     -w /workspace/repo \
#     amls-solution \
#     python task04/explain.py
#
# The script writes all derived outputs under:
#   solution/artifacts/task04/
#
# Methods:
#   1. Vanilla input-gradient saliency for TP, TN, FP, and FN examples.
#   2. Signed occlusion sensitivity on the most confident FP and FN examples.
#   3. Quantitative error analysis by original AI source class.
#   4. Mean normalized saliency comparison for real vs. AI images.
#
# Important interpretation note:
# Saliency measures local sensitivity, not causality. Occlusion results also
# depend on patch size and replacement value and may introduce artificial
# patterns. The generated explanations must therefore be discussed critically.

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = REPO_DIR / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from common import (  # noqa: E402
    ARTIFACTS_DIR,
    BATCH,
    DATA_DIR,
    IMG_SIZE,
    SEED,
    SmallCNN,
    load_split,
    set_seed,
)

OUT_DIR = ARTIFACTS_DIR / "task04"

N_EXAMPLES = 8
N_OCCLUSION = 4
N_MEAN = 100
PATCH = 8

SOURCE_NAMES = {
    0: "Real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "Midjourney",
}


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def score_logits(model, images, batch=BATCH):
    """Return model logits for uint8 images shaped (N, H, W, 3)."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = x[start:start + batch].float().div_(255.0)
            outputs.append(model(xb).detach().cpu())

    if not outputs:
        return np.empty(0, dtype=np.float32)

    return torch.cat(outputs).numpy()


def input_gradient_saliency(model, image):
    """Absolute input gradient |d(AI logit)/d(pixel)|, max over RGB."""
    x = (
        torch.from_numpy(np.ascontiguousarray(image[None]))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
    )
    x.requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)

    logit = model(x)[0]
    logit.backward()

    grad = x.grad.detach().abs().max(dim=1).values.squeeze(0)
    return grad.cpu().numpy().astype(np.float32)


def normalize_for_display(saliency_map):
    """Robustly scale one map to [0, 1] for qualitative display."""
    saliency_map = np.asarray(saliency_map, dtype=np.float32)
    scale = float(np.percentile(saliency_map, 99.0))

    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(saliency_map)

    return np.clip(saliency_map / scale, 0.0, 1.0)


def normalize_saliency_mass(saliency_map):
    """Normalize a map so its values sum to one for mean-attention analysis."""
    saliency_map = np.asarray(saliency_map, dtype=np.float64)
    total = float(saliency_map.sum())

    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(saliency_map, dtype=np.float32)

    return (saliency_map / total).astype(np.float32)


def center_mass(normalized_map):
    """Fraction of normalized saliency inside the central 50% of the image."""
    h, w = normalized_map.shape
    r0, r1 = h // 4, 3 * h // 4
    c0, c1 = w // 4, 3 * w // 4
    return float(normalized_map[r0:r1, c0:c1].sum())


def occlusion_map(model, image, base_logit):
    """Signed change in AI logit after replacing each patch.

    Positive values mean that hiding the region reduces AI evidence.
    Negative values mean that hiding the region increases AI evidence.
    """
    occluded = []
    positions = []

    fill = np.rint(image.mean(axis=(0, 1))).astype(np.uint8)

    for row in range(0, IMG_SIZE, PATCH):
        for col in range(0, IMG_SIZE, PATCH):
            patched = image.copy()
            patched[row:row + PATCH, col:col + PATCH] = fill
            occluded.append(patched)
            positions.append((row // PATCH, col // PATCH))

    occluded_logits = score_logits(model, np.stack(occluded))

    n_rows = (IMG_SIZE + PATCH - 1) // PATCH
    n_cols = (IMG_SIZE + PATCH - 1) // PATCH
    heat = np.zeros((n_rows, n_cols), dtype=np.float32)

    for (row, col), occluded_logit in zip(positions, occluded_logits):
        heat[row, col] = float(base_logit - occluded_logit)

    return heat


def source_classes(split):
    """Read original source_class values in the same order as load_split()."""
    values = []

    for parquet_path in sorted((DATA_DIR / split).glob("*.parquet")):
        try:
            column = pd.read_parquet(
                parquet_path, columns=["source_class"]
            )["source_class"]
        except Exception:
            column = pd.read_parquet(
                parquet_path, columns=["source class"]
            )["source class"]

        values.append(column.to_numpy(dtype=np.int64))

    if not values:
        raise FileNotFoundError(f"No parquet files found for split '{split}'")

    return np.concatenate(values)


def category_indices(y_true, y_pred, scores):
    """Return deterministic confidence-ranked TP, TN, FP, and FN indices."""
    masks = {
        "true_positive_ai": (y_true == 1) & (y_pred == 1),
        "true_negative_real": (y_true == 0) & (y_pred == 0),
        "false_positive_real_as_ai": (y_true == 0) & (y_pred == 1),
        "false_negative_ai_as_real": (y_true == 1) & (y_pred == 0),
    }

    result = {}

    for name, mask in masks.items():
        indices = np.where(mask)[0]

        if name in {"true_positive_ai", "false_positive_real_as_ai"}:
            order = np.argsort(-scores[indices])
        else:
            order = np.argsort(scores[indices])

        result[name] = indices[order]

    return result


def example_title(index, source_class, score, threshold, category):
    source_name = SOURCE_NAMES.get(int(source_class), f"class {int(source_class)}")
    return (
        f"idx={index} | {source_name}\n"
        f"score={score:.3f} | thr={threshold:.3f}\n"
        f"{category.replace('_', ' ')}"
    )


def saliency_grid(
    model,
    images,
    indices,
    source_values,
    scores,
    threshold,
    category,
    output_path,
):
    indices = np.asarray(indices, dtype=np.int64)

    if len(indices) == 0:
        print(f"skip {output_path.name}: no examples")
        return []

    maps = [
        normalize_for_display(input_gradient_saliency(model, images[index]))
        for index in indices
    ]

    fig, axes = plt.subplots(
        2,
        len(indices),
        figsize=(2.15 * len(indices), 4.8),
        squeeze=False,
    )

    records = []

    for col, (index, saliency_map) in enumerate(zip(indices, maps)):
        axes[0, col].imshow(images[index])
        axes[0, col].set_title(
            example_title(
                index,
                source_values[index],
                scores[index],
                threshold,
                category,
            ),
            fontsize=7,
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(images[index], alpha=0.55)
        axes[1, col].imshow(
            saliency_map,
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            alpha=0.70,
        )
        axes[1, col].axis("off")

        records.append(
            {
                "analysis": "saliency",
                "category": category,
                "rank": int(col + 1),
                "validation_index": int(index),
                "source_class": int(source_values[index]),
                "source_name": SOURCE_NAMES.get(
                    int(source_values[index]),
                    f"class {int(source_values[index])}",
                ),
                "true_label": int(source_values[index] != 0),
                "predicted_label": int(scores[index] > threshold),
                "score": float(scores[index]),
                "threshold": float(threshold),
            }
        )

    fig.suptitle(
        f"{category.replace('_', ' ').title()}: "
        "original image and normalized input-gradient overlay",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")

    return records


def occlusion_grid(
    model,
    images,
    indices,
    source_values,
    logits,
    scores,
    threshold,
    category,
    output_path,
):
    indices = np.asarray(indices, dtype=np.int64)

    if len(indices) == 0:
        print(f"skip {output_path.name}: no examples")
        return []

    maps = [
        occlusion_map(model, images[index], logits[index])
        for index in indices
    ]

    all_values = np.concatenate([heat.ravel() for heat in maps])
    vmax = float(np.percentile(np.abs(all_values), 99.0))

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        2,
        len(indices),
        figsize=(2.15 * len(indices), 4.8),
        squeeze=False,
    )

    records = []

    for col, (index, heat) in enumerate(zip(indices, maps)):
        axes[0, col].imshow(images[index])
        axes[0, col].set_title(
            example_title(
                index,
                source_values[index],
                scores[index],
                threshold,
                category,
            ),
            fontsize=7,
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(images[index], alpha=0.55)
        axes[1, col].imshow(
            heat,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.70,
            interpolation="nearest",
            extent=(0, IMG_SIZE, IMG_SIZE, 0),
        )
        axes[1, col].axis("off")

        records.append(
            {
                "analysis": "occlusion",
                "category": category,
                "rank": int(col + 1),
                "validation_index": int(index),
                "source_class": int(source_values[index]),
                "source_name": SOURCE_NAMES.get(
                    int(source_values[index]),
                    f"class {int(source_values[index])}",
                ),
                "true_label": int(source_values[index] != 0),
                "predicted_label": int(scores[index] > threshold),
                "score": float(scores[index]),
                "threshold": float(threshold),
                "base_logit": float(logits[index]),
                "max_positive_logit_drop": float(heat.max()),
                "most_negative_logit_change": float(heat.min()),
            }
        )

    fig.suptitle(
        f"{category.replace('_', ' ').title()}: "
        "signed occlusion sensitivity overlay\n"
        "red = region supporting AI; blue = region supporting real",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")

    return records


def sample_mean_saliency_indices(y_true, source_values, n_per_group, seed=SEED):
    """Random real sample and source-stratified AI sample."""
    rng = np.random.default_rng(seed)

    real_candidates = np.where(y_true == 0)[0]
    n_real = min(n_per_group, len(real_candidates))
    real_indices = rng.choice(real_candidates, size=n_real, replace=False)

    ai_classes = sorted(
        int(value)
        for value in np.unique(source_values)
        if int(value) != 0
    )

    selected_ai = []
    used = set()

    if ai_classes:
        base = n_per_group // len(ai_classes)
        remainder = n_per_group % len(ai_classes)

        for class_position, source_class in enumerate(ai_classes):
            quota = base + int(class_position < remainder)
            candidates = np.where(source_values == source_class)[0]
            take = min(quota, len(candidates))

            if take > 0:
                chosen = rng.choice(candidates, size=take, replace=False)
                selected_ai.extend(int(index) for index in chosen)
                used.update(int(index) for index in chosen)

    remaining = n_per_group - len(selected_ai)

    if remaining > 0:
        all_ai = np.where(y_true == 1)[0]
        unused_ai = np.array(
            [index for index in all_ai if int(index) not in used],
            dtype=np.int64,
        )
        take = min(remaining, len(unused_ai))

        if take > 0:
            selected_ai.extend(
                int(index)
                for index in rng.choice(unused_ai, size=take, replace=False)
            )

    return (
        np.asarray(real_indices, dtype=np.int64),
        np.asarray(selected_ai, dtype=np.int64),
    )


def mean_saliency_analysis(model, images, y_true, source_values, output_path):
    real_indices, ai_indices = sample_mean_saliency_indices(
        y_true,
        source_values,
        N_MEAN,
    )

    real_maps = [
        normalize_saliency_mass(input_gradient_saliency(model, images[index]))
        for index in real_indices
    ]
    ai_maps = [
        normalize_saliency_mass(input_gradient_saliency(model, images[index]))
        for index in ai_indices
    ]

    if not real_maps or not ai_maps:
        print("skip mean saliency comparison: one class has no examples")
        return {}

    mean_real = np.mean(real_maps, axis=0)
    mean_ai = np.mean(ai_maps, axis=0)

    shared_vmax = float(
        np.percentile(
            np.concatenate([mean_real.ravel(), mean_ai.ravel()]),
            99.0,
        )
    )

    if not np.isfinite(shared_vmax) or shared_vmax <= 0.0:
        shared_vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    for axis, mean_map, title in (
        (
            axes[0],
            mean_real,
            f"Real: mean normalized saliency (n={len(real_indices)})",
        ),
        (
            axes[1],
            mean_ai,
            f"AI: mean normalized saliency (n={len(ai_indices)})",
        ),
    ):
        image = axis.imshow(
            mean_map,
            cmap="inferno",
            vmin=0.0,
            vmax=shared_vmax,
        )
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Real vs. AI spatial distribution of input-gradient saliency\n"
        "(each image normalized to unit saliency mass before averaging)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")

    real_center = [center_mass(value) for value in real_maps]
    ai_center = [center_mass(value) for value in ai_maps]

    ai_source_counts = {
        SOURCE_NAMES.get(int(source_class), str(int(source_class))): int(
            (source_values[ai_indices] == source_class).sum()
        )
        for source_class in sorted(np.unique(source_values[ai_indices]))
    }

    return {
        "n_real": int(len(real_indices)),
        "n_ai": int(len(ai_indices)),
        "ai_source_counts": ai_source_counts,
        "center_saliency_mass_real_mean": float(np.mean(real_center)),
        "center_saliency_mass_real_std": float(np.std(real_center)),
        "center_saliency_mass_ai_mean": float(np.mean(ai_center)),
        "center_saliency_mass_ai_std": float(np.std(ai_center)),
        "definition": (
            "Center mass is the fraction of per-image normalized absolute "
            "input-gradient saliency inside the central 50% of the image."
        ),
    }


def split_summary(y_true, scores, source_values, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.int64)
    y_pred = (scores > threshold).astype(np.int64)

    if not (len(y_true) == len(scores) == len(source_values)):
        raise ValueError("Labels, scores, and source classes have different lengths")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    per_source = {}

    for source_class in sorted(np.unique(source_values)):
        mask = source_values == source_class
        class_scores = scores[mask]
        class_predictions = y_pred[mask]
        source_class = int(source_class)

        entry = {
            "source_class": source_class,
            "source_name": SOURCE_NAMES.get(
                source_class,
                f"class {source_class}",
            ),
            "count": int(mask.sum()),
            "mean_score": float(class_scores.mean()),
            "median_score": float(np.median(class_scores)),
        }

        if source_class == 0:
            entry["false_positive_rate"] = float(
                (class_predictions == 1).mean()
            )
        else:
            entry["recall"] = float((class_predictions == 1).mean())
            entry["false_negatives"] = int(
                (class_predictions == 0).sum()
            )

        per_source[str(source_class)] = entry

    real_mask = y_true == 0
    ai_mask = y_true == 1

    return {
        "counts": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
        "fpr_real": float((y_pred[real_mask] == 1).mean()),
        "recall_ai": float((y_pred[ai_mask] == 1).mean()),
        "mean_score_real": float(scores[real_mask].mean()),
        "median_score_real": float(np.median(scores[real_mask])),
        "mean_score_ai": float(scores[ai_mask].mean()),
        "median_score_ai": float(np.median(scores[ai_mask])),
        "per_source_class": per_source,
    }


def source_robustness_table(clean_summary, augmented_summary):
    rows = []

    clean_sources = clean_summary["per_source_class"]
    augmented_sources = augmented_summary["per_source_class"]

    common_ids = sorted(
        set(clean_sources).intersection(augmented_sources),
        key=int,
    )

    for source_id in common_ids:
        clean_entry = clean_sources[source_id]
        augmented_entry = augmented_sources[source_id]

        row = {
            "source_class": int(source_id),
            "source_name": clean_entry["source_name"],
            "clean_count": int(clean_entry["count"]),
            "augmented_count": int(augmented_entry["count"]),
            "clean_mean_score": float(clean_entry["mean_score"]),
            "augmented_mean_score": float(augmented_entry["mean_score"]),
        }

        if int(source_id) == 0:
            row["clean_fpr"] = float(clean_entry["false_positive_rate"])
            row["augmented_fpr"] = float(
                augmented_entry["false_positive_rate"]
            )
            row["fpr_change_augmented_minus_clean"] = float(
                augmented_entry["false_positive_rate"]
                - clean_entry["false_positive_rate"]
            )
        else:
            row["clean_recall"] = float(clean_entry["recall"])
            row["augmented_recall"] = float(augmented_entry["recall"])
            row["recall_change_augmented_minus_clean"] = float(
                augmented_entry["recall"] - clean_entry["recall"]
            )

        rows.append(row)

    return rows


def validate_required_files():
    required = [
        ARTIFACTS_DIR / "models" / "model_aug.pt",
        ARTIFACTS_DIR / "models" / "threshold_aug.json",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Task 4 requires the completed Task 3 artifacts. Missing:\n"
            f"{formatted}"
        )


def main():
    set_seed()
    validate_required_files()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "models" / "model_aug.pt"
    threshold_path = ARTIFACTS_DIR / "models" / "threshold_aug.json"

    model = SmallCNN()
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with threshold_path.open(encoding="utf-8") as file:
        threshold_metadata = json.load(file)

    if "threshold" not in threshold_metadata:
        raise KeyError(f"'threshold' is missing from {threshold_path}")

    threshold = float(threshold_metadata["threshold"])

    images, labels = load_split("validation")
    sources = source_classes("validation")

    if len(images) != len(sources):
        raise ValueError(
            "Validation images and source-class metadata have different lengths"
        )

    logits = score_logits(model, images)
    scores = sigmoid_np(logits)
    predictions = (scores > threshold).astype(np.int64)

    categories = category_indices(labels, predictions, scores)

    selected_records = []

    for category, indices in categories.items():
        chosen = indices[:N_EXAMPLES]
        selected_records.extend(
            saliency_grid(
                model=model,
                images=images,
                indices=chosen,
                source_values=sources,
                scores=scores,
                threshold=threshold,
                category=category,
                output_path=OUT_DIR / f"saliency_{category}.png",
            )
        )

    for category in (
        "false_positive_real_as_ai",
        "false_negative_ai_as_real",
    ):
        chosen = categories[category][:N_OCCLUSION]
        selected_records.extend(
            occlusion_grid(
                model=model,
                images=images,
                indices=chosen,
                source_values=sources,
                logits=logits,
                scores=scores,
                threshold=threshold,
                category=category,
                output_path=OUT_DIR / f"occlusion_{category}.png",
            )
        )

    mean_saliency = mean_saliency_analysis(
        model=model,
        images=images,
        y_true=labels,
        source_values=sources,
        output_path=OUT_DIR / "mean_saliency_real_vs_ai.png",
    )

    clean_summary = split_summary(
        labels,
        scores,
        sources,
        threshold,
    )

    augmented_images, augmented_labels = load_split(
        "validation_augmented"
    )
    augmented_sources = source_classes("validation_augmented")

    if len(augmented_images) != len(augmented_sources):
        raise ValueError(
            "Augmented validation images and source-class metadata have "
            "different lengths"
        )

    augmented_logits = score_logits(model, augmented_images)
    augmented_scores = sigmoid_np(augmented_logits)

    augmented_summary = split_summary(
        augmented_labels,
        augmented_scores,
        augmented_sources,
        threshold,
    )

    robustness_rows = source_robustness_table(
        clean_summary,
        augmented_summary,
    )

    summary = {
        "model": str(model_path.relative_to(SOLUTION_DIR)),
        "threshold": threshold,
        "threshold_metadata": threshold_metadata,
        "methods": {
            "saliency": (
                "Absolute gradient of the AI logit with respect to input "
                "pixels, reduced by maximum over RGB channels."
            ),
            "occlusion": (
                f"Signed AI-logit change after replacing {PATCH}x{PATCH} "
                "patches with the per-image RGB mean."
            ),
            "mean_saliency": (
                "Each absolute-gradient map is normalized to unit mass before "
                "class averaging; the AI sample is stratified by source class."
            ),
        },
        "limitations": [
            "Input-gradient saliency measures local sensitivity, not causality.",
            "Absolute gradients show location but not whether evidence points "
            "toward real or AI.",
            "Gradient maps can emphasize edges because of the convolutional "
            "architecture, even when those edges are not semantically causal.",
            "Occlusion introduces artificial inputs and depends on patch size, "
            "stride, and replacement value.",
            "Neither method proves the absence of diffuse dataset shortcuts "
            "such as compression, resampling, or texture statistics.",
        ],
        "validation": clean_summary,
        "validation_augmented": augmented_summary,
        "mean_saliency_comparison": mean_saliency,
        "source_class_robustness": robustness_rows,
    }

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"wrote {OUT_DIR / 'summary.json'}")

    pd.DataFrame(robustness_rows).to_csv(
        OUT_DIR / "source_class_robustness.csv",
        index=False,
    )
    print(f"wrote {OUT_DIR / 'source_class_robustness.csv'}")

    selected_columns = [
        "analysis",
        "category",
        "rank",
        "validation_index",
        "source_class",
        "source_name",
        "true_label",
        "predicted_label",
        "score",
        "threshold",
        "base_logit",
        "max_positive_logit_drop",
        "most_negative_logit_change",
    ]
    selected_frame = pd.DataFrame(selected_records).reindex(
        columns=selected_columns
    )
    selected_frame.to_csv(
        OUT_DIR / "selected_examples.csv",
        index=False,
    )
    print(f"wrote {OUT_DIR / 'selected_examples.csv'}")

    print("\nTask 4 summary")
    print(
        "validation: "
        f"recall_ai={clean_summary['recall_ai']:.4f} "
        f"fpr={clean_summary['fpr_real']:.4f}"
    )
    print(
        "validation_augmented: "
        f"recall_ai={augmented_summary['recall_ai']:.4f} "
        f"fpr={augmented_summary['fpr_real']:.4f}"
    )
    print(f"all outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()