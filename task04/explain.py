"""Task 1.4: saliency, occlusion and error analysis for the Task 3 model."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "solution"))

from common import (  # noqa: E402
    ARTIFACTS_DIR,
    BATCH,
    IMG_SIZE,
    SEED,
    SmallCNN,
    load_split,
    save_json,
    set_seed,
)

OUTPUT_DIR = ARTIFACTS_DIR / "task04"
PATCH_SIZE = 8
EXAMPLES_PER_CATEGORY = 2
MEAN_EXAMPLES_PER_CLASS = 100
SOURCE_NAMES = {
    0: "Real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "Midjourney",
}
CATEGORIES = (
    ("true_positive", "TP: AI correctly detected"),
    ("true_negative", "TN: real correctly detected"),
    ("false_positive", "FP: real predicted as AI"),
    ("false_negative", "FN: AI predicted as real"),
)


def logits(model, images, batch_size=BATCH):
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size].float().div_(255.0)
            output.append(model(batch).cpu())
    return torch.cat(output).numpy()


def saliency(model, image):
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image[None]))
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
    )
    tensor.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(tensor)[0].backward()
    return tensor.grad.abs().max(dim=1).values[0].detach().numpy()


def display_map(values):
    scale = float(np.percentile(values, 99))
    return np.clip(values / scale, 0, 1) if scale > 0 else np.zeros_like(values)


def mass_map(values):
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def occlusion_map(model, image, base_logit):
    patched_images = []
    positions = []
    fill = np.rint(image.mean(axis=(0, 1))).astype(np.uint8)

    for row in range(0, IMG_SIZE, PATCH_SIZE):
        for column in range(0, IMG_SIZE, PATCH_SIZE):
            patched = image.copy()
            patched[row:row + PATCH_SIZE, column:column + PATCH_SIZE] = fill
            patched_images.append(patched)
            positions.append((row // PATCH_SIZE, column // PATCH_SIZE))

    patched_logits = logits(model, np.stack(patched_images))
    side = IMG_SIZE // PATCH_SIZE
    heatmap = np.zeros((side, side), dtype=np.float32)
    for (row, column), patched_logit in zip(positions, patched_logits):
        heatmap[row, column] = base_logit - patched_logit
    return heatmap


def select_examples(labels, predictions, scores):
    masks = {
        "true_positive": (labels == 1) & (predictions == 1),
        "true_negative": (labels == 0) & (predictions == 0),
        "false_positive": (labels == 0) & (predictions == 1),
        "false_negative": (labels == 1) & (predictions == 0),
    }
    selected = {}
    for category, mask in masks.items():
        indices = np.flatnonzero(mask)
        descending = category in {"true_positive", "false_positive"}
        order = np.argsort(scores[indices])
        if descending:
            order = order[::-1]
        selected[category] = indices[order[:EXAMPLES_PER_CATEGORY]]
    return selected


def title(index, source, score):
    return f"idx={index} | {SOURCE_NAMES.get(int(source), source)}\nscore={score:.3f}"


def plot_saliency_examples(model, images, sources, scores, selected):
    columns = [
        (category, label, int(index))
        for category, label in CATEGORIES
        for index in selected[category]
    ]
    if not columns:
        return

    figure, axes = plt.subplots(2, len(columns), figsize=(2.0 * len(columns), 4.5), squeeze=False)
    for column, (category, label, index) in enumerate(columns):
        axes[0, column].imshow(images[index])
        axes[0, column].set_title(f"{label}\n{title(index, sources[index], scores[index])}", fontsize=7)
        axes[0, column].axis("off")

        axes[1, column].imshow(images[index], alpha=0.55)
        axes[1, column].imshow(
            display_map(saliency(model, images[index])),
            cmap="inferno",
            vmin=0,
            vmax=1,
            alpha=0.7,
        )
        axes[1, column].axis("off")

    figure.suptitle("Input-gradient saliency: original images and overlays")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "saliency_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_occlusion_errors(model, images, sources, scores, image_logits, selected):
    columns = [
        (category, label, int(index))
        for category, label in CATEGORIES[2:]
        for index in selected[category]
    ]
    if not columns:
        return

    maps = [
        occlusion_map(model, images[index], image_logits[index])
        for _, _, index in columns
    ]
    limit = float(np.percentile(np.abs(np.concatenate([m.ravel() for m in maps])), 99))
    limit = limit if limit > 0 else 1.0

    figure, axes = plt.subplots(2, len(columns), figsize=(2.2 * len(columns), 4.7), squeeze=False)
    for column, ((_, label, index), heatmap) in enumerate(zip(columns, maps)):
        axes[0, column].imshow(images[index])
        axes[0, column].set_title(f"{label}\n{title(index, sources[index], scores[index])}", fontsize=7)
        axes[0, column].axis("off")

        axes[1, column].imshow(images[index], alpha=0.55)
        axes[1, column].imshow(
            heatmap,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            alpha=0.7,
            interpolation="nearest",
            extent=(0, IMG_SIZE, IMG_SIZE, 0),
        )
        axes[1, column].axis("off")

    figure.suptitle(
        "Signed occlusion sensitivity (red supports AI, blue supports real)"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "occlusion_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def sample_mean_indices(labels, sources):
    rng = np.random.default_rng(SEED)
    real = np.flatnonzero(labels == 0)
    real = rng.choice(real, min(MEAN_EXAMPLES_PER_CLASS, len(real)), replace=False)

    ai = []
    ai_sources = [value for value in sorted(np.unique(sources)) if value != 0]
    quota = max(1, MEAN_EXAMPLES_PER_CLASS // max(1, len(ai_sources)))
    for source in ai_sources:
        candidates = np.flatnonzero(sources == source)
        take = min(quota, len(candidates))
        ai.extend(rng.choice(candidates, take, replace=False).tolist())
    return np.asarray(real), np.asarray(ai[:MEAN_EXAMPLES_PER_CLASS])


def plot_mean_saliency(model, images, labels, sources):
    real_indices, ai_indices = sample_mean_indices(labels, sources)
    real_maps = [mass_map(saliency(model, images[index])) for index in real_indices]
    ai_maps = [mass_map(saliency(model, images[index])) for index in ai_indices]
    mean_real = np.mean(real_maps, axis=0)
    mean_ai = np.mean(ai_maps, axis=0)
    limit = float(np.percentile(np.concatenate([mean_real.ravel(), mean_ai.ravel()]), 99))
    limit = limit if limit > 0 else 1.0

    figure, axes = plt.subplots(1, 2, figsize=(7, 3.3))
    for axis, values, label in (
        (axes[0], mean_real, f"Real (n={len(real_indices)})"),
        (axes[1], mean_ai, f"AI, source-stratified (n={len(ai_indices)})"),
    ):
        image = axis.imshow(values, cmap="inferno", vmin=0, vmax=limit)
        axis.set_title(label)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Mean normalized input-gradient saliency")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mean_saliency.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    row = slice(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
    column = slice(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
    return {
        "real_examples": int(len(real_indices)),
        "ai_examples": int(len(ai_indices)),
        "real_center_mass": float(mean_real[row, column].sum()),
        "ai_center_mass": float(mean_ai[row, column].sum()),
    }


def split_summary(labels, scores, sources, threshold):
    predictions = (scores > threshold).astype(np.int64)
    summary = {
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
        "recall_ai": float((predictions[labels == 1] == 1).mean()),
        "fpr_real": float((predictions[labels == 0] == 1).mean()),
        "mean_score_real": float(scores[labels == 0].mean()),
        "mean_score_ai": float(scores[labels == 1].mean()),
        "per_source": {},
    }
    for source in sorted(np.unique(sources)):
        mask = sources == source
        entry = {
            "name": SOURCE_NAMES.get(int(source), str(source)),
            "count": int(mask.sum()),
            "mean_score": float(scores[mask].mean()),
        }
        if source == 0:
            entry["fpr"] = float((predictions[mask] == 1).mean())
        else:
            entry["recall"] = float((predictions[mask] == 1).mean())
        summary["per_source"][str(int(source))] = entry
    return summary


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "models" / "model_aug.pt"
    threshold_path = ARTIFACTS_DIR / "models" / "threshold_aug.json"
    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Run train_augmented.py before explain.py")

    model = SmallCNN()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    with threshold_path.open(encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])

    images, labels, sources = load_split("validation", with_source=True)
    image_logits = logits(model, images)
    scores = 1.0 / (1.0 + np.exp(-np.clip(image_logits, -50, 50)))
    predictions = (scores > threshold).astype(np.int64)
    selected = select_examples(labels, predictions, scores)

    plot_saliency_examples(model, images, sources, scores, selected)
    plot_occlusion_errors(
        model, images, sources, scores, image_logits, selected
    )
    mean_summary = plot_mean_saliency(model, images, labels, sources)

    augmented_images, augmented_labels, augmented_sources = load_split(
        "validation_augmented", with_source=True
    )
    augmented_scores = 1.0 / (
        1.0 + np.exp(-np.clip(logits(model, augmented_images), -50, 50))
    )

    summary = {
        "threshold": threshold,
        "validation": split_summary(labels, scores, sources, threshold),
        "validation_augmented": split_summary(
            augmented_labels, augmented_scores, augmented_sources, threshold
        ),
        "selected_validation_indices": {
            category: [int(index) for index in indices]
            for category, indices in selected.items()
        },
        "mean_saliency": mean_summary,
        "methods": {
            "saliency": "absolute gradient of the AI logit with respect to pixels",
            "occlusion": (
                f"signed AI-logit change after replacing {PATCH_SIZE}x{PATCH_SIZE} "
                "patches with the image mean"
            ),
        },
        "limitations": [
            "Saliency measures local sensitivity, not causality.",
            "Absolute gradients do not show the direction of evidence.",
            "Occlusion depends on patch size and creates artificial inputs.",
            "These maps cannot rule out diffuse compression or texture shortcuts.",
        ],
    }
    save_json(summary, OUTPUT_DIR / "summary.json")
    print(
        f"Task 4 complete: validation recall={summary['validation']['recall_ai']:.4f}, "
        f"fpr={summary['validation']['fpr_real']:.4f}. Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
