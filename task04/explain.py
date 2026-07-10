# task04/explain.py — Task 1.4 explainability for the final (Task 3) model.
# Lives OUTSIDE solution/ as required by the assignment structure.
#
# Run with the solution Docker image (repo mounted, artifacts already built):
#   docker run --rm -v <repo>:/workspace/repo -w /workspace/repo \
#     amls-solution python task04/explain.py
#
# Methods: vanilla-gradient saliency maps, occlusion analysis, false-positive /
# false-negative analysis, and a real-vs-ai comparison of mean saliency.
# Outputs (figures + summary.json) are written to task04/outputs/.

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
sys.path.insert(0, str(REPO_DIR / "solution"))

from common import (  # noqa: E402
    ARTIFACTS_DIR,
    DATA_DIR,
    IMG_SIZE,
    SmallCNN,
    load_split,
    score_images,
    set_seed,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
N_EXAMPLES = 8        # images per category in the saliency grids
N_OCCLUSION = 4       # images per category in the occlusion grids
N_MEAN = 100          # images per class for the mean-saliency comparison
PATCH = 8             # occlusion patch side and stride (pixels)


def saliency(model, img):
    """Vanilla gradient saliency |d logit / d pixel|, max over RGB channels."""
    x = torch.from_numpy(img[None].copy()).permute(0, 3, 1, 2).float().div_(255.0)
    x.requires_grad_(True)
    model.eval()
    model(x).backward()
    return x.grad.abs().max(dim=1).values.squeeze(0).numpy()


def occlusion_map(model, img, base_score):
    """Score drop when a gray patch occludes each region (higher = evidence for ai)."""
    occluded, positions = [], []
    for i in range(0, IMG_SIZE, PATCH):
        for j in range(0, IMG_SIZE, PATCH):
            patched = img.copy()
            patched[i:i + PATCH, j:j + PATCH] = 128
            occluded.append(patched)
            positions.append((i // PATCH, j // PATCH))
    scores = score_images(model, np.stack(occluded))
    heat = np.zeros((IMG_SIZE // PATCH, IMG_SIZE // PATCH), np.float32)
    for (r, c), s in zip(positions, scores):
        heat[r, c] = base_score - s
    return heat


def grid_figure(images, maps, scores, title, path):
    n = len(images)
    if n == 0:
        print(f"skip {path.name}: no examples")
        return
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6), squeeze=False)
    for k in range(n):
        axes[0][k].imshow(images[k])
        axes[0][k].set_title(f"score={scores[k]:.2f}", fontsize=8)
        axes[0][k].axis("off")
        axes[1][k].imshow(maps[k], cmap="inferno")
        axes[1][k].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def source_classes(split):
    """Raw source_class values of a labeled split, same row order as load_split."""
    values = []
    for f in sorted((DATA_DIR / split).glob("*.parquet")):
        try:
            col = pd.read_parquet(f, columns=["source_class"])["source_class"]
        except Exception:
            col = pd.read_parquet(f, columns=["source class"])["source class"]
        values.append(col.to_numpy())
    return np.concatenate(values)


def split_summary(y, scores, src, threshold):
    pred = (scores > threshold).astype(int)
    per_class_recall = {
        int(c): float((pred[src == c] == 1).mean()) for c in sorted(set(src)) if c != 0
    }
    return {
        "counts": {
            "tp": int(((y == 1) & (pred == 1)).sum()),
            "tn": int(((y == 0) & (pred == 0)).sum()),
            "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()),
        },
        "fpr_real": float((pred[y == 0] == 1).mean()),
        "recall_ai": float((pred[y == 1] == 1).mean()),
        "mean_score_real": float(scores[y == 0].mean()),
        "mean_score_ai": float(scores[y == 1].mean()),
        "recall_per_source_class": per_class_recall,
    }


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = SmallCNN()
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "models" / "model_aug.pt", weights_only=True)
    )
    with open(ARTIFACTS_DIR / "models" / "threshold_aug.json") as f:
        threshold = json.load(f)["threshold"]

    X, y = load_split("validation")
    scores = score_images(model, X)
    pred = (scores > threshold).astype(int)

    # --- categories, most confident first -------------------------------------
    categories = {
        "true_positive_ai": np.where((y == 1) & (pred == 1))[0][
            np.argsort(-scores[(y == 1) & (pred == 1)])
        ],
        "true_negative_real": np.where((y == 0) & (pred == 0))[0][
            np.argsort(scores[(y == 0) & (pred == 0)])
        ],
        "false_positive_real_as_ai": np.where((y == 0) & (pred == 1))[0][
            np.argsort(-scores[(y == 0) & (pred == 1)])
        ],
        "false_negative_ai_missed": np.where((y == 1) & (pred == 0))[0][
            np.argsort(scores[(y == 1) & (pred == 0)])
        ],
    }

    # --- saliency grids per category -------------------------------------------
    for name, idx in categories.items():
        idx = idx[:N_EXAMPLES]
        grid_figure(
            [X[i] for i in idx],
            [saliency(model, X[i]) for i in idx],
            [scores[i] for i in idx],
            f"{name} (top: image, bottom: gradient saliency)",
            OUT_DIR / f"saliency_{name}.png",
        )

    # --- occlusion analysis on the model's mistakes ----------------------------
    for name in ("false_positive_real_as_ai", "false_negative_ai_missed"):
        idx = categories[name][:N_OCCLUSION]
        grid_figure(
            [X[i] for i in idx],
            [occlusion_map(model, X[i], scores[i]) for i in idx],
            [scores[i] for i in idx],
            f"{name} (top: image, bottom: occlusion score-drop)",
            OUT_DIR / f"occlusion_{name}.png",
        )

    # --- what the model attends to: real vs ai ---------------------------------
    real_idx = np.where(y == 0)[0][:N_MEAN]
    ai_idx = np.where(y == 1)[0][:N_MEAN]
    mean_real = np.mean([saliency(model, X[i]) for i in real_idx], axis=0)
    mean_ai = np.mean([saliency(model, X[i]) for i in ai_idx], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, m, title in ((axes[0], mean_real, f"mean saliency: real (n={len(real_idx)})"),
                         (axes[1], mean_ai, f"mean saliency: ai (n={len(ai_idx)})")):
        im = ax.imshow(m, cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_saliency_real_vs_ai.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'mean_saliency_real_vs_ai.png'}")

    # --- quantitative summary (both validation splits) -------------------------
    summary = {
        "model": "artifacts/models/model_aug.pt",
        "threshold": threshold,
        "validation": split_summary(y, scores, source_classes("validation"), threshold),
    }
    Xa, ya = load_split("validation_augmented")
    summary["validation_augmented"] = split_summary(
        ya, score_images(model, Xa), source_classes("validation_augmented"), threshold
    )
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
