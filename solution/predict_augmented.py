# predict_augmented.py — Task 3 inference: write artifacts/task03/predictions.csv.

import argparse
import json

import torch

from common import ARTIFACTS_DIR, SmallCNN, load_split, score_images, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.parse_args()

    set_seed()

    models_dir = ARTIFACTS_DIR / "models"
    model = SmallCNN()
    model.load_state_dict(torch.load(models_dir / "model_aug.pt", weights_only=True))
    with open(models_dir / "threshold_aug.json") as f:
        threshold = json.load(f)["threshold"]

    images, row_ids = load_split("predict")
    preds = (score_images(model, images) > threshold).astype(int)

    out_dir = ARTIFACTS_DIR / "task03"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    with open(out_path, "w") as f:
        f.write("row_id,predicted_label\n")
        for row_id, pred in zip(row_ids, preds):
            f.write(f"{row_id},{pred}\n")

    print(f"Wrote {len(preds)} predictions ({int(preds.sum())} ai_generated) -> {out_path}")


if __name__ == "__main__":
    main()
