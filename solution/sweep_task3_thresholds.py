import numpy as np
import torch

from common import (
    ARTIFACTS_DIR,
    SmallCNN,
    calibrate_threshold,
    evaluate,
    load_split,
    score_images,
    set_seed,
)


def main():
    set_seed()

    checkpoint = (
        ARTIFACTS_DIR
        / "models"
        / "model_aug_balanced_pil_step641.pt"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = SmallCNN()
    model.load_state_dict(
        torch.load(checkpoint, weights_only=True)
    )

    cal_clean_x, cal_clean_y = load_split("calibration")
    cal_aug_x, cal_aug_y = load_split("calibration_augmented")

    clean_real_scores = score_images(
        model,
        cal_clean_x[cal_clean_y == 0],
    )
    aug_real_scores = score_images(
        model,
        cal_aug_x[cal_aug_y == 0],
    )

    validation_data = {}
    for split in ("validation", "validation_augmented"):
        x, y = load_split(split)
        validation_data[split] = (
            y,
            score_images(model, x),
        )

    for target_fpr in (0.16, 0.17, 0.18, 0.19):
        threshold_clean = calibrate_threshold(
            clean_real_scores,
            target_fpr=target_fpr,
        )
        threshold_aug = calibrate_threshold(
            aug_real_scores,
            target_fpr=target_fpr,
        )
        threshold = max(threshold_clean, threshold_aug)

        print("=" * 70)
        print(
            f"calibration_target={target_fpr:.2f} "
            f"threshold={threshold:.6f} "
            f"(clean={threshold_clean:.6f}, "
            f"aug={threshold_aug:.6f})"
        )

        for split, (y, scores) in validation_data.items():
            metrics = evaluate(y, scores, threshold)
            print(
                f"{split}: "
                f"recall={metrics['recall']:.4f} "
                f"fpr={metrics['fpr']:.4f} "
                f"auc={metrics['roc_auc']:.4f}"
            )


if __name__ == "__main__":
    main()