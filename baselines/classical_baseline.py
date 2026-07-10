# baselines/classical_baseline.py — second model family for the Task 1.2
# comparison (classical baseline on engineered features vs. the packaged CNN).
# Development/report code: lives OUTSIDE solution/ because only the single best
# pipeline is packaged, as allowed by the assignment.
#
# Run with the solution Docker image (repo mounted, prepared cache built):
#   docker run --rm -v <repo>:/workspace/repo -w /workspace/repo \
#     amls-solution python baselines/classical_baseline.py
#
# Protocol mirrors train.py exactly: train on the cleaned cache, calibrate the
# threshold on data/calibration REAL images (FPR <= 20%), verify on
# data/validation and data/validation_augmented.

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "solution"))

from common import (  # noqa: E402
    ARTIFACTS_DIR,
    SEED,
    calibrate_threshold,
    evaluate,
    load_split,
    set_seed,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
N_FFT_BINS = 8  # radial bins of the log-magnitude spectrum


def features(images):
    """Engineered features for uint8 images (N, H, W, 3).

    Per-channel mean/std, gradient-magnitude mean/std, saturation mean/std and
    a radial profile of the FFT log-magnitude spectrum (AI generators tend to
    leave characteristic high-frequency artifacts).
    """
    x = images.astype(np.float32) / 255.0
    n, h, w, _ = x.shape
    feats = []

    feats.append(x.mean(axis=(1, 2)))  # (N, 3)
    feats.append(x.std(axis=(1, 2)))   # (N, 3)

    gray = x.mean(axis=3)
    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    gm = np.sqrt(
        np.diff(gray, axis=1)[:, :, :-1] ** 2 + np.diff(gray, axis=2)[:, :-1, :] ** 2
    )
    feats.append(np.stack([gx, gy, gm.mean(axis=(1, 2)), gm.std(axis=(1, 2))], axis=1))

    sat = x.max(axis=3) - x.min(axis=3)  # cheap saturation proxy
    feats.append(np.stack([sat.mean(axis=(1, 2)), sat.std(axis=(1, 2))], axis=1))

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray), axes=(1, 2))))
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    bins = np.minimum((radius / radius.max() * N_FFT_BINS).astype(int), N_FFT_BINS - 1)
    radial = np.stack(
        [spectrum[:, bins == b].mean(axis=1) for b in range(N_FFT_BINS)], axis=1
    )
    feats.append(radial)

    return np.concatenate(feats, axis=1)


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz")
    X_train, y_train = features(cache["X"]), cache["y"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=8,
    )
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    cal_X, cal_y = load_split("calibration")
    cal_scores_real = model.predict_proba(features(cal_X[cal_y == 0]))[:, 1]
    threshold = calibrate_threshold(cal_scores_real)

    results = {
        "model_family": "RandomForest on engineered features "
                        "(color/gradient/saturation stats + FFT radial profile)",
        "n_features": int(X_train.shape[1]),
        "train_seconds": round(train_seconds, 1),
        "threshold": threshold,
    }
    for split in ("validation", "validation_augmented"):
        sx, sy = load_split(split)
        scores = model.predict_proba(features(sx))[:, 1]
        results[split] = evaluate(sy, scores, threshold)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"threshold={threshold:.6f} (train {train_seconds:.0f}s)")
    for split in ("validation", "validation_augmented"):
        m = results[split]
        print(f"{split}: recall_ai={m['recall']:.4f} fpr={m['fpr']:.4f} "
              f"roc_auc={m['roc_auc']:.4f}")
    print(f"wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
