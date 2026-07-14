"""Classical Task 1.2 baseline: Random Forest on 20 engineered features."""

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
    save_json,
    set_seed,
)

OUTPUT = ARTIFACTS_DIR / "baseline" / "metrics.json"
FFT_BINS = 8


def extract_features(images, chunk_size=2048):
    return np.concatenate(
        [
            extract_chunk(images[start:start + chunk_size])
            for start in range(0, len(images), chunk_size)
        ]
    )


def extract_chunk(images):
    values = images.astype(np.float32) / 255.0
    _, height, width, _ = values.shape

    gray = values.mean(axis=3)
    dx = np.diff(gray, axis=2)
    dy = np.diff(gray, axis=1)
    magnitude = np.sqrt(dx[:, :-1] ** 2 + dy[:, :, :-1] ** 2)
    chroma = values.max(axis=3) - values.min(axis=3)

    spectrum = np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2(gray), axes=(1, 2)))
    )
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt((yy - height / 2) ** 2 + (xx - width / 2) ** 2)
    radial_bin = np.minimum(
        (radius / radius.max() * FFT_BINS).astype(int), FFT_BINS - 1
    )
    radial_profile = np.stack(
        [spectrum[:, radial_bin == index].mean(axis=1) for index in range(FFT_BINS)],
        axis=1,
    )

    return np.concatenate(
        [
            values.mean(axis=(1, 2)),
            values.std(axis=(1, 2)),
            np.stack(
                [
                    np.abs(dx).mean(axis=(1, 2)),
                    np.abs(dy).mean(axis=(1, 2)),
                    magnitude.mean(axis=(1, 2)),
                    magnitude.std(axis=(1, 2)),
                ],
                axis=1,
            ),
            np.stack(
                [chroma.mean(axis=(1, 2)), chroma.std(axis=(1, 2))], axis=1
            ),
            radial_profile,
        ],
        axis=1,
    )


def main():
    set_seed()
    cache = np.load(ARTIFACTS_DIR / "prepared" / "train.npz", mmap_mode="r")

    start = time.perf_counter()
    training_features = extract_features(cache["X"])
    feature_seconds = time.perf_counter() - start

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=8,
    )
    start = time.perf_counter()
    model.fit(training_features, cache["y"].astype(int))
    fit_seconds = time.perf_counter() - start

    calibration_images, calibration_labels = load_split("calibration")
    real_scores = model.predict_proba(
        extract_features(calibration_images[calibration_labels == 0])
    )[:, 1]
    threshold = calibrate_threshold(real_scores)

    results = {
        "model_family": (
            "Random Forest on color, gradient, RGB-chroma and FFT radial features"
        ),
        "n_features": int(training_features.shape[1]),
        "feature_extraction_seconds": feature_seconds,
        "model_fit_seconds": fit_seconds,
        "threshold": threshold,
    }
    for split in ("validation", "validation_augmented"):
        images, labels = load_split(split)
        scores = model.predict_proba(extract_features(images))[:, 1]
        results[split] = evaluate(labels, scores, threshold)

    save_json(results, OUTPUT)
    print(
        f"Baseline complete in {feature_seconds + fit_seconds:.1f}s: "
        f"validation recall={results['validation']['recall']:.4f}, "
        f"fpr={results['validation']['fpr']:.4f}, "
        f"augmented recall={results['validation_augmented']['recall']:.4f}, "
        f"fpr={results['validation_augmented']['fpr']:.4f}"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
