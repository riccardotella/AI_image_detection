"""Task 1.1: deterministic exploration and cleaning of data/train."""

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "artifacts" / "clean"


def get_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict) and "bytes" in value:
        return get_bytes(value["bytes"])
    return None


def inspect_image(value):
    data = get_bytes(value)
    if data is None:
        return {"valid": False, "error": "unsupported image value"}

    result = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_size": len(data),
    }
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            result.update(
                valid=True,
                width=width,
                height=height,
                aspect_ratio=width / height,
                format=image.format,
                mode=image.mode,
                error="",
            )
    except Exception as error:
        result.update(valid=False, error=f"{type(error).__name__}: {error}")
    return result


def describe(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        key: float(value)
        for key, value in values.describe(percentiles=[0.25, 0.5, 0.75]).items()
    }


def counts(series):
    return {str(key): int(value) for key, value in series.value_counts().items()}


def class_stats(frame):
    valid = frame[frame["valid"]]
    return {
        "count": int(len(frame)),
        "kept": int(frame["keep"].sum()),
        "keep_rate": float(frame["keep"].mean()),
        "width": describe(valid["width"]),
        "height": describe(valid["height"]),
        "aspect_ratio": describe(valid["aspect_ratio"]),
        "file_size": describe(valid["file_size"]),
        "formats": counts(valid["format"]),
        "modes": counts(valid["mode"]),
        "top_dimensions": [
            {"width": int(width), "height": int(height), "count": int(count)}
            for (width, height), count in (
                valid.groupby(["width", "height"]).size().nlargest(10).items()
            )
        ],
    }


def save_json(data, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_figures(frame, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    binary_counts = frame["label"].value_counts().reindex([0, 1], fill_value=0)
    binary_counts.index = ["real", "AI-generated"]
    plt.figure(figsize=(5.5, 3.6))
    binary_counts.plot.bar()
    plt.title("Binary class distribution")
    plt.ylabel("images")
    plt.xticks(rotation=0)
    save_figure(figures_dir / "class_distribution.png")

    valid = frame[frame["valid"]]
    plt.figure(figsize=(5.5, 4.4))
    for label, name in ((0, "real"), (1, "AI-generated")):
        subset = valid[valid["label"] == label]
        plt.scatter(subset["width"], subset["height"], s=7, alpha=0.3, label=name)
    plt.title("Original image dimensions")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.legend()
    save_figure(figures_dir / "image_size_distribution.png")

    real = valid.loc[valid["label"] == 0, "file_size"].to_numpy(float)
    ai = valid.loc[valid["label"] == 1, "file_size"].to_numpy(float)
    low = max(1.0, float(min(real.min(), ai.min())))
    high = float(max(real.max(), ai.max()))
    bins = np.logspace(np.log10(low), np.log10(high), 50)
    plt.figure(figsize=(5.5, 3.6))
    plt.hist(real, bins=bins, density=True, alpha=0.55, label="real")
    plt.hist(ai, bins=bins, density=True, alpha=0.55, label="AI-generated")
    plt.xscale("log")
    plt.title("Encoded file-size distribution")
    plt.xlabel("bytes (log scale)")
    plt.ylabel("density")
    plt.legend()
    save_figure(figures_dir / "file_size_distribution.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()
    start = time.perf_counter()

    files = sorted(TRAIN_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {TRAIN_DIR}")

    rows = []
    for path in files:
        print(f"Reading {path}")
        frame = pd.read_parquet(path)
        label_column = next(
            (column for column in ("source_class", "source class") if column in frame),
            None,
        )
        if label_column is None:
            raise ValueError(f"No source-class column in {path}")

        for row_index, (image, source) in enumerate(
            zip(frame["image"], frame[label_column])
        ):
            source = int(source)
            rows.append(
                {
                    "parquet_file": path.name,
                    "row_idx": row_index,
                    "source_class": source,
                    "label": int(source != 0),
                    **inspect_image(image),
                }
            )

    manifest = pd.DataFrame(rows)
    hashed = manifest[manifest["sha256"].notna()]
    group_sizes = hashed.groupby("sha256").size()
    duplicate_hashes = set(group_sizes[group_sizes > 1].index)
    conflicting_hashes = set(
        hashed.groupby("sha256")["label"].nunique().loc[lambda values: values > 1].index
    )

    duplicate_copy = manifest.duplicated("sha256", keep="first") & manifest["sha256"].notna()
    conflict = manifest["sha256"].isin(conflicting_hashes)
    manifest["keep"] = (
        manifest["valid"]
        & ~duplicate_copy
        & ~conflict
    )
    manifest["removal_reason"] = np.select(
        [
            ~manifest["valid"],
            conflict,
            duplicate_copy,
        ],
        [
            "invalid_or_unreadable",
            "duplicate_with_conflicting_labels",
            "exact_duplicate",
        ],
        default="kept",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_columns = ["parquet_file", "row_idx", "label", "keep", "removal_reason"]
    manifest[output_columns].to_csv(OUTPUT_DIR / "cleaned_manifest.csv", index=False)

    valid = manifest[manifest["valid"]]
    stats = {
        "cleaning_rule": {
            "require_full_decode": True,
            "same_label_duplicates": "keep first occurrence",
            "conflicting_label_duplicates": "remove whole group",
        },
        "total": int(len(manifest)),
        "valid": int(manifest["valid"].sum()),
        "kept": int(manifest["keep"].sum()),
        "removed": int((~manifest["keep"]).sum()),
        "duplicate_groups": int(len(duplicate_hashes)),
        "duplicate_copies_removed": int(duplicate_copy.sum()),
        "conflicting_duplicate_groups": int(len(conflicting_hashes)),
        "removal_reasons": counts(manifest["removal_reason"]),
        "source_classes": counts(manifest["source_class"]),
        "binary_classes": counts(manifest["label"]),
        "all_valid_images": {
            "width": describe(valid["width"]),
            "height": describe(valid["height"]),
            "aspect_ratio": describe(valid["aspect_ratio"]),
            "file_size": describe(valid["file_size"]),
            "formats": counts(valid["format"]),
            "modes": counts(valid["mode"]),
        },
        "real": class_stats(manifest[manifest["label"] == 0]),
        "ai_generated": class_stats(manifest[manifest["label"] == 1]),
        "interpretation_note": (
            "Differences in dimensions, encoding and file size are dataset correlations, "
            "not causal evidence of authenticity."
        ),
    }
    save_json(stats, OUTPUT_DIR / "stats.json")
    make_figures(manifest, OUTPUT_DIR / "figures")

    elapsed = time.perf_counter() - start
    print(
        f"Cleaning complete: {stats['kept']}/{stats['total']} kept, "
        f"{stats['duplicate_copies_removed']} duplicate copies removed, "
        f"{stats['conflicting_duplicate_groups']} conflicting groups."
    )
    print(f"Elapsed {elapsed:.1f}s / timeout {args.timeout_seconds}s")


if __name__ == "__main__":
    main()
