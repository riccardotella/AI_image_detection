# clean.py — deterministic exploration and cleaning for Task 1.1.
# Reads only data/train/ and writes only under artifacts/clean/.

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe inside Docker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MIN_SIDE = 32


def get_label_column(df):
    """Return the source-class column name used by the parquet file."""
    for name in ("source_class", "source class"):
        if name in df.columns:
            return name
    raise ValueError(f"Label column not found. Columns are: {list(df.columns)}")


def get_image_bytes(value):
    """Normalize the possible parquet representations of a binary image."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict) and "bytes" in value:
        nested = value["bytes"]
        if isinstance(nested, bytes):
            return nested
        if isinstance(nested, (bytearray, memoryview)):
            return bytes(nested)
    return None


def inspect_image(image_bytes):
    """Decode one image and return metadata used for exploration and cleaning."""
    if image_bytes is None:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "pixel_count": None,
            "format": None,
            "mode": None,
            "file_size": None,
            "sha256": None,
            "error": "missing or unsupported image bytes",
        }

    sha256 = hashlib.sha256(image_bytes).hexdigest()

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()  # force full decoding, so truncated/corrupt files fail here
            width, height = image.size
            image_format = image.format
            image_mode = image.mode

        return {
            "valid": True,
            "width": int(width),
            "height": int(height),
            "aspect_ratio": float(width / height) if height else None,
            "pixel_count": int(width * height),
            "format": image_format,
            "mode": image_mode,
            "file_size": int(len(image_bytes)),
            "sha256": sha256,
            "error": "",
        }
    except Exception as exc:  # malformed inputs are recorded, not silently ignored
        return {
            "valid": False,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "pixel_count": None,
            "format": None,
            "mode": None,
            "file_size": int(len(image_bytes)),
            "sha256": sha256,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _save_figure(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_class_distribution(df, output_path):
    """Binary class distribution after merging source classes 1..5 into AI."""
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    counts.index = ["real", "ai_generated"]

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Binary class distribution")
    plt.xlabel("class")
    plt.ylabel("number of images")
    plt.xticks(rotation=0)
    _save_figure(output_path)


def plot_source_class_distribution(df, output_path):
    """Original six-way source-class distribution for dataset diagnostics."""
    counts = df["source_class"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.title("Original source-class distribution")
    plt.xlabel("source_class (0=real, 1..5=AI generators)")
    plt.ylabel("number of images")
    plt.xticks(rotation=0)
    _save_figure(output_path)


def plot_size_distribution(df, output_path):
    """Original width/height distribution, separated by binary class."""
    valid = df[df["valid"]]
    real = valid[valid["label"] == 0]
    ai = valid[valid["label"] == 1]

    plt.figure(figsize=(6, 5))
    plt.scatter(real["width"], real["height"], s=8, alpha=0.35, label="real")
    plt.scatter(ai["width"], ai["height"], s=8, alpha=0.25, label="ai_generated")
    plt.title("Original image-size distribution")
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.legend()
    _save_figure(output_path)


def plot_file_size_distribution(df, output_path):
    """Encoded file-size distribution, useful for identifying format shortcuts."""
    valid = df[df["valid"] & df["file_size"].notna() & (df["file_size"] > 0)]
    real = valid.loc[valid["label"] == 0, "file_size"].to_numpy(dtype=float)
    ai = valid.loc[valid["label"] == 1, "file_size"].to_numpy(dtype=float)

    all_sizes = np.concatenate([real, ai])
    low = max(1.0, float(all_sizes.min()))
    high = max(low * 1.01, float(all_sizes.max()))
    bins = np.logspace(np.log10(low), np.log10(high), 50)

    plt.figure(figsize=(6, 4))
    plt.hist(real, bins=bins, alpha=0.55, density=True, label="real")
    plt.hist(ai, bins=bins, alpha=0.55, density=True, label="ai_generated")
    plt.xscale("log")
    plt.title("Encoded file-size distribution")
    plt.xlabel("file size (bytes, log scale)")
    plt.ylabel("density")
    plt.legend()
    _save_figure(output_path)


def plot_aspect_ratio_distribution(df, output_path):
    """Aspect-ratio distribution, separated by binary class."""
    valid = df[df["valid"] & df["aspect_ratio"].notna()]
    real = valid.loc[valid["label"] == 0, "aspect_ratio"]
    ai = valid.loc[valid["label"] == 1, "aspect_ratio"]

    # Robust display range: extreme outliers remain in the CSV/JSON statistics.
    upper = float(valid["aspect_ratio"].quantile(0.995))
    upper = max(upper, 1.0)
    bins = np.linspace(0.0, upper, 50)

    plt.figure(figsize=(6, 4))
    plt.hist(real.clip(upper=upper), bins=bins, alpha=0.55, density=True, label="real")
    plt.hist(ai.clip(upper=upper), bins=bins, alpha=0.55, density=True, label="ai_generated")
    plt.title("Original aspect-ratio distribution")
    plt.xlabel("width / height")
    plt.ylabel("density")
    plt.legend()
    _save_figure(output_path)


def _describe(series):
    """JSON-safe numeric summary."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    desc = values.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {str(key): float(value) for key, value in desc.items()}


def _counts(series):
    """JSON-safe value counts with string keys."""
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _top_dimensions(df, limit=10):
    valid = df[df["valid"]]
    counts = valid.groupby(["width", "height"], dropna=False).size()
    counts = counts.sort_values(ascending=False).head(limit)
    return [
        {"width": int(width), "height": int(height), "count": int(count)}
        for (width, height), count in counts.items()
    ]


def _atomic_to_csv(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(payload, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    tmp.replace(path)


def _removal_reason(row):
    if not row["valid"]:
        return "invalid_or_unreadable"
    if row["width"] < MIN_SIDE or row["height"] < MIN_SIDE:
        return "side_below_minimum"
    if row["duplicate_label_conflict"]:
        return "duplicate_with_conflicting_labels"
    if row["is_exact_duplicate"]:
        return "exact_duplicate"
    return "kept"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()
    start = time.perf_counter()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" / "train"
    output_dir = script_dir / "artifacts" / "clean"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    rows = []
    for parquet_file in parquet_files:
        print(f"Reading {parquet_file}")
        df = pd.read_parquet(parquet_file)
        label_col = get_label_column(df)

        # Store the positional row index because prepare.py later reads with iloc.
        for row_position, (image_cell, source_value) in enumerate(
            zip(df["image"].tolist(), df[label_col].tolist())
        ):
            source_class = int(source_value)
            rows.append(
                {
                    "parquet_file": parquet_file.name,
                    "row_idx": int(row_position),
                    "source_class": source_class,
                    "label": 0 if source_class == 0 else 1,
                    **inspect_image(get_image_bytes(image_cell)),
                }
            )

    manifest = pd.DataFrame(rows)

    # Duplicate diagnostics. The first occurrence is deterministic because files
    # and row positions are traversed in sorted order.
    hashed = manifest[manifest["sha256"].notna()]
    duplicate_sizes = hashed.groupby("sha256").size()
    duplicate_hashes = set(duplicate_sizes[duplicate_sizes > 1].index)
    conflicting_hashes = set(
        hashed.groupby("sha256")["label"].nunique().loc[lambda s: s > 1].index
    )

    manifest["is_exact_duplicate"] = (
        manifest["sha256"].notna()
        & manifest.duplicated(subset=["sha256"], keep="first")
    )
    manifest["duplicate_label_conflict"] = manifest["sha256"].isin(conflicting_hashes)

    base_keep = (
        manifest["valid"]
        & (manifest["width"] >= MIN_SIDE)
        & (manifest["height"] >= MIN_SIDE)
    )

    # Same-label duplicates: keep the first occurrence only.
    # Conflicting-label duplicates: remove the whole ambiguous group.
    manifest["keep"] = (
        base_keep
        & ~manifest["is_exact_duplicate"]
        & ~manifest["duplicate_label_conflict"]
    )
    manifest["removal_reason"] = manifest.apply(_removal_reason, axis=1)

    valid = manifest[manifest["valid"]]
    kept = manifest[manifest["keep"]]
    removed = manifest[~manifest["keep"]]

    _atomic_to_csv(manifest, output_dir / "cleaned_manifest.csv")
    _atomic_to_csv(removed, output_dir / "removed_records.csv")

    stats = {
        "cleaning_rule": {
            "minimum_side_pixels": MIN_SIDE,
            "require_full_decode": True,
            "same_label_exact_duplicates": "keep first deterministic occurrence",
            "conflicting_label_exact_duplicates": "remove entire ambiguous group",
        },
        "n_total": int(len(manifest)),
        "n_valid": int(manifest["valid"].sum()),
        "n_invalid": int((~manifest["valid"]).sum()),
        "n_kept": int(manifest["keep"].sum()),
        "n_removed": int((~manifest["keep"]).sum()),
        "n_exact_duplicate_groups": int(len(duplicate_hashes)),
        "n_exact_duplicate_images_beyond_first": int(manifest["is_exact_duplicate"].sum()),
        "n_conflicting_duplicate_groups": int(len(conflicting_hashes)),
        "n_conflicting_duplicate_images": int(manifest["duplicate_label_conflict"].sum()),
        "removal_reason_distribution": _counts(manifest["removal_reason"]),
        "source_class_distribution": _counts(manifest["source_class"]),
        "binary_label_distribution": _counts(manifest["label"]),
        "kept_source_class_distribution": _counts(kept["source_class"]),
        "kept_binary_label_distribution": _counts(kept["label"]),
        "formats": _counts(valid["format"]),
        "modes": _counts(valid["mode"]),
        "width_summary": _describe(valid["width"]),
        "height_summary": _describe(valid["height"]),
        "aspect_ratio_summary": _describe(valid["aspect_ratio"]),
        "pixel_count_summary": _describe(valid["pixel_count"]),
        "file_size_summary": _describe(valid["file_size"]),
    }
    _atomic_json(stats, output_dir / "stats.json")

    stats_per_class = {}
    for label, name in ((0, "real"), (1, "ai_generated")):
        class_all = manifest[manifest["label"] == label]
        class_valid = class_all[class_all["valid"]]
        class_kept = class_all[class_all["keep"]]
        stats_per_class[name] = {
            "n_total": int(len(class_all)),
            "n_valid": int(len(class_valid)),
            "n_kept": int(len(class_kept)),
            "valid_rate": float(class_all["valid"].mean()),
            "keep_rate": float(class_all["keep"].mean()),
            "width_summary": _describe(class_valid["width"]),
            "height_summary": _describe(class_valid["height"]),
            "aspect_ratio_summary": _describe(class_valid["aspect_ratio"]),
            "pixel_count_summary": _describe(class_valid["pixel_count"]),
            "file_size_summary": _describe(class_valid["file_size"]),
            "formats": _counts(class_valid["format"]),
            "modes": _counts(class_valid["mode"]),
            "top_original_dimensions": _top_dimensions(class_valid),
        }
    _atomic_json(stats_per_class, output_dir / "stats_per_class.json")

    # A compact, report-oriented summary of possible dataset shortcuts.
    shortcut_candidates = {
        "interpretation_note": (
            "Differences listed here are correlations in the dataset, not causal "
            "evidence of image authenticity. They should be discussed as possible shortcuts."
        ),
        "real": {
            "median_width": stats_per_class["real"]["width_summary"].get("50%"),
            "median_height": stats_per_class["real"]["height_summary"].get("50%"),
            "median_aspect_ratio": stats_per_class["real"]["aspect_ratio_summary"].get("50%"),
            "median_file_size": stats_per_class["real"]["file_size_summary"].get("50%"),
            "formats": stats_per_class["real"]["formats"],
            "top_original_dimensions": stats_per_class["real"]["top_original_dimensions"],
        },
        "ai_generated": {
            "median_width": stats_per_class["ai_generated"]["width_summary"].get("50%"),
            "median_height": stats_per_class["ai_generated"]["height_summary"].get("50%"),
            "median_aspect_ratio": stats_per_class["ai_generated"]["aspect_ratio_summary"].get("50%"),
            "median_file_size": stats_per_class["ai_generated"]["file_size_summary"].get("50%"),
            "formats": stats_per_class["ai_generated"]["formats"],
            "top_original_dimensions": stats_per_class["ai_generated"]["top_original_dimensions"],
        },
        "classical_baseline_connection": (
            "A classical model using global color, gradient, RGB-chroma, and FFT "
            "features can quantify how predictive such non-semantic cues are."
        ),
    }
    _atomic_json(shortcut_candidates, output_dir / "shortcut_candidates.json")

    # Figures for the report. class_distribution.png keeps the simple historical
    # filename expected by the project notes; the additional plots are diagnostic.
    plot_class_distribution(manifest, figures_dir / "class_distribution.png")
    plot_source_class_distribution(manifest, figures_dir / "source_class_distribution.png")
    plot_size_distribution(manifest, figures_dir / "image_size_distribution.png")
    plot_file_size_distribution(manifest, figures_dir / "file_size_distribution.png")
    plot_aspect_ratio_distribution(manifest, figures_dir / "aspect_ratio_distribution.png")

    elapsed = time.perf_counter() - start
    print("Cleaning completed.")
    print(f"Total images: {stats['n_total']}")
    print(f"Valid images: {stats['n_valid']}")
    print(f"Kept images: {stats['n_kept']}")
    print(f"Removed images: {stats['n_removed']}")
    print(
        "Exact duplicates: "
        f"{stats['n_exact_duplicate_groups']} groups, "
        f"{stats['n_exact_duplicate_images_beyond_first']} repeated images"
    )
    print(
        "Conflicting-label duplicate groups: "
        f"{stats['n_conflicting_duplicate_groups']}"
    )
    print(f"Elapsed {elapsed:.1f}s / timeout {args.timeout_seconds}s")


if __name__ == "__main__":
    main()