# clean.py

import argparse
import io
import json
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


MIN_SIDE = 32


def get_label_column(df):
    if "source_class" in df.columns:
        return "source_class"
    if "source class" in df.columns:
        return "source class"
    raise ValueError(f"Label column not found. Columns are: {list(df.columns)}")


def get_image_bytes(x):
    # Normal case: image column contains bytes
    if isinstance(x, bytes):
        return x

    # Some parquet formats store binary data as dict-like objects
    if isinstance(x, dict) and "bytes" in x:
        return x["bytes"]

    return None


def inspect_image(image_bytes):
    """
    Try to decode image and extract simple metadata.
    """
    if image_bytes is None:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "file_size": None,
            "sha256": None,
            "error": "missing bytes",
        }

    sha = hashlib.sha256(image_bytes).hexdigest()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()

        width, height = img.size

        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": img.format,
            "mode": img.mode,
            "file_size": len(image_bytes),
            "sha256": sha,
            "error": "",
        }

    except Exception as e:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "file_size": len(image_bytes),
            "sha256": sha,
            "error": str(e),
        }

def plot_class_distribution(df, output_path):
    counts = df["source_class"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Class distribution")
    plt.xlabel("source_class")
    plt.ylabel("count")

def plot_size_distribution(df, output_path):
    valid = df[df["valid"]]

    real = valid[valid["label"] == 0]
    ai = valid[valid["label"] == 1]

    plt.figure(figsize=(6, 5))
    plt.scatter(real["width"], real["height"], s=8, alpha=0.4, label="real")
    plt.scatter(ai["width"], ai["height"], s=8, alpha=0.4, label="ai_generated")

    plt.title("Image size distribution")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_size_distribution(df, output_path):
    valid = df[df["valid"]]

    plt.figure(figsize=(6, 5))
    plt.scatter(valid["width"], valid["height"], s=8, alpha=0.4)
    plt.title("Image size distribution")
    plt.xlabel("width")
    plt.ylabel("height")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

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

        for row_idx, row in df.iterrows():
            image_bytes = get_image_bytes(row["image"])
            meta = inspect_image(image_bytes)

            source_class = int(row[label_col])
            binary_label = 0 if source_class == 0 else 1

            rows.append({
                "parquet_file": parquet_file.name,
                "row_idx": row_idx,
                "source_class": source_class,
                "label": binary_label,
                **meta,
            })

    manifest = pd.DataFrame(rows)

    # Deterministic cleaning rule:
    # keep only readable images with a reasonable minimum side length.
    manifest["keep"] = (
        manifest["valid"]
        & (manifest["width"] >= MIN_SIDE)
        & (manifest["height"] >= MIN_SIDE)
    )

    # Optional but useful: remove exact duplicates, keeping the first occurrence.
    duplicated = manifest.duplicated(subset=["sha256"], keep="first")
    manifest.loc[duplicated, "keep"] = False

    # Save manifest for prepare.py
    manifest.to_csv(output_dir / "cleaned_manifest.csv", index=False)

    # Save removed records for the report/debugging
    manifest[~manifest["keep"]].to_csv(output_dir / "removed_records.csv", index=False)

    # Simple statistics
    valid = manifest[manifest["valid"]]
    kept = manifest[manifest["keep"]]

    stats = {
        "n_total": int(len(manifest)),
        "n_valid": int(manifest["valid"].sum()),
        "n_kept": int(manifest["keep"].sum()),
        "n_removed": int((~manifest["keep"]).sum()),
        "source_class_distribution": manifest["source_class"].value_counts().sort_index().to_dict(),
        "binary_label_distribution": manifest["label"].value_counts().sort_index().to_dict(),
        "formats": valid["format"].value_counts().to_dict(),
        "modes": valid["mode"].value_counts().to_dict(),
        "width_summary": valid["width"].describe().to_dict(),
        "height_summary": valid["height"].describe().to_dict(),
        "file_size_summary": valid["file_size"].describe().to_dict(),
        "kept_source_class_distribution": kept["source_class"].value_counts().sort_index().to_dict(),
        "kept_binary_label_distribution": kept["label"].value_counts().sort_index().to_dict(),
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    real = manifest[manifest["label"] == 0]
    ai = manifest[manifest["label"] == 1]
    
    valid_real = real[real["valid"]]
    valid_ai = ai[ai["valid"]]
    
    stats_per_class = {
        "n_total": {
            "real": int(len(real)),
            "ai_generated": int(len(ai)),
        },
        "valid_rate": {
            "real": float(real["valid"].mean()),
            "ai_generated": float(ai["valid"].mean()),
        },
        "keep_rate": {
            "real": float(real["keep"].mean()),
            "ai_generated": float(ai["keep"].mean()),
        },
        "width_summary": {
            "real": valid_real["width"].describe().to_dict(),
            "ai_generated": valid_ai["width"].describe().to_dict(),
        },
        "height_summary": {
            "real": valid_real["height"].describe().to_dict(),
            "ai_generated": valid_ai["height"].describe().to_dict(),
        },
        "file_size_summary": {
            "real": valid_real["file_size"].describe().to_dict(),
            "ai_generated": valid_ai["file_size"].describe().to_dict(),
        },
        "formats": {
            "real": valid_real["format"].value_counts().to_dict(),
            "ai_generated": valid_ai["format"].value_counts().to_dict(),
        },
        "modes": {
            "real": valid_real["mode"].value_counts().to_dict(),
            "ai_generated": valid_ai["mode"].value_counts().to_dict(),
        },
    }
    
    with open(output_dir / "stats_per_class.json", "w") as f:
        json.dump(stats_per_class, f, indent=2)

    # Figures for report
    plot_class_distribution(manifest, figures_dir / "class_distribution.png")
    plot_size_distribution(manifest, figures_dir / "image_size_distribution.png")

    print("Cleaning completed.")
    print(f"Total images: {stats['n_total']}")
    print(f"Valid images: {stats['n_valid']}")
    print(f"Kept images: {stats['n_kept']}")
    print(f"Removed images: {stats['n_removed']}")


if __name__ == "__main__":
    main()
