# prepare.py — build the cleaned training cache used by train.py.
# Reads ONLY data/train/ (never data/predict/); writes ONLY under artifacts/.

import argparse
import time

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, DATA_DIR, _decode, _image_bytes, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()
    start = time.perf_counter()

    set_seed()

    manifest_path = ARTIFACTS_DIR / "clean" / "cleaned_manifest.csv"
    out_dir = ARTIFACTS_DIR / "prepared"
    out_path = out_dir / "train.npz"

    if out_path.exists():
        print(f"{out_path} already exists, skipping.")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found - run clean.py first")

    manifest = pd.read_csv(manifest_path)
    kept = manifest[manifest["keep"]].sort_values(["parquet_file", "row_idx"])

    images, labels = [], []
    for parquet_file, group in kept.groupby("parquet_file", sort=True):
        df = pd.read_parquet(DATA_DIR / "train" / parquet_file, columns=["image"])
        for row_idx, label in zip(group["row_idx"], group["label"]):
            images.append(_decode(_image_bytes(df["image"].iloc[int(row_idx)])))
            labels.append(label)

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int8)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first so a timeout kill cannot leave a corrupt cache
    # that the skip-if-exists check would then trust.
    tmp_path = out_dir / "train.npz.tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, X=X, y=y)
    tmp_path.rename(out_path)

    print(f"Prepared {len(y)} images ({int((y == 0).sum())} real, "
          f"{int((y == 1).sum())} ai) -> {out_path}")
    print(f"Elapsed {time.perf_counter() - start:.1f}s / budget {args.timeout_seconds}s")


if __name__ == "__main__":
    main()
