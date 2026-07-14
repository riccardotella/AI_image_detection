"""Build the cleaned 96x96 uint8 training cache."""

import argparse
import time

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, DATA_DIR, decode_image, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()
    start = time.perf_counter()
    set_seed()

    manifest_path = ARTIFACTS_DIR / "clean" / "cleaned_manifest.csv"
    output = ARTIFACTS_DIR / "prepared" / "train.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run clean.py first: {manifest_path} is missing")

    manifest = pd.read_csv(manifest_path)
    kept = manifest[manifest["keep"]].sort_values(["parquet_file", "row_idx"])

    images, labels = [], []
    for file_name, group in kept.groupby("parquet_file", sort=True):
        frame = pd.read_parquet(DATA_DIR / "train" / file_name, columns=["image"])
        for row_index, label in zip(group["row_idx"], group["label"]):
            images.append(decode_image(frame["image"].iloc[int(row_index)]))
            labels.append(label)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=np.stack(images),
            y=np.asarray(labels, dtype=np.int8),
        )
    temporary.replace(output)

    labels = np.asarray(labels)
    print(
        f"Prepared {len(labels)} images "
        f"({int((labels == 0).sum())} real, {int((labels == 1).sum())} AI) "
        f"in {time.perf_counter() - start:.1f}s / timeout {args.timeout_seconds}s"
    )


if __name__ == "__main__":
    main()
