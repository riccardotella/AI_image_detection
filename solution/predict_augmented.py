"""Task 3 inference."""

import argparse

from common import run_prediction

parser = argparse.ArgumentParser()
parser.add_argument("--timeout_seconds", type=int, default=600)
parser.parse_args()
run_prediction("model_aug.pt", "threshold_aug.json", "task03")
