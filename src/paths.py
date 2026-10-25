"""All project paths in one place.

Every other file does `from src.paths import DATA_DIR, FIGURES, ...`
instead of typing a path. ROOT is found from this file's own location,
so the code works on any computer and with spaces in the folder name.
"""
from pathlib import Path

# This file is <ROOT>/src/paths.py, so ROOT is two levels up.
ROOT = Path(__file__).resolve().parent.parent

# Dataset (READ-ONLY: we never write into it).
DATA_DIR = ROOT / "screw"

# Everything we produce goes under outputs/.
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"          # small .png plots, committed
RESULTS = OUTPUTS / "results"          # small .csv/.json files, committed
CHECKPOINTS = OUTPUTS / "checkpoints"  # model weights, big arrays, git-ignored

# Create the output folders once, when this module is first imported.
for folder in (FIGURES, RESULTS, CHECKPOINTS):
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick self-check: python -m src.paths
    for name in ("ROOT", "DATA_DIR", "OUTPUTS", "FIGURES", "RESULTS", "CHECKPOINTS"):
        p = globals()[name]
        print(f"{name:12s} exists={p.exists()!s:5s} {p}")
    n_train = len(list((DATA_DIR / "train" / "good").glob("*.png")))
    print("train/good images:", n_train)   # expect 320
