"""Demo: inspect screw images with both frozen models.

    python demo.py screw/test/scratch_neck/004.png [more paths ...] [--no-show]

or in Python / a notebook:
    from demo import inspect
    inspect("screw/test/scratch_neck/004.png")

FROZEN ONLY: the checkpoints are loaded (never refitted) and T_image / T_pixel are
read from outputs/results/thresholds.csv (never recomputed). Nothing in
outputs/results is written. For pc_raw (main model) and ae_raw it shows: score vs
T_image, the GOOD/DEFECTIVE decision (score > T_image, as in evaluate.py), the
heat-map, the predicted region (heat-map > T_pixel) and, for test images, the true
defect mask. Figure: outputs/figures/demo_<folder>_<file>.png.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.autoencoder import AutoencoderDetector
from src.data import DATA_DIR, get_split, load_image, load_mask
from src.patchcore import PatchCoreDetector
from src.paths import CHECKPOINTS, FIGURES, RESULTS, ROOT

MODELS = {"pc_raw": (PatchCoreDetector, "patchcore_bank.pt", "PatchCore (main model)"),
          "ae_raw": (AutoencoderDetector, "autoencoder.pt", "Autoencoder")}
_cache = {}                                       # models and thresholds are loaded only once


def _frozen():
    """Load both checkpoints and the frozen thresholds (stop if anything is missing)."""
    if not _cache:
        th_file = RESULTS / "thresholds.csv"
        for f in [th_file] + [CHECKPOINTS / m[1] for m in MODELS.values()]:
            if not f.exists():
                raise FileNotFoundError(f"{f} is missing. Run the earlier steps; demo.py never refits.")
        _cache["th"] = pd.read_csv(th_file).set_index("variant")
        for name, (cls, file, _) in MODELS.items():
            det = cls.load(CHECKPOINTS / file)
            assert det.config["normalize_brightness"] is False, f"{file} is not the raw variant"
            _cache[name] = det
    return _cache


def _resolve(image_path):
    """Path relative to the current folder, else to the project root, or absolute."""
    p = Path(image_path)
    if not p.is_absolute():
        p = p if p.exists() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    return p.resolve()


def _source(p):
    """Where the image comes from (train / val / test / outside) and its path inside screw/."""
    try:
        rel = p.relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return "outside the dataset", None
    split = get_split()
    if rel in split["train"]:
        return "train (seen by the models)", rel
    if rel in split["val"]:
        return "validation (used to set T)", rel
    if rel.startswith("test/"):
        return f"test / {rel.split('/')[1]}", rel
    return "dataset (other folder)", rel


def _gt(rel):
    """GT mask for test images (zeros for test/good); None if not available."""
    if rel is None or not rel.startswith("test/"):
        return None
    defect = rel.split("/")[1]
    return load_mask(None if defect == "good" else f"ground_truth/{defect}/{Path(rel).stem}_mask.png")


def inspect(image_path, show=True, save=True):
    import matplotlib.pyplot as plt
    fz = _frozen()
    p = _resolve(image_path)
    source, rel = _source(p)
    img = load_image(p)                         # DATA_DIR / absolute path = absolute path
    gt = _gt(rel)
    print(f"\n{p.name} ({p.parent.name}): {source}")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    out = {"source": source, "figure": None}
    for row, (name, (_, _, label)) in zip(axes, MODELS.items()):
        T, T_pix = fz["th"].loc[name, "T_image"], fz["th"].loc[name, "T_pixel"]
        score, heatmap = fz[name].predict(img)
        decision = "DEFECTIVE" if score > T else "GOOD"
        region = heatmap > T_pix
        res = {"score": score, "T_image": T, "decision": decision, "heatmap": heatmap, "region": region}
        line = (f"  {name}: score {score:.4g}  T {T:.4g}  (score/T {score / T:.2f})  -> {decision}"
                f"  | predicted pixels {int(region.sum())}")
        region_title = f"predicted region (> T_pixel {T_pix:.3g})"
        if gt is not None:
            truth = "DEFECTIVE" if gt.any() else "GOOD"
            res["correct"] = decision == truth
            line += f"  | truth {truth} -> {'correct' if res['correct'] else 'WRONG'}"
            if gt.any():
                inter, union = (region & (gt == 1)).sum(), (region | (gt == 1)).sum()
                res["iou"], res["dice"] = inter / union, 2 * inter / (region.sum() + gt.sum())
                line += f"  | IoU {res['iou']:.3f} Dice {res['dice']:.3f}"
                region_title += f"\nIoU {res['iou']:.3f}, Dice {res['dice']:.3f}"
        print(line)
        out[name] = res

        row[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"{label}\nscore {score:.4g} vs T {T:.4g} (x{score / T:.2f}) -> {decision}",
                         fontsize=9, color="red" if decision == "DEFECTIVE" else "green")
        # fixed colour scale from the frozen T_pixel, so different images are comparable
        im = row[1].imshow(heatmap, cmap="jet", vmin=0.5 * T_pix, vmax=1.5 * T_pix)
        cb = fig.colorbar(im, ax=row[1], fraction=0.046)
        cb.set_ticks([0.5 * T_pix, T_pix, 1.5 * T_pix])
        cb.set_ticklabels(["0.5 T_pix", "T_pix", "1.5 T_pix"])
        row[1].set_title("heat-map", fontsize=9)
        row[2].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[2].imshow(np.ma.masked_where(~region, region), cmap="autumn", alpha=0.6)
        if gt is not None and gt.any():
            row[2].contour(gt, levels=[0.5], colors="lime", linewidths=1)
        row[2].set_title(region_title, fontsize=9)
        row[3].imshow(img, cmap="gray", vmin=0, vmax=1)
        if gt is None:
            row[3].set_title("GT: not available", fontsize=9)
        elif gt.any():
            row[3].imshow(np.ma.masked_where(gt == 0, gt), cmap="winter", alpha=0.6)
            row[3].set_title("GT mask (true defect)", fontsize=9)
        else:
            row[3].set_title("GT: good, no defect", fontsize=9)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"{rel or p.name}  -  {source}")
    fig.tight_layout()
    if save:
        out["figure"] = FIGURES / f"demo_{p.parent.name}_{p.stem}.png"
        fig.savefig(out["figure"], dpi=90)
        print("  saved", out["figure"])
    if show:
        plt.show()
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect screw images with the frozen models.")
    parser.add_argument("paths", nargs="+", help="image path(s), e.g. screw/test/scratch_neck/004.png")
    parser.add_argument("--no-show", action="store_true", help="only save the figure, don't open a window")
    args = parser.parse_args()
    for path in args.paths:
        inspect(path, show=not args.no_show)
