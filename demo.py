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

v2: --category screw (default) is exactly v1. Another category (e.g. transistor) reads
its own models, thresholds and data folder and saves to outputs/<category>/figures.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.autoencoder import AutoencoderDetector
from src.categories import get_category
from src.data import get_split, load_image, load_mask, to_display
from src.patchcore import PatchCoreDetector
from src.paths import ROOT

MODELS = {"pc_raw": (PatchCoreDetector, "patchcore_bank.pt", "PatchCore (main model)"),
          "ae_raw": (AutoencoderDetector, "autoencoder.pt", "Autoencoder")}
_cache = {}                                       # per category: models and thresholds, loaded once


def _frozen(category="screw"):
    """Load both checkpoints and the frozen thresholds of a category (stop if anything is missing)."""
    if category not in _cache:
        cat = get_category(category)
        th_file = cat["results"] / "thresholds.csv"
        for f in [th_file] + [cat["checkpoints"] / m[1] for m in MODELS.values()]:
            if not f.exists():
                raise FileNotFoundError(f"{f} is missing. Run the earlier steps; demo.py never refits.")
        th = pd.read_csv(th_file)                      # screw (v1): column "variant"; transistor: "model"
        fz = {"th": th.set_index("variant" if "variant" in th.columns else "model")}
        for name, (cls, file, _) in MODELS.items():
            det = cls.load(cat["checkpoints"] / file)
            assert det.config["normalize_brightness"] is False, f"{file} is not the raw variant"
            assert det.config["category"] == category, f"{file} belongs to {det.config['category']}"
            fz[name] = det
        _cache[category] = fz
    return _cache[category]


def _resolve(image_path):
    """Path relative to the current folder, else to the project root, or absolute."""
    p = Path(image_path)
    if not p.is_absolute():
        p = p if p.exists() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)
    return p.resolve()


def _source(p, category="screw"):
    """Where the image comes from (train / val / test / outside) and its path inside the data folder."""
    try:
        rel = p.relative_to(get_category(category)["data"].resolve()).as_posix()
    except ValueError:
        return "outside the dataset", None
    split = get_split(category)
    if rel in split["train"]:
        return "train (seen by the models)", rel
    if rel in split["val"]:
        return "validation (used to set T)", rel
    if rel.startswith("test/"):
        return f"test / {rel.split('/')[1]}", rel
    return "dataset (other folder)", rel


def _gt(rel, category="screw"):
    """GT mask for test images (zeros for test/good); None if not available."""
    if rel is None or not rel.startswith("test/"):
        return None
    defect = rel.split("/")[1]
    return load_mask(None if defect == "good" else f"ground_truth/{defect}/{Path(rel).stem}_mask.png",
                     category=category)


def inspect(image_path, show=True, save=True, category="screw"):
    import matplotlib.pyplot as plt
    fz = _frozen(category)
    p = _resolve(image_path)
    source, rel = _source(p, category)
    img = load_image(p, category=category)      # data folder / absolute path = absolute path
    gt = _gt(rel, category)
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

        row[0].imshow(to_display(img), cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"{label}\nscore {score:.4g} vs T {T:.4g} (x{score / T:.2f}) -> {decision}",
                         fontsize=9, color="red" if decision == "DEFECTIVE" else "green")
        # fixed colour scale from the frozen T_pixel, so different images are comparable
        im = row[1].imshow(heatmap, cmap="jet", vmin=0.5 * T_pix, vmax=1.5 * T_pix)
        cb = fig.colorbar(im, ax=row[1], fraction=0.046)
        cb.set_ticks([0.5 * T_pix, T_pix, 1.5 * T_pix])
        cb.set_ticklabels(["0.5 T_pix", "T_pix", "1.5 T_pix"])
        row[1].set_title("heat-map", fontsize=9)
        row[2].imshow(to_display(img), cmap="gray", vmin=0, vmax=1)
        row[2].imshow(np.ma.masked_where(~region, region), cmap="autumn", alpha=0.6)
        if gt is not None and gt.any():
            row[2].contour(gt, levels=[0.5], colors="lime", linewidths=1)
        row[2].set_title(region_title, fontsize=9)
        row[3].imshow(to_display(img), cmap="gray", vmin=0, vmax=1)
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
        figures = get_category(category)["figures"]
        figures.mkdir(parents=True, exist_ok=True)
        out["figure"] = figures / f"demo_{p.parent.name}_{p.stem}.png"
        fig.savefig(out["figure"], dpi=90)
        print("  saved", out["figure"])
    if show:
        plt.show()
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect images with the frozen models.")
    parser.add_argument("paths", nargs="+", help="image path(s), e.g. screw/test/scratch_neck/004.png")
    parser.add_argument("--no-show", action="store_true", help="only save the figure, don't open a window")
    parser.add_argument("--category", default="screw", help="MVTec category (default screw = v1)")
    args = parser.parse_args()
    for path in args.paths:
        inspect(path, show=not args.no_show, category=args.category)
