"""SYNTHETIC DEFECTS ON VALIDATION IMAGES (v2, phase 2)

  python phase2.py --examples   only the example figure + parameter table (no model loaded)
  python phase2.py              baseline: the 4 frozen v1 variants on 40 good + 160 synthetic

The generator (src/synthetic.py) and all rules are fixed in decisions.md (v2, Phase 2).
Only the 40 validation images are used; no test image is loaded. Thresholds are read
from thresholds.csv, nothing is retrained. Outputs go to outputs/v2/phase2/.
Every file in outputs/results, figures, checkpoints and analysis is hashed before and
after to prove that v1 (and phase 1) is untouched.
"""
import gc
import hashlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score

from src.data import get_split, load_images
from src.metrics import pixel_metrics
from src.paths import CHECKPOINTS, FIGURES, OUTPUTS, RESULTS
from src.synthetic import TYPES, load_clean, make_synthetic

OUT = OUTPUTS / "v2" / "phase2"
LABEL = "synthetic defects on validation images"
VARIANTS = ["ae_raw", "ae_bn", "pc_raw", "pc_bn"]
RADIUS = 8                                   # same hit rule as phase 1
EXAMPLE_IMAGES = [0, 20]                     # fixed rule (decisions.md)
COLORS = {"ae_raw": "C0", "ae_bn": "C1", "pc_raw": "C2", "pc_bn": "C3"}


class _Tee:
    """Write console output to the screen and to the log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def _hashes():
    dirs = (RESULTS, FIGURES, CHECKPOINTS, OUTPUTS / "analysis")
    return {p.relative_to(OUTPUTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


def _write_csv(df, name):
    """CSV whose first line says what it is (read it back with pd.read_csv(..., comment="#"))."""
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        df.to_csv(f, index=False, lineterminator="\n")


def plot_examples(clean, images, masks, path):
    """Validation images 0 and 20, all 4 types: full image + outline, crop before / after, mask."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 8, figsize=(16, 8.6))
    for e, (i, k) in enumerate([(i, k) for i in EXAMPLE_IMAGES for k in range(len(TYPES))]):
        j = i * len(TYPES) + k
        x, m, c = images[j], masks[j], clean[i]
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = np.clip(cy - 24, 0, 208), np.clip(cx - 24, 0, 208)
        row, col = e // 2, (e % 2) * 4
        ax = axes[row, col:col + 4]
        ax[0].imshow(x, cmap="gray", vmin=0, vmax=1)
        ax[0].contour(m, levels=[0.5], colors="r", linewidths=0.8)
        ax[0].set_title(f"val {i}: {TYPES[k]} ({100 * m.mean():.2f}%)", fontsize=8)
        ax[1].imshow(c[y0:y0 + 48, x0:x0 + 48], cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("crop before", fontsize=8)
        ax[2].imshow(x[y0:y0 + 48, x0:x0 + 48], cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("crop after", fontsize=8)
        ax[3].imshow(m[y0:y0 + 48, x0:x0 + 48], cmap="gray", vmin=0, vmax=1)
        ax[3].set_title("crop mask", fontsize=8)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Synthetic defects at 256 px, as the models see them (crops 48 x 48 px)\n{LABEL}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def evaluate_variant(scores_good, maps_good, scores_syn, maps_syn, masks, types, T, T_pix):
    """All phase 2 metrics of one variant: per synthetic type and for all 160."""
    dist = np.stack([distance_transform_edt(m == 0) for m in masks])
    rows = []
    for g in TYPES + ["all synthetic"]:
        sel = np.ones(len(types), bool) if g == "all synthetic" else types == g
        s = scores_syn[sel]
        y = np.r_[np.zeros(len(scores_good)), np.ones(sel.sum())]
        pix = pixel_metrics(np.concatenate([np.zeros_like(masks[:len(maps_good)]), masks[sel]]),
                            np.concatenate([maps_good, maps_syn[sel]]), T_pix)
        hits, dists = [], []
        for mp, d in zip(maps_syn[sel], dist[sel]):
            py, px = np.unravel_index(np.argmax(mp), mp.shape)
            dists.append(d[py, px])
            hits.append(d[py, px] <= RADIUS)
        rows.append({"group": g, "n": int(sel.sum()),
                     "auroc": roc_auc_score(y, np.r_[scores_good, s]),
                     "recall": float((s > T).mean()), "detected": int((s > T).sum()),
                     "pixel_auroc": pix["pixel_auroc"], "iou": pix["iou"], "dice": pix["dice"],
                     "hit_rate": float(np.mean(hits)), "hits": int(np.sum(hits)),
                     "median_distance_px": float(np.median(dists)),
                     "chance_hit_rate": float((dist[sel] <= RADIUS).mean(axis=(1, 2)).mean())})
    return pd.DataFrame(rows)


def plot_baseline(res, path):
    import matplotlib.pyplot as plt
    groups = TYPES + ["all synthetic"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, lab in ((axes[0], "auroc", "image AUROC (40 good vs synthetic)"),
                         (axes[1], "recall", "recall at the frozen T_image"),
                         (axes[2], "hit_rate", f"hit rate: heat-map max within {RADIUS} px")):
        for k, name in enumerate(VARIANTS):
            r = res[res.variant == name].set_index("group").loc[groups]
            x = np.arange(len(groups)) + (k - 1.5) * 0.2
            ax.bar(x, r[col], 0.18, color=COLORS[name], label=name,
                   hatch="//" if name.startswith("ae") else None, edgecolor="white", linewidth=0.5)
        if col == "auroc":
            ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(np.arange(len(groups)), groups, fontsize=8, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Baseline: frozen v1 models and thresholds | {LABEL}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    examples_only = "--examples" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    log = "examples_log.txt" if examples_only else "phase2_log.txt"
    sys.stdout = _Tee(sys.__stdout__, open(OUT / log, "w", encoding="utf-8"))
    t0 = time.time()
    print(f"phase2.py {'--examples ' if examples_only else ''}started {datetime.now():%Y-%m-%d %H:%M} | {LABEL}")
    before = _hashes()
    print(f"hashed {len(before)} files in outputs/results, figures, checkpoints, analysis")

    files = get_split()["val"]
    clean = load_clean(files)
    assert np.array_equal(clean, load_images(files)), "1024 -> 256 pipeline differs from load_images"
    print(f"{len(files)} validation images; 1024 -> 256 pipeline identical to data.load_images")
    images, masks, info = make_synthetic(files)
    types = info.type.to_numpy()
    _write_csv(info.round(5), "synthetic_defects.csv")
    print(f"generated {len(images)} synthetic images in {time.time() - t0:.0f} s")
    pd.set_option("display.width", 200)
    print(info.groupby("type")[["area_pct", "max_change", "mean_change_in_mask", "draws",
                                "changed_px_outside"]].agg(["min", "median", "max"]).round(3).loc[TYPES].to_string())
    plot_examples(clean, images, masks, OUT / "synthetic_examples.png")
    print("saved synthetic_examples.png")

    if not examples_only:
        import torch
        from evaluate import _load
        th = pd.read_csv(RESULTS / "thresholds.csv").set_index("variant")
        results = []
        for name in VARIANTS:
            det = _load(name)
            T, T_pix = float(th.loc[name, "T_image"]), float(th.loc[name, "T_pixel"])
            sg, mg = det.predict_many(clean)
            ss, ms = det.predict_many(images)
            saved = pd.read_csv(RESULTS / f"{name}_val_scores.csv").score.to_numpy()
            print(f"[{name}] good val scores vs {name}_val_scores.csv: max abs diff {np.abs(sg - saved).max():.2e}"
                  f" | good above T: {(sg > T).sum()}/40")
            assert np.allclose(sg, saved, rtol=1e-5, atol=1e-6), f"{name}: validation scores differ"
            r = evaluate_variant(sg, mg, ss, ms, masks, types, T, T_pix)
            r.insert(0, "variant", name)
            results.append(r)
            info[f"{name}_score_over_T"] = ss / T
            del det
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        res = pd.concat(results, ignore_index=True)
        _write_csv(res, "baseline_metrics.csv")
        _write_csv(info.round(5), "synthetic_defects.csv")       # now with the scores / T
        for col in ("auroc", "recall", "pixel_auroc", "iou", "dice", "hit_rate", "median_distance_px"):
            print(f"\n{col}:")
            print(res.pivot(index="group", columns="variant", values=col)
                  .loc[TYPES + ["all synthetic"], VARIANTS].round(3).to_string())
        print("\nchance hit rate (random peak):",
              res[res.variant == "pc_raw"].set_index("group").chance_hit_rate.round(3).to_dict())
        real = pd.read_csv(RESULTS / "metrics.csv").set_index("variant")
        print("\nobservation only (not a target): real test set vs all synthetic")
        print(pd.DataFrame({"real AUROC": real.loc[VARIANTS, "auroc"],
                            "synthetic AUROC": res[res.group == "all synthetic"].set_index("variant").auroc,
                            "real recall": real.loc[VARIANTS, "recall"],
                            "synthetic recall": res[res.group == "all synthetic"].set_index("variant").recall})
              .round(3).to_string())
        plot_baseline(res, OUT / "baseline_metrics.png")

    after = _hashes()
    check = pd.DataFrame([{"file": f, "sha256_before": h, "sha256_after": after.get(f, ""),
                           "unchanged": h == after.get(f)} for f, h in before.items()])
    _write_csv(check, "v1_hash_check.csv")
    assert set(after) == set(before) and check.unchanged.all(), "a frozen file changed!"
    print(f"\nv1 untouched: all {len(before)} files have identical sha256 before and after")
    print(f"run time: {time.time() - t0:.0f} s")
