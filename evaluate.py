"""Final evaluation on the 160 test images (step 7). Run ONCE: python evaluate.py

Everything is fixed a priori, BEFORE any test image is loaded:
  - 4 variants at 256 px: ae_raw, ae_bn (autoencoder without / with brightness
    normalisation) and pc_raw, pc_bn (PatchCore). Missing checkpoints are built
    by their own module (train/val only); existing ones are never retrained.
  - thresholds from each variant's 40 VALIDATION images only:
    T_image = 99th percentile of the val scores, T_pixel = 99.9th percentile of
    all val heat-map pixels.
  - perturbations for the robustness test, applied to the 256 px [0, 1] images
    before predict (the bn variants then still apply their own brightness shift),
    clipped to [0, 1], identical for every variant:
      brightness_minus  x - 0.05
      brightness_plus   x + 0.05
      contrast_low      mean + 0.8 * (x - mean)
      contrast_high     mean + 1.2 * (x - mean)
      rotate_10         10 deg about the centre, bilinear, corners = image's median grey
      noise             Gaussian, sigma 0.02, seed 42 + image index

Phase A  val only: thresholds + val robustness      -> results/thresholds.csv, <variant>_val_scores.csv
Phase B  the ONLY place test images are scored      -> results/<variant>_scores.csv,
                                                       checkpoints/<variant>_heatmaps.npz, results/robustness.csv
Phase C  report from the saved files (no model)     -> results/metrics.csv, per_defect.csv, figures/eval_*.png
Console output is also written to results/evaluate_log.txt.

If a figure needs fixing after the run: python evaluate.py --report-only
(redoes Phase C from the saved scores; no model or threshold is touched; it prints to
the screen only, so the log of the official run stays as it was).

Safe re-run (the official files are never touched): python evaluate.py --out-dir outputs/rerun
  Same code, settings and threshold rules, but every output (thresholds, scores, heat-maps,
  metrics, figures, log) goes to <out-dir>/results, <out-dir>/figures, <out-dir>/checkpoints.
  Models are read from outputs/checkpoints; a missing one is built from train/val only and
  saved in <out-dir>/checkpoints. The folder must not be (or contain, or be inside) an
  official folder; a folder that already holds a finished run is refused.
  python evaluate.py --report-only --out-dir outputs/rerun redoes Phase C of that folder.
"""
import argparse
import gc
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate as nd_rotate

from src.autoencoder import AutoencoderDetector
from src.data import DEFECT_TYPES, get_split, list_test, load_images, load_mask
from src.metrics import (Q_IMAGE, Q_PIXEL, image_metrics, per_defect_table, pixel_metrics,
                         plot_roc, threshold_from_val)
from src.patchcore import PatchCoreDetector
from src.paths import CHECKPOINTS, DATA_DIR, FIGURES, OUTPUTS, RESULTS, ROOT

# name: (class, checkpoint, module that builds it, normalize_brightness)
VARIANTS = {
    "ae_raw": (AutoencoderDetector, "autoencoder.pt", "src.autoencoder", False),
    "ae_bn": (AutoencoderDetector, "autoencoder_bn.pt", "src.autoencoder", True),
    "pc_raw": (PatchCoreDetector, "patchcore_bank.pt", "src.patchcore", False),
    "pc_bn": (PatchCoreDetector, "patchcore_bank_bn.pt", "src.patchcore", True),
}

# Where this run writes: the official folders, unless --out-dir moves all three
# into one separate folder (OUT_DIR), so a re-run never touches the official files.
OUT_DIR = None
RES_DIR, FIG_DIR, HEAT_DIR = RESULTS, FIGURES, CHECKPOINTS


# ---- perturbations (fixed a priori, see docstring) -------------------------------
def _contrast(x, k):
    m = x.mean(axis=(1, 2), keepdims=True)
    return np.clip(m + k * (x - m), 0, 1)


def _rotate10(x):
    return np.stack([nd_rotate(img, 10, reshape=False, order=1, mode="constant",
                               cval=float(np.median(img))) for img in x]).astype(np.float32)


def _noise(x):
    return np.stack([np.clip(img + np.random.default_rng(42 + i).normal(0, 0.02, img.shape), 0, 1)
                     for i, img in enumerate(x)]).astype(np.float32)


PERTURBATIONS = {
    "brightness_minus": lambda x: np.clip(x - 0.05, 0, 1),
    "brightness_plus": lambda x: np.clip(x + 0.05, 0, 1),
    "contrast_low": lambda x: _contrast(x, 0.8),
    "contrast_high": lambda x: _contrast(x, 1.2),
    "rotate_10": _rotate10,
    "noise": _noise,
}


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


def _model_path(name):
    """The official checkpoint; in an --out-dir run, a missing one lives in <out-dir>/checkpoints."""
    official = CHECKPOINTS / VARIANTS[name][1]
    return official if official.exists() or OUT_DIR is None else HEAT_DIR / official.name


def _load(name):
    cls, file, _, bn = VARIANTS[name]
    det = cls.load(_model_path(name))
    assert det.config["normalize_brightness"] == bn, f"{file}: wrong normalize_brightness"
    return det


def _free(det):
    del det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build(name, path):
    """--out-dir only: build a missing model from the 280 train / 40 val images like its module
    (python -m src.autoencoder / src.patchcore) does, but save it only to `path`
    (the module would also write its figures and history into outputs/)."""
    cls, _, _, bn = VARIANTS[name]
    split = get_split()
    train = load_images(split["train"])
    if cls is AutoencoderDetector:
        val = load_images(split["val"])
        try:
            det = cls(normalize_brightness=bn).fit(train, val)
        except torch.cuda.OutOfMemoryError:                  # same fallback as src/autoencoder.py
            torch.cuda.empty_cache()
            print("out of GPU memory, retrying with batch size 4")
            det = cls(normalize_brightness=bn, batch_size=4).fit(train, val)
    else:
        det = cls(normalize_brightness=bn).fit(train)
    det.save(path)
    _free(det)


# ---- Phase A: validation only -----------------------------------------------------
def phase_a():
    split = get_split()
    val_files, val_images = split["val"], load_images(split["val"])
    val_pert = {p: f(val_images) for p, f in PERTURBATIONS.items()}
    frozen, val_rob = [], {}
    for name, (_, file, module, bn) in VARIANTS.items():
        path = _model_path(name)
        if not path.exists() and OUT_DIR is not None:
            print(f"[{name}] {file} missing in {CHECKPOINTS} -> building it (train/val only), "
                  f"saved only to {path}")
            _build(name, path)
        elif not path.exists():
            print(f"[{name}] {file} missing -> building it with `python -m {module}` (train/val only)")
            r = subprocess.run([sys.executable, "-m", module] + (["--normalize-brightness"] if bn else []),
                               cwd=ROOT, capture_output=True, text=True)
            print(r.stdout)
            if r.returncode:
                print(r.stderr)
                sys.exit(f"building {file} failed")
        det = _load(name)
        print(f"[{name}] {path} (modified {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M})")
        s, m = det.predict_many(val_images)
        pd.DataFrame({"file": val_files, "score": s}).to_csv(RES_DIR / f"{name}_val_scores.csv", index=False)
        frozen.append({"variant": name, "T_image": threshold_from_val(s, Q_IMAGE),
                       "T_pixel": threshold_from_val(m, Q_PIXEL),
                       "val_min": float(s.min()), "val_median": float(np.median(s)),
                       "val_max": float(s.max()), "val_heatmap_median": float(np.median(m))})
        val_rob[name] = {"clean": s, **{p: det.predict_many(x)[0] for p, x in val_pert.items()}}
        _free(det)
    th = pd.DataFrame(frozen)
    th.to_csv(RES_DIR / "thresholds.csv", index=False)
    print(th.to_string(index=False))
    print("THRESHOLDS FROZEN\n")
    return th.set_index("variant"), val_rob


# ---- Phase B: the only place test images are scored ------------------------------
def phase_b(th, val_rob):
    test = list_test()
    files = [t["file"] for t in test]
    labels = np.array([t["label"] for t in test])
    images = load_images(files)
    test_pert = {p: f(images) for p, f in PERTURBATIONS.items()}
    print(f"loaded {len(files)} test images")
    rows = []
    for name in VARIANTS:
        T = th.loc[name, "T_image"]
        det = _load(name)
        t0 = time.time()
        s, m = det.predict_many(images)
        print(f"[{name}] scored the test set in {time.time() - t0:.1f} s")
        df = pd.DataFrame({"file": files, "defect": [t["defect"] for t in test], "label": labels, "score": s})
        df["pred"] = (df.score > T).astype(int)
        df.to_csv(RES_DIR / f"{name}_scores.csv", index=False)
        np.savez_compressed(HEAT_DIR / f"{name}_heatmaps.npz", heatmaps=m.astype(np.float32),
                            files=np.array(files))
        del m
        v0 = val_rob[name]["clean"]
        for p in ["clean"] + list(PERTURBATIONS):
            ts = s if p == "clean" else det.predict_many(test_pert[p])[0]
            im, vs = image_metrics(labels, ts, T), val_rob[name][p]
            rows.append({"variant": name, "perturbation": p,
                         "val_mean_rel_change": float(np.mean((vs - v0) / v0)),
                         "val_frr": float((vs > T).mean()), "test_auroc": im["auroc"],
                         "test_frr": im["false_reject_rate"], "test_recall": im["recall"]})
        _free(det)
    pd.DataFrame(rows).to_csv(RES_DIR / "robustness.csv", index=False)
    print()


# ---- Phase C: report from saved files ---------------------------------------------
def _examples(df, T):
    """Fixed rules: highest FP, lowest FN, median-score TP of each defect type."""
    good, bad = df[df.label == 0], df[df.label == 1]
    fp, fn = good[good.score > T], bad[bad.score <= T]
    rows = [("false positive", fp.score.idxmax()) if len(fp) else ("no FP", good.score.idxmax()),
            ("missed defect", fn.score.idxmin()) if len(fn) else ("no FN", bad.score.idxmin())]
    for d in DEFECT_TYPES:
        tp = df[(df.defect == d) & (df.score > T)].sort_values("score")
        rows.append((f"TP {d} (median)", tp.index[len(tp) // 2]) if len(tp)
                    else (f"{d}: missed", df[df.defect == d].score.idxmax()))
    return rows


def _plot_examples(name, df, maps, images, masks, t):
    import matplotlib.pyplot as plt
    rows = _examples(df, t.T_image)
    vmin = t.val_heatmap_median
    vmax = max(float(maps[i].max()) for _, i in rows)
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.1 * len(rows)))
    for ax, (title, i) in zip(axes, rows):
        decision = "defective" if df.score[i] > t.T_image else "good"
        ax[0].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax[0].set_title(f"{title}\n{df.file[i]}\nscore {df.score[i]:.4g}, T {t.T_image:.4g} -> {decision}",
                        fontsize=8)
        ax[1].imshow(maps[i], cmap="jet", vmin=vmin, vmax=vmax)
        ax[1].set_title("heat-map", fontsize=8)
        pred = maps[i] > t.T_pixel
        ax[2].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax[2].imshow(np.ma.masked_where(~pred, pred), cmap="autumn", alpha=0.6)
        ax[2].set_title("predicted region (heat-map > T_pixel)", fontsize=8)
        ax[3].imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax[3].imshow(np.ma.masked_where(masks[i] == 0, masks[i]), cmap="winter", alpha=0.6)
        ax[3].set_title("GT mask", fontsize=8)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"{name}: examples chosen by fixed rules")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"eval_examples_{name}.png", dpi=70)
    plt.close(fig)


def phase_c():
    import matplotlib.pyplot as plt
    th = pd.read_csv(RES_DIR / "thresholds.csv").set_index("variant")
    test = list_test()
    files = [t["file"] for t in test]
    labels = np.array([t["label"] for t in test])
    images = load_images(files)
    masks = np.stack([load_mask(t["mask"]) for t in test])
    results = {}
    for name in VARIANTS:
        df = pd.read_csv(RES_DIR / f"{name}_scores.csv")
        assert list(df.file) == files
        results[name] = (df, np.load(HEAT_DIR / f"{name}_heatmaps.npz")["heatmaps"])

    # 8. metrics
    metric_rows, defect_tables = [], []
    for name, (df, maps) in results.items():
        t = th.loc[name]
        row = {"variant": name, **image_metrics(labels, df.score, t.T_image),
               **pixel_metrics(masks, maps, t.T_pixel)}
        row.pop("T"), row.pop("T_pix")
        metric_rows.append({**row, "T_image": t.T_image, "T_pixel": t.T_pixel})
        table = per_defect_table(df, t.T_image).assign(variant=name, iou=np.nan, dice=np.nan)
        for d in DEFECT_TYPES:
            idx = (df.defect == d).to_numpy()
            px = pixel_metrics(masks[idx], maps[idx], t.T_pixel)
            table.loc[table.defect == d, "iou"] = px["iou"]
            table.loc[table.defect == d, "dice"] = px["dice"]
        defect_tables.append(table)
    metrics = pd.DataFrame(metric_rows)
    per_defect = pd.concat(defect_tables)[["variant", "defect", "n", "flagged", "recall",
                                           "auroc_vs_good", "iou", "dice"]]
    metrics.to_csv(RES_DIR / "metrics.csv", index=False)
    per_defect.to_csv(RES_DIR / "per_defect.csv", index=False)

    # 9. ROC
    plot_roc(labels, {name: df.score for name, (df, _) in results.items()}, FIG_DIR / "eval_roc.png")

    # 10. score histograms
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, (df, _)) in zip(axes.flat, results.items()):
        val = pd.read_csv(RES_DIR / f"{name}_val_scores.csv").score
        allv = np.r_[df.score, val]
        bins = np.linspace(allv.min(), allv.max(), 40)
        ax.hist(df.score[df.label == 0], bins=bins, alpha=0.6, label="test good (41)")
        ax.hist(df.score[df.label == 1], bins=bins, alpha=0.6, label="test defective (119)")
        ax.hist(val, bins=bins, histtype="step", color="k", label="validation good (40)")
        ax.axvline(th.loc[name, "T_image"], color="r", ls="--", label="T_image")
        ax.set_title(name)
        ax.set_xlabel("anomaly score")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_score_hist.png", dpi=100)
    plt.close(fig)

    # 11. examples
    for name in ("ae_raw", "pc_raw"):
        df, maps = results[name]
        _plot_examples(name, df, maps, images, masks, th.loc[name])

    # 12. score vs defect area
    bad = [i for i, t in enumerate(test) if t["label"] == 1]
    area = np.array([100 * load_mask(test[i]["mask"], size=1024).mean() for i in bad])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("ae_raw", "pc_raw")):
        df = results[name][0]
        rel = df.score.to_numpy()[bad] / th.loc[name, "T_image"]
        for d in DEFECT_TYPES:
            sel = np.array([test[i]["defect"] == d for i in bad])
            ax.scatter(area[sel], rel[sel], s=18, label=d)
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("defect area (% of image, 1024 px mask)")
        ax.set_ylabel("score / T_image  (> 1 = detected)")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_score_vs_area.png", dpi=100)
    plt.close(fig)

    # 13. summary
    pd.set_option("display.width", 220)
    print("== SUMMARY (test set, thresholds from validation) ==")
    cols = ["variant", "T_image", "auroc", "precision", "recall", "f1", "false_reject_rate",
            "missed_defect_rate", "TN", "FP", "FN", "TP", "pixel_auroc", "iou", "dice"]
    print(metrics[cols].round(3).to_string(index=False))
    print("\nrecall per defect type:")
    pd_bad = per_defect[per_defect.defect != "good"]
    print(pd_bad.pivot(index="defect", columns="variant", values="recall")[list(VARIANTS)].round(3).to_string())
    print("\nAUROC per defect type (vs test good):")
    print(pd_bad.pivot(index="defect", columns="variant", values="auroc_vs_good")[list(VARIANTS)]
          .round(3).to_string())
    rob = pd.read_csv(RES_DIR / "robustness.csv")
    order = ["clean"] + list(PERTURBATIONS)
    for col in ("test_auroc", "test_frr", "val_frr", "val_mean_rel_change"):
        print(f"\nrobustness: {col}")
        print(rob.pivot(index="perturbation", columns="variant", values=col).loc[order, list(VARIANTS)]
              .round(3).to_string())


def _check_out_dir(out):
    """--out-dir must be a separate folder: not an official folder or the dataset, not inside
    one of them, and not a folder that contains them (such as outputs/ or the project root)."""
    for p in (RESULTS, FIGURES, CHECKPOINTS, DATA_DIR):
        p = p.resolve()
        if out == p or p in out.parents or out in p.parents:
            sys.exit(f"--out-dir {out} is not allowed: it overlaps {p}. Use a new folder, e.g. outputs/rerun")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final, one-time evaluation on the test set.")
    parser.add_argument("--report-only", action="store_true",
                        help="only redo Phase C (tables and figures) from the saved scores")
    parser.add_argument("--out-dir", help="safe re-run: write every output into this folder, e.g. "
                                          "outputs/rerun (the official files are never touched)")
    args = parser.parse_args()

    # every check comes BEFORE any file is opened, so a refused run changes nothing
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR = (OUT_DIR if OUT_DIR.is_absolute() else ROOT / OUT_DIR).resolve()
        _check_out_dir(OUT_DIR)
        RES_DIR, FIG_DIR, HEAT_DIR = OUT_DIR / "results", OUT_DIR / "figures", OUT_DIR / "checkpoints"
    if args.report_only:
        missing = [str(p) for p in [RES_DIR / "robustness.csv"] +
                   [HEAT_DIR / f"{v}_heatmaps.npz" for v in VARIANTS] if not p.exists()]
        if missing:
            sys.exit("--report-only needs the saved scores and heat-maps of a finished run. Missing:\n  "
                     + "\n  ".join(missing) + ("\n(download the Release, see README.md)" if OUT_DIR is None else ""))
    elif (RES_DIR / "metrics.csv").exists():
        if OUT_DIR is None:
            sys.exit("metrics.csv exists: the evaluation was already run. Use --report-only for figures, "
                     "or --out-dir outputs/rerun for a separate re-run.")
        sys.exit(f"{OUT_DIR} already holds a finished run: delete it or choose a new folder, e.g. outputs/rerun2")

    if OUT_DIR is not None:
        for d in (RES_DIR, FIG_DIR, HEAT_DIR):
            d.mkdir(parents=True, exist_ok=True)
    if OUT_DIR is not None or not args.report_only:
        log = open(RES_DIR / "evaluate_log.txt", "a" if args.report_only else "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log)
    # (an official --report-only prints to the screen only: the log of the official run stays as it was)
    t0 = time.time()
    print(f"evaluate.py started {datetime.now():%Y-%m-%d %H:%M}"
          + (" (report only)" if args.report_only else "")
          + (f" | out-dir {OUT_DIR}" if OUT_DIR is not None else ""))
    if OUT_DIR is not None and not (OUT_DIR.parent == OUTPUTS.resolve() and OUT_DIR.name.startswith("rerun")):
        print(f"note: git only ignores outputs/rerun*/, not {OUT_DIR}: never commit it")
    if not args.report_only:
        print("== PHASE A: validation only ==")
        thresholds, val_rob = phase_a()
        print("== PHASE B: test set ==")
        phase_b(thresholds, val_rob)
    print("== PHASE C: report ==")
    phase_c()
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
