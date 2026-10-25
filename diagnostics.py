"""POST-HOC DIAGNOSTICS, run after the final evaluation: python diagnostics.py

Question: in evaluate.py, a 10 deg rotation and mild noise pushed ALL 40 good
validation images of PatchCore above its threshold. Is that caused by the angle,
by the flat grey corner wedges, by the resampling blur at 256 px, or would
realistic camera noise do the same?

This script changes NOTHING: it uses only the 40 VALIDATION images, the 4 existing
checkpoints (never refitted) and the frozen thresholds read from thresholds.csv.
No test image is loaded, and metrics.csv / robustness.csv / thresholds.csv are
hashed before and after to prove they are untouched. These results must not be
used to change any model or threshold.

Conditions (same 40 images for every variant):
  clean               standard load_images at 256 (must match <variant>_val_scores.csv)
  rotate_10_256       reference: evaluate.py's own rotate_10 (flat grey wedges)
  noise_256           reference: evaluate.py's own noise (sigma 0.02 at 256 px)
  rot90               diagnostic: np.rot90, exact, no interpolation, no wedges
  rot10_reflect_1024  diagnostic: 10 deg at 1024 px, reflect padding, then standard resize
  wedges_only         diagnostic: no rotation, the same wedges as rotate_10 pasted in
  noise_1024          diagnostic: sigma 0.02 added at 1024 px BEFORE resizing (camera-like)
  clean_via_1024      control A: the 1024 pipeline without perturbation (must equal clean)
  rot10_reflect_256   control B: 10 deg at 256 with reflect: blur at 256 but no wedges
"""
import gc
import hashlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import rotate as nd_rotate

from evaluate import PERTURBATIONS, VARIANTS, _Tee, _load
from src.data import DATA_DIR, IMG_SIZE, get_split, load_images
from src.paths import FIGURES, RESULTS

FROZEN = [RESULTS / f for f in ("metrics.csv", "robustness.csv", "thresholds.csv")]
NOTE = "post-hoc diagnostic, validation only"
TITLE = "POST-HOC DIAGNOSTIC - validation images only, frozen models and thresholds"


def _hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN}


def _load_1024(rel_path):
    """The raw 1024 x 1024 image as uint8 (no resizing)."""
    return np.asarray(Image.open(DATA_DIR / rel_path).convert("L"))


def _resize_uint8(img_uint8):
    """Exactly the resize of data.load_image: PIL bilinear on uint8, then / 255."""
    img = Image.fromarray(img_uint8).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_uint8(x01):
    return np.round(np.clip(x01, 0, 1) * 255).astype(np.uint8)


def build_conditions(files, clean):
    big = [_load_1024(f) for f in files]
    # the wedges of evaluate.py's rotate_10: same rotate call on an all-ones image, cval 0
    cover = nd_rotate(np.ones((IMG_SIZE, IMG_SIZE), np.float32), 10, reshape=False, order=1,
                      mode="constant", cval=0.0)
    wedges = np.stack([img * cover + np.median(img) * (1 - cover) for img in clean]).astype(np.float32)
    rot1024 = [nd_rotate(b.astype(np.float32) / 255, 10, reshape=False, order=1, mode="reflect")
               for b in big]
    noisy1024 = [b / 255 + np.random.default_rng(42 + i).normal(0, 0.02, b.shape)
                 for i, b in enumerate(big)]
    return {
        "clean": ("reference", "standard load_images at 256", clean),
        "rotate_10_256": ("reference", "evaluate.py rotate_10 (flat grey wedges)",
                          PERTURBATIONS["rotate_10"](clean)),
        "noise_256": ("reference", "evaluate.py noise, sigma 0.02 at 256", PERTURBATIONS["noise"](clean)),
        "rot90": ("diagnostic", "np.rot90, no interpolation, no wedges",
                  np.ascontiguousarray(np.rot90(clean, k=1, axes=(1, 2)))),
        "rot10_reflect_1024": ("diagnostic", "10 deg at 1024, reflect padding, then resize",
                               np.stack([_resize_uint8(_to_uint8(r)) for r in rot1024])),
        "wedges_only": ("diagnostic", "no rotation, rotate_10 wedges pasted in", wedges),
        "noise_1024": ("diagnostic", "sigma 0.02 at 1024 before resize",
                       np.stack([_resize_uint8(_to_uint8(n)) for n in noisy1024])),
        "clean_via_1024": ("control", "1024 pipeline, no change (must equal clean)",
                           np.stack([_resize_uint8(b) for b in big])),
        "rot10_reflect_256": ("control", "10 deg at 256, reflect: blur but no wedges",
                              np.stack([nd_rotate(img, 10, reshape=False, order=1, mode="reflect")
                                        for img in clean]).astype(np.float32)),
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sys.stdout = _Tee(sys.__stdout__, open(RESULTS / "diagnostics_log.txt", "w", encoding="utf-8"))
    t0 = time.time()
    print(TITLE)
    before = _hashes()
    for k, v in before.items():
        print(f"sha256 {k}: {v}")

    th = pd.read_csv(RESULTS / "thresholds.csv").set_index("variant")   # frozen, not recomputed
    files = get_split()["val"]
    clean = load_images(files)
    conds = build_conditions(files, clean)

    rows, maps_shown, clean_scores = [], {}, {}
    for name in VARIANTS:
        det = _load(name)
        T = th.loc[name, "T_image"]
        scores = {c: det.predict_many(x)[0] for c, (_, _, x) in conds.items()}
        s0 = scores["clean"]
        clean_scores[name] = s0
        saved = pd.read_csv(RESULTS / f"{name}_val_scores.csv").score.to_numpy()
        print(f"[{name}] clean vs {name}_val_scores.csv: max abs diff {np.abs(s0 - saved).max():.2e} | "
              f"clean_via_1024 vs clean: max abs diff {np.abs(scores['clean_via_1024'] - s0).max():.2e}")
        for c, (kind, desc, _) in conds.items():
            rel = (scores[c] - s0) / s0
            rows.append({"variant": name, "condition": c, "kind": kind, "description": desc,
                         "mean_rel_change": float(rel.mean()), "median_rel_change": float(np.median(rel)),
                         "val_frr": float((scores[c] > T).mean()), "n_above_T": int((scores[c] > T).sum()),
                         "max_score_over_T": float(scores[c].max() / T), "note": NOTE})
        if name in ("ae_raw", "pc_raw"):                       # heat-maps for the example figure
            maps_shown[name] = det
        else:
            del det
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diag = pd.DataFrame(rows)
    diag.to_csv(RESULTS / "diagnostics.csv", index=False)

    # reference rows must reproduce robustness.csv (validation part)
    rob = pd.read_csv(RESULTS / "robustness.csv").set_index(["variant", "perturbation"])
    print("\nreference check vs robustness.csv (val_frr, val_mean_rel_change):")
    for name in VARIANTS:
        for c, p in (("rotate_10_256", "rotate_10"), ("noise_256", "noise")):
            d = diag[(diag.variant == name) & (diag.condition == c)].iloc[0]
            r = rob.loc[(name, p)]
            print(f"  {name:6s} {p:9s} here {d.val_frr:.3f} / {d.mean_rel_change:+.4f}   "
                  f"robustness.csv {r.val_frr:.3f} / {r.val_mean_rel_change:+.4f}")

    order = list(conds)
    pd.set_option("display.width", 200)
    for col, scale in (("val_frr", 1), ("mean_rel_change", 100)):
        print(f"\n{col}{' (%)' if scale == 100 else ''}:")
        print((diag.pivot(index="condition", columns="variant", values=col).loc[order, list(VARIANTS)]
               * scale).round(3 if scale == 1 else 1).to_string())

    # figure 1: grouped bars
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for ax, col, lab in ((axes[0], "val_frr", "val FRR at frozen T_image"),
                         (axes[1], "mean_rel_change", "mean relative score change vs clean")):
        diag.pivot(index="condition", columns="variant", values=col).loc[order, list(VARIANTS)] \
            .plot.bar(ax=ax, rot=20)
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(TITLE)
    fig.tight_layout()
    fig.savefig(FIGURES / "diag_summary.png", dpi=100)
    plt.close(fig)

    # figure 2: fixed rule = val image with the median clean pc_raw score
    i = int(np.argsort(clean_scores["pc_raw"])[len(files) // 2])
    maps = {m: {c: maps_shown[m].predict_many(x[i:i + 1])[1][0] for c, (_, _, x) in conds.items()}
            for m in ("ae_raw", "pc_raw")}
    fig, axes = plt.subplots(len(order), 3, figsize=(8, 2.6 * len(order)))
    for row, c in zip(axes, order):
        row[0].imshow(conds[c][2][i], cmap="gray", vmin=0, vmax=1)
        row[0].set_title(c, fontsize=8)
        for k, m in enumerate(("ae_raw", "pc_raw"), start=1):
            row[k].imshow(maps[m][c], cmap="jet", vmin=th.loc[m, "val_heatmap_median"],
                          vmax=max(float(v.max()) for v in maps[m].values()))
            row[k].set_title(f"{m} heat-map", fontsize=8)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"{TITLE}\n{files[i]} (median clean pc_raw score)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])                 # leave room for the title
    fig.savefig(FIGURES / "diag_examples.png", dpi=70)
    plt.close(fig)
    del maps_shown

    after = _hashes()
    assert after == before, "a frozen results file changed!"
    print("\nfrozen files unchanged (sha256 identical before and after)")
    print(f"run time: {time.time() - t0:.0f} s")
