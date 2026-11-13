"""CHECKS FOR v2 PHASE 3b (no candidate is scored on synthetic or test images): python phase3_checks.py

  1. background masks of the 40 validation screws: figure + table (by eye and by number:
     is any part of the screw, e.g. a thread tip or the head edge, cut off?)
  2. unit checks: SSIM, the autoencoder loss, and the mask application on 2 validation
     images (shape and position of the maximum only; no score is reported)
  3. feasibility of pc_512 with 10 TRAIN images only (time, GPU and RAM peak, extrapolation);
     the test bank lives only in memory and is deleted.

Everything is written to outputs/v2/phase3/. v1 files (outputs/results, figures,
checkpoints, analysis) and outputs/v2/phase2 are hashed before and after.
"""
import gc
import hashlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.autoencoder import AutoencoderDetector, _loss, ssim_map
from src.data import DATA_DIR, get_split, load_images
from src.masks import DILATE, apply_mask, background_masks, mask_to_size, object_mask_1024
from src.patchcore import PatchCoreDetector
from src.paths import CHECKPOINTS, FIGURES, OUTPUTS, RESULTS
from src.synthetic import screw_region

OUT = OUTPUTS / "v2" / "phase3"
LABEL = "v2 phase 3b checks (no candidate scored)"


class _Tee:
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
    dirs = (RESULTS, FIGURES, CHECKPOINTS, OUTPUTS / "analysis", OUTPUTS / "v2" / "phase2")
    return {p.relative_to(OUTPUTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


def _ram_mb():
    return psutil.Process().memory_info().rss / 2**20


# ---- 1. masks of the 40 validation screws ------------------------------------------
def mask_check(files):
    import matplotlib.pyplot as plt
    rows, shown = [], []
    for i, f in enumerate(files):
        img = np.asarray(Image.open(DATA_DIR / f).convert("L"), dtype=np.float32) / 255.0
        screw = screw_region(img)
        obj = object_mask_1024(img)
        loose = img < np.median(img) - 0.15                      # looser rule (includes the shadow)
        out = loose & ~obj
        dist = distance_transform_edt(~screw)
        dark = (img < np.median(img) - 0.25) & ~obj               # metal-dark, between both rules
        rows.append({"val_index": i, "file": f, "screw_area_pct": 100 * screw.mean(),
                     "mask_area_pct": 100 * obj.mean(), "loose_px_outside_mask": int(out.sum()),
                     "loose_max_dist_px": float(dist[out].max()) if out.any() else 0.0,
                     "dark025_px_outside_mask": int(dark.sum())})
        shown.append((np.asarray(Image.fromarray(np.round(img * 255).astype(np.uint8))
                                 .resize((256, 256), Image.Resampling.BILINEAR)) / 255.0,
                      mask_to_size(screw, 256), mask_to_size(obj, 256), mask_to_size(out, 256)))
    fig, axes = plt.subplots(5, 8, figsize=(20.8, 13.6))
    for a, (r, (im, sc, ob, lo)) in zip(axes.flat, zip(rows, shown)):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.contour(ob, levels=[0.5], colors="deepskyblue", linewidths=1.0)
        a.contour(sc, levels=[0.5], colors="orange", linewidths=0.6)
        if lo.any():
            ys, xs = np.nonzero(lo)
            a.scatter(xs, ys, s=1, c="magenta")
        a.set_title(f"val {r['val_index']}: mask {r['mask_area_pct']:.1f}%", fontsize=8)
        a.axis("off")
    fig.suptitle("Background mask of the 40 validation screws (256 px): blue = mask (screw + 32 px at 1024), "
                 "orange = screw rule,\nmagenta = darker than median - 0.15 but outside the mask (shadow or cut-off part?)"
                 f" | {LABEL}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "val_masks.png", dpi=100)
    plt.close(fig)
    return pd.DataFrame(rows)


# ---- 2. unit checks ---------------------------------------------------------------
def unit_checks(files):
    x = torch.from_numpy(load_images(files[:4]))[:, None]
    s = ssim_map(x, x)
    print(f"SSIM(x, x): shape {tuple(s.shape)}, min {s.min().item():.6f}, max {s.max().item():.6f} (must be 1)")
    g = torch.Generator().manual_seed(0)
    losses = [_loss((x + sd * torch.randn(x.shape, generator=g)).clamp(0, 1), x, "mse_ssim").item()
              for sd in (0.10, 0.05, 0.02, 0.0)]
    print("loss 0.5 MSE + 0.5 (1 - SSIM) for reconstruction noise sd 0.10 / 0.05 / 0.02 / 0: "
          + " / ".join(f"{v:.4f}" for v in losses), "-> decreasing:", all(a > b for a, b in zip(losses, losses[1:])))
    print("mse loss unchanged for v1:", _loss(x * 0.9, x, "mse").item() == torch.nn.functional.mse_loss(x * 0.9, x).item())

    two = files[:2]
    imgs, m256 = load_images(two), background_masks(two, 256)
    for name, det in (("ae_raw + mask", AutoencoderDetector.load(CHECKPOINTS / "autoencoder.pt")),
                      ("pc_raw + mask", PatchCoreDetector.load(CHECKPOINTS / "patchcore_bank.pt"))):
        _, maps = det.predict_many(imgs, masks=m256)
        ok = [m[np.unravel_index(np.argmax(m), m.shape)] for m in m256]
        print(f"{name}: heat-maps {maps.shape}, maximum inside the mask: {ok}, "
              f"outside = minimum: {[bool((h[~m] == h.min()).all()) for h, m in zip(maps, m256)]}")
        del det
        gc.collect()
        torch.cuda.empty_cache()
    try:
        background_masks(two, 256, category="transistor")
        print("transistor mask -> NO ERROR (bad)")
    except ValueError as e:
        print("transistor mask ->", e)


# ---- 3. feasibility of pc_512 -------------------------------------------------------
def feasibility():
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    train = get_split()["train"][:10]
    x = load_images(train, size=512)
    log(f"10 TRAIN images at 512 px: {x.shape}")
    torch.cuda.reset_peak_memory_stats()
    ram0 = _ram_mb()
    det = PatchCoreDetector(size=512, patch_sample=0.25)
    t0 = time.time()
    feats = [det._embed(x[s:s + 8]) for s in range(0, 10, 8)]
    torch.cuda.synchronize()
    t_emb = (time.time() - t0) / 10
    log(f"feature extraction: {t_emb:.2f} s per image (batch 8), grid {tuple(feats[0].shape[2:])}")
    t0 = time.time()
    det.fit(x)                                               # 10 images: 10 x 1024 vectors -> coreset 1,024
    log(f"fit on 10 images (25% sample, coreset 10%): {time.time() - t0:.1f} s, bank {tuple(det.bank.shape)}")

    # coreset cost on a pool of the full size (286,720 x 1536, random numbers), 300 iterations
    pool = torch.randn(286_720, 1536)
    g = torch.Generator().manual_seed(42)
    proj = (torch.randn(1536, 128, generator=g) / 128 ** 0.5).to(det.device)
    t0 = time.time()
    z = torch.cat([c.to(det.device) @ proj for c in pool.split(16384)])
    z2 = (z ** 2).sum(dim=1)
    min_d = (z2 - 2 * z @ z[0] + z2[0]).clamp_min(0)
    t_proj = time.time() - t0
    t0 = time.time()
    for _ in range(300):
        i = int(torch.argmax(min_d))
        min_d = torch.minimum(min_d, (z2 - 2 * z @ z[i] + z2[i]).clamp_min(0))
    torch.cuda.synchronize()
    t_it = (time.time() - t0) / 300
    log(f"coreset on a full-size pool (286,720 x 1536): projection {t_proj:.1f} s, {1000 * t_it:.2f} ms per "
        f"iteration -> about {(t_proj + 28_672 * t_it) / 60:.1f} min for 28,672 (same as v1's pool size)")
    log(f"RAM for the full 25% pool: {pool.numel() * 4 / 2**30:.2f} GB (same as v1 at 256)")
    del pool, z, z2, min_d
    gc.collect()

    # nearest-neighbour search: bank of full size (28,672) built by sampling the 10-image features
    allf = torch.cat([f.permute(0, 2, 3, 1).reshape(-1, 1536) for f in feats])
    det.bank = allf[torch.from_numpy(np.random.default_rng(0).choice(len(allf), 28_672))].to(det.device)
    torch.cuda.synchronize()
    t0 = time.time()
    scores, maps = det.predict_many(x[:8])                    # TRAIN images, timing only
    torch.cuda.synchronize()
    t_pred = (time.time() - t0) / 8
    log(f"scoring at 512 against a 28,672-vector bank: {t_pred:.2f} s per image, heat-maps {maps.shape} "
        f"-> about {200 * t_pred / 60:.1f} min for 200 images (40 good + 160 synthetic)")
    log(f"full build estimate for 280 images: {280 * t_emb / 60:.1f} min features + coreset as above")
    log(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB of "
        f"{torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MB")
    log(f"peak RAM of this process: {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB "
        f"(at start of the test {ram0:.0f} MB; machine {psutil.virtual_memory().total / 2**30:.1f} GB)")
    del det, feats, allf, scores, maps
    gc.collect()
    torch.cuda.empty_cache()
    log("test bank deleted (it was never saved)")
    (OUT / "feasibility.txt").write_text(f"# pc_512 feasibility, 10 TRAIN images only | {LABEL}\n"
                                         + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(OUT / "phase3_checks_log.txt", "w", encoding="utf-8"))
    t0 = time.time()
    print(f"phase3_checks.py started {datetime.now():%Y-%m-%d %H:%M} | {LABEL}")
    before = _hashes()
    print(f"hashed {len(before)} files (v1 outputs, analysis, v2/phase2)")
    files = get_split()["val"]

    print("\n== 1. BACKGROUND MASKS OF THE 40 VALIDATION SCREWS ==")
    tab = mask_check(files)
    with open(OUT / "val_masks.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        tab.round(4).to_csv(f, index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(tab.drop(columns=["file"]).describe().loc[["min", "50%", "max"]].round(3).to_string())
    print("images with pixels darker than median - 0.25 outside the mask (metal level, shadow excluded):",
          tab[tab.dark025_px_outside_mask > 0][["val_index", "dark025_px_outside_mask"]].to_string(index=False)
          if (tab.dark025_px_outside_mask > 0).any() else "none")

    print("\n== 2. UNIT CHECKS ==")
    unit_checks(files)

    print("\n== 3. FEASIBILITY pc_512 (10 TRAIN images) ==")
    feasibility()

    after = _hashes()
    assert after == before, "a frozen file changed!"
    print(f"\nv1 untouched: all {len(before)} files have identical sha256 before and after")
    print(f"run time: {time.time() - t0:.0f} s")
