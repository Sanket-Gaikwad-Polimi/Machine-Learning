"""v2 PHASE 3c: build and score the 6 candidates ONCE, then apply the Phase 3b rule: python phase3c.py

Everything is fixed in decisions.md (v2, Phase 3b), committed before this script ran.
  1. build (train/val only): ae_ssim and the pc_512 bank -> outputs/checkpoints/v2/
  2. thresholds of every candidate from its own 40 good validation images (v1 rule)
  3. score the 40 good validation + 160 synthetic images with the 2 baselines (must equal
     Phase 2) and the 6 candidates
  4. metrics as in Phase 2 + paired cluster bootstrap against the baseline
  5. the decision rule, in code, and the selection of v2 per model
No test image is loaded: every file read must be under screw/train/good (checked).
Outputs: outputs/v2/phase3/ (CSV, PNG, log). Earlier files are hashed before and after.
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
from sklearn.metrics import roc_auc_score

from phase2 import evaluate_variant
from src.autoencoder import AutoencoderDetector
from src.data import get_split, load_images
from src.masks import background_masks
from src.metrics import Q_IMAGE, Q_PIXEL, threshold_from_val
from src.patchcore import PatchCoreDetector
from src.paths import CHECKPOINTS, FIGURES, OUTPUTS, RESULTS, ROOT
from src.synthetic import TYPES, make_synthetic

OUT = OUTPUTS / "v2" / "phase3"
V2_CKPT = CHECKPOINTS / "v2"
LABEL = "v2 phase 3c: candidates on synthetic defects (validation images only)"
N_BOOT, SEED = 2000, 42
MAX_HIT_DROP, MAX_TYPE_DROP = 0.05, 0.10

# candidate: (model, input size, background mask, baseline)
CANDIDATES = {
    "pc_raw": ("pc_v1", 256, False, None),
    "pc_512": ("pc_512", 512, False, "pc_raw"),
    "pc_mask": ("pc_v1", 256, True, "pc_raw"),
    "pc_512_mask": ("pc_512", 512, True, "pc_raw"),
    "ae_raw": ("ae_v1", 256, False, None),
    "ae_ssim": ("ae_ssim", 256, False, "ae_raw"),
    "ae_mask": ("ae_v1", 256, True, "ae_raw"),
    "ae_ssim_mask": ("ae_ssim", 256, True, "ae_raw"),
}
COLORS = {"pc_raw": "0.35", "pc_512": "C2", "pc_mask": "C4", "pc_512_mask": "C6",
          "ae_raw": "0.65", "ae_ssim": "C0", "ae_mask": "C1", "ae_ssim_mask": "C9"}
FILES_READ = {}


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


def _guard(files, n_reads=1):
    """Refuse anything that is not a good training image of screw (no test, no transistor)."""
    for f in files:
        assert f.startswith("train/good/") and ".." not in f, f"refused: {f}"
    FILES_READ["screw/train/good"] = FILES_READ.get("screw/train/good", 0) + n_reads * len(files)
    return files


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


PHASE3B_FILES = ("val_masks.png", "val_masks.csv", "feasibility.txt", "phase3_checks_log.txt")


def _frozen_hashes():
    """v1 outputs, v1 checkpoints (not checkpoints/v2), analysis, v2/phase2 and the phase 3b files."""
    dirs = (RESULTS, FIGURES, CHECKPOINTS, OUTPUTS / "analysis", OUTPUTS / "v2" / "phase2")
    h = {p.relative_to(OUTPUTS).as_posix(): _sha(p) for d in dirs if d.exists()
         for p in sorted(d.rglob("*")) if p.is_file() and V2_CKPT not in p.parents}
    h.update({(OUT / f).relative_to(OUTPUTS).as_posix(): _sha(OUT / f) for f in PHASE3B_FILES})
    return h


def _code_hashes():
    files = [ROOT / "decisions.md", ROOT / "phase2.py", ROOT / "phase3_checks.py", ROOT / "phase3c.py"]
    files += sorted((ROOT / "src").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): _sha(p) for p in files}


def _write_csv(df, name):
    with open(OUT / name, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        df.to_csv(f, index=False, lineterminator="\n")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- 1. build -------------------------------------------------------------------------
def build(split):
    V2_CKPT.mkdir(parents=True, exist_ok=True)
    info = {}
    path = V2_CKPT / "ae_ssim.pt"
    if path.exists():
        print(f"ae_ssim: {path} exists, loaded (not retrained)")
        det = AutoencoderDetector.load(path)
    else:
        train = load_images(_guard(split["train"]))
        val = load_images(_guard(split["val"]))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            det = AutoencoderDetector(loss="mse_ssim").fit(train, val)
        except torch.cuda.OutOfMemoryError:                  # same technical fallback as v1
            _free()
            print("out of GPU memory, retrying with batch size 4")
            det = AutoencoderDetector(loss="mse_ssim", batch_size=4).fit(train, val)
        info["ae_ssim_build_min"] = (time.time() - t0) / 60
        info["ae_ssim_gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        det.save(path)
        print(f"ae_ssim trained in {info['ae_ssim_build_min']:.1f} min, best epoch {det.best_epoch}, "
              f"GPU peak {info['ae_ssim_gpu_peak_mb']:.0f} MB, saved {path}")
        del train, val
    h = pd.DataFrame(det.history)
    _write_csv(h, "ae_ssim_history.csv")
    info["ae_ssim_best_epoch"] = det.best_epoch
    plot_training(h, det.best_epoch, OUT / "ae_ssim_training.png")
    del det
    _free()

    path = V2_CKPT / "pc_512.pt"
    if path.exists():
        print(f"pc_512: {path} exists, loaded (not rebuilt)")
    else:
        train512 = load_images(_guard(split["train"]), size=512)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        det = PatchCoreDetector(size=512, patch_sample=0.25).fit(train512)
        info["pc_512_build_min"] = (time.time() - t0) / 60
        info["pc_512_gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        info["pc_512_ram_peak_mb"] = psutil.Process().memory_info().peak_wset / 2**20
        det.save(path)
        print(f"pc_512 built in {info['pc_512_build_min']:.1f} min, bank {tuple(det.bank.shape)}, GPU peak "
              f"{info['pc_512_gpu_peak_mb']:.0f} MB, RAM peak {info['pc_512_ram_peak_mb']:.0f} MB, saved {path}")
        del det, train512
        _free()
    return info


def plot_training(h, best, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(h.epoch, h.train_loss, label="train (280 good)")
    ax.plot(h.epoch, h.val_loss, label="validation (40 good)")
    ax.axvline(best, color="gray", ls="--", lw=1, label=f"best epoch {best}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss = 0.5 MSE + 0.5 (1 - SSIM), log scale")
    ax.set_title(f"ae_ssim training (same data, epochs, seed, augmentation as v1 ae_raw)\n{LABEL}", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---- 3. scoring -------------------------------------------------------------------------
def _model(key):
    if key == "pc_v1":
        return PatchCoreDetector.load(CHECKPOINTS / "patchcore_bank.pt")
    if key == "pc_512":
        return PatchCoreDetector.load(V2_CKPT / "pc_512.pt")
    if key == "ae_v1":
        return AutoencoderDetector.load(CHECKPOINTS / "autoencoder.pt")
    return AutoencoderDetector.load(V2_CKPT / "ae_ssim.pt")


def score_all(data):
    out = {}
    for key in ("pc_v1", "pc_512", "ae_v1", "ae_ssim"):
        det = _model(key)
        for name, (mk, size, masked, _) in CANDIDATES.items():
            if mk != key:
                continue
            t0 = time.time()
            r = {}
            for part in ("good", "syn"):
                x = data[f"{part}_{size}"]
                m = data[f"{part}_bg_{size}"] if masked else None
                r[f"s_{part}"], r[f"m_{part}"] = det.predict_many(x, masks=m)
            r["T"] = threshold_from_val(r["s_good"], Q_IMAGE)
            r["T_pix"] = threshold_from_val(r["m_good"], Q_PIXEL)
            out[name] = r
            print(f"[{name}] scored 40 + 160 images in {time.time() - t0:.0f} s | T_image {r['T']:.5g} "
                  f"T_pixel {r['T_pix']:.5g}")
        del det
        _free()
    return out


def check_baselines(res):
    """Baselines must reproduce v1 thresholds, val scores and the Phase 2 synthetic scores."""
    th = pd.read_csv(RESULTS / "thresholds.csv").set_index("variant")
    syn = pd.read_csv(OUTPUTS / "v2" / "phase2" / "synthetic_defects.csv", comment="#")
    for name in ("pc_raw", "ae_raw"):
        r = res[name]
        val = pd.read_csv(RESULTS / f"{name}_val_scores.csv").score.to_numpy()
        d_val = np.abs(r["s_good"] - val).max()
        d_syn = np.abs(r["s_syn"] / th.loc[name, "T_image"] - syn[f"{name}_score_over_T"].to_numpy()).max()
        d_T = abs(r["T"] - th.loc[name, "T_image"]) / th.loc[name, "T_image"]
        d_Tp = abs(r["T_pix"] - th.loc[name, "T_pixel"]) / th.loc[name, "T_pixel"]
        print(f"[{name}] reproduction: val scores max diff {d_val:.1e} | synthetic score/T max diff {d_syn:.1e} "
              f"(CSV rounded to 5 decimals) | T_image rel diff {d_T:.1e} | T_pixel rel diff {d_Tp:.1e}")
        assert d_val < 1e-6 and d_syn < 1e-5 and d_T < 1e-6 and d_Tp < 1e-5, f"{name} does not reproduce v1"


# ---- 5. bootstrap and rule --------------------------------------------------------------
def bootstrap(res):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 40, size=(N_BOOT, 40))             # 40 base images with replacement
    syn_idx = (idx[:, :, None] * len(TYPES) + np.arange(len(TYPES))).reshape(N_BOOT, -1)
    y = np.r_[np.zeros(40), np.ones(40 * len(TYPES))]
    auc = {}
    for name, r in res.items():
        auc[name] = np.array([roc_auc_score(y, np.r_[r["s_good"][g], r["s_syn"][s]])
                              for g, s in zip(idx, syn_idx)])
    rows = []
    for name, (_, _, _, base) in CANDIDATES.items():
        if base is None:
            continue
        d = auc[name] - auc[base]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"candidate": name, "baseline": base, "gain_ci_low": lo, "gain_ci_high": hi,
                     "gain_boot_mean": d.mean(), "share_le_0": float((d <= 0).mean()), "n_resamples": N_BOOT})
    return pd.DataFrame(rows)


def decide(metrics, boot):
    m = metrics.set_index(["candidate", "group"])
    rows = []
    for _, b in boot.iterrows():
        c, base = b.candidate, b.baseline
        gain = m.loc[(c, "all synthetic"), "auroc"] - m.loc[(base, "all synthetic"), "auroc"]
        hit = m.loc[(c, "all synthetic"), "hit_rate"] - m.loc[(base, "all synthetic"), "hit_rate"]
        per = {t: m.loc[(c, t), "auroc"] - m.loc[(base, t), "auroc"] for t in TYPES}
        worst = min(per, key=per.get)
        c1, c2, c3 = b.gain_ci_low > 0, hit >= -MAX_HIT_DROP, per[worst] >= -MAX_TYPE_DROP
        rows.append({"candidate": c, "baseline": base,
                     "auroc_baseline": m.loc[(base, "all synthetic"), "auroc"],
                     "auroc_candidate": m.loc[(c, "all synthetic"), "auroc"],
                     "auroc_gain": gain, "gain_ci_low": b.gain_ci_low, "gain_ci_high": b.gain_ci_high,
                     "hit_rate_change": hit, "worst_type": worst, "worst_type_auroc_change": per[worst],
                     "c1_ci_low_gt_0": c1, "c2_hit_drop_le_0.05": c2, "c3_type_drop_le_0.10": c3,
                     "verdict": "BEATS BASELINE" if c1 and c2 and c3 else "does not beat baseline"})
    table = pd.DataFrame(rows)
    selection = []
    for base, fam in (("pc_raw", "PatchCore"), ("ae_raw", "autoencoder")):
        t = table[(table.baseline == base) & (table.verdict == "BEATS BASELINE")]
        chosen = t.sort_values("auroc_candidate", ascending=False).candidate.iloc[0] if len(t) else base
        selection.append({"model": fam, "v2": chosen, "rule": "highest AUROC among passing candidates"
                          if len(t) else "no candidate passed: v2 = v1"})
    return table, pd.DataFrame(selection)


# ---- figures ----------------------------------------------------------------------------
def plot_forest(table, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(table))[::-1]
    for yi, (_, r) in zip(y, table.iterrows()):
        ok = r.verdict == "BEATS BASELINE"
        ax.errorbar(r.auroc_gain, yi, xerr=[[r.auroc_gain - r.gain_ci_low], [r.gain_ci_high - r.auroc_gain]],
                    fmt="o", ms=7, capsize=5, lw=2, color="C2" if ok else "C3")
        ax.text(r.gain_ci_high + 0.01, yi, f"{r.auroc_gain:+.3f} [{r.gain_ci_low:+.3f}, {r.gain_ci_high:+.3f}]  "
                + ("PASS" if ok else "fail"), va="center", fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, [f"{r.candidate} vs {r.baseline}" for _, r in table.iterrows()])
    lo, hi = table.gain_ci_low.min(), table.gain_ci_high.max()
    ax.set_xlim(min(lo, 0) - 0.03, hi + 0.25)
    ax.set_xlabel("image AUROC gain over the baseline (40 good vs 160 synthetic), 95% cluster-bootstrap CI")
    ax.set_title(f"Decision rule, criterion 1 (CI lower bound > 0); verdict uses all 3 criteria\n{LABEL}", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_per_type(metrics, path):
    import matplotlib.pyplot as plt
    groups = TYPES + ["all synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, fam in zip(axes, (["pc_raw", "pc_512", "pc_mask", "pc_512_mask"],
                              ["ae_raw", "ae_ssim", "ae_mask", "ae_ssim_mask"])):
        for k, name in enumerate(fam):
            r = metrics[metrics.candidate == name].set_index("group").loc[groups]
            x = np.arange(len(groups)) + (k - 1.5) * 0.2
            ax.bar(x, r.auroc, 0.18, color=COLORS[name], label=name + (" (baseline)" if k == 0 else ""))
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(np.arange(len(groups)), groups, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(("PatchCore" if fam[0].startswith("pc") else "Autoencoder") + ": image AUROC per synthetic type",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    axes[0].set_ylabel("AUROC (40 good vs 40 or 160 synthetic)")
    fig.suptitle(LABEL, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(OUT / "phase3c_log.txt", "w", encoding="utf-8"))
    t_start = time.time()
    print(f"phase3c.py started {datetime.now():%Y-%m-%d %H:%M} | {LABEL}")
    code_before = _code_hashes()
    print("sha256 of the code and decisions (fixed before scoring):")
    for k, v in code_before.items():
        print(f"  {v}  {k}")
    before = _frozen_hashes()
    print(f"hashed {len(before)} earlier files (v1 outputs, checkpoints except v2/, analysis, v2/phase2, v2/phase3)")

    split = get_split()
    print("\n== 1. BUILD (train/val only) ==")
    info = build(split)

    print("\n== 2-3. DATA AND SCORING ==")
    val = _guard(split["val"], n_reads=1)
    data = {"good_256": load_images(_guard(val)), "good_512": load_images(_guard(val), size=512),
            "good_bg_256": background_masks(_guard(val), 256), "good_bg_512": background_masks(_guard(val), 512)}
    images, gt, syn_info, extra = make_synthetic(_guard(val), image_sizes=(512,), bg_mask_sizes=(256, 512))
    data.update({"syn_256": images, "syn_512": extra["images"][512],
                 "syn_bg_256": extra["bg_masks"][256], "syn_bg_512": extra["bg_masks"][512]})
    types = syn_info.type.to_numpy()
    print(f"files read: {FILES_READ} (each read counted; distinct files: 280 train + 40 val, all screw/train/good)")
    res = score_all(data)
    check_baselines(res)
    _write_csv(pd.DataFrame([{"candidate": n, "T_image": r["T"], "T_pixel": r["T_pix"]} for n, r in res.items()]),
               "candidate_thresholds.csv")
    scores = pd.DataFrame({"file": list(val) + list(syn_info.file), "type": ["good"] * 40 + list(types),
                           **{n: np.r_[r["s_good"], r["s_syn"]] for n, r in res.items()}})
    _write_csv(scores, "candidate_scores.csv")

    print("\n== 4. METRICS ==")
    metrics = pd.concat([evaluate_variant(r["s_good"], r["m_good"], r["s_syn"], r["m_syn"], gt, types,
                                          r["T"], r["T_pix"]).assign(candidate=n) for n, r in res.items()])
    metrics = metrics[["candidate"] + [c for c in metrics.columns if c != "candidate"]]
    _write_csv(metrics, "candidate_metrics.csv")
    pd.set_option("display.width", 250)
    order = list(CANDIDATES)
    for col in ("auroc", "hit_rate", "recall", "pixel_auroc", "iou", "dice"):
        print(f"\n{col}:")
        print(metrics.pivot(index="candidate", columns="group", values=col).loc[order, TYPES + ["all synthetic"]]
              .round(3).to_string())

    print(f"\n== 5. BOOTSTRAP ({N_BOOT} paired cluster resamples, seed {SEED}) AND DECISION RULE ==")
    boot = bootstrap(res)
    _write_csv(boot, "bootstrap.csv")
    table, selection = decide(metrics, boot)
    _write_csv(table, "decision_table.csv")
    _write_csv(selection, "selection.csv")
    show = table.copy()
    show["gain [95% CI]"] = [f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]" for g, lo, hi in
                             zip(show.auroc_gain, show.gain_ci_low, show.gain_ci_high)]
    show["worst type"] = [f"{t} {d:+.3f}" for t, d in zip(show.worst_type, show.worst_type_auroc_change)]
    print(show[["candidate", "baseline", "auroc_baseline", "auroc_candidate", "gain [95% CI]", "hit_rate_change",
                "worst type", "c1_ci_low_gt_0", "c2_hit_drop_le_0.05", "c3_type_drop_le_0.10", "verdict"]]
          .round(3).to_string(index=False))
    print("\nSELECTION:")
    print(selection.to_string(index=False))
    plot_forest(table, OUT / "decision_forest.png")
    plot_per_type(metrics, OUT / "per_type_auroc.png")
    if info:
        print("\nbuild info:", {k: round(v, 2) for k, v in info.items()})

    after = _frozen_hashes()
    changed = [k for k, v in before.items() if after.get(k) != v]
    print(f"\nearlier files: {len(before)} hashed, {len(changed)} changed {changed}")
    assert not changed, "an earlier file changed!"
    assert _code_hashes() == code_before, "code or decisions changed during the run!"
    print("code and decisions.md unchanged during the run")
    print(f"run time: {(time.time() - t_start) / 60:.1f} min")
