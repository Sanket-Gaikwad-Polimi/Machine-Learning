"""v2 PHASE 4c: the test sets, each scored exactly once (decisions.md, v2 Phase 4 plan and 4b predictions).

  python phase4.py --score screw        pc_512 (Phase 3c bank and thresholds) on the 160 screw test
                                        images; "post-hoc: screw test set seen before"
  python phase4.py --score transistor   build ae_raw, pc_raw, pc_512 from the 173 transistor train
                                        images (40 val for best epoch and thresholds), then score the
                                        100 transistor test images once with each
  python phase4.py --report             all metrics, CIs and figures from the SAVED scores (no model)

One-shot discipline: a scoring pass first saves the raw scores and heat-maps, refuses to run if
they already exist, and the report is computed only from those saved files.
Every run logs the sha256 of decisions.md and all code before, and checks them after.
"""
import argparse
import gc
import hashlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import roc_auc_score, roc_curve

from src.autoencoder import AutoencoderDetector
from src.categories import get_category
from src.data import get_split, list_test, load_images, load_mask, to_display
from src.metrics import Q_IMAGE, Q_PIXEL, image_metrics, pixel_metrics, threshold_from_val
from src.patchcore import PatchCoreDetector
from src.paths import CHECKPOINTS, FIGURES, OUTPUTS, RESULTS, ROOT

SCREW_OUT = OUTPUTS / "v2" / "phase4" / "screw_pc512"
SCREW_LABEL = "post-hoc: screw test set seen before (pc_512, declared extra, not selected)"
TR = get_category("transistor")
TR_LABEL = "transistor test set, scored once (v2 = v1 models; pc_512 = declared extra)"
TR_MODELS = ["pc_raw", "ae_raw", "pc_512"]
RADIUS, N_BOOT, SEED = 8, 2000, 42
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


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _code_hashes():
    files = [ROOT / "decisions.md"] + sorted(ROOT.glob("*.py")) + sorted((ROOT / "src").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): _sha(p) for p in files}


def _earlier_hashes():
    """Files that must not change: v1 outputs and checkpoints, analysis, v2 phases 2-3, transistor 4a."""
    dirs = [RESULTS, FIGURES, CHECKPOINTS, OUTPUTS / "analysis", OUTPUTS / "v2" / "phase2", OUTPUTS / "v2" / "phase3"]
    h = {p.relative_to(OUTPUTS).as_posix(): _sha(p) for d in dirs for p in sorted(d.rglob("*"))
         if p.is_file() and TR["checkpoints"] not in p.parents
         and not p.name.startswith("pc_512_screw_test")}
    for f in ("results/split.json", "results/eda_brightness.csv", "results/eda_defect_area.csv",
              "figures/eda_transistor.png"):
        p = OUTPUTS / "transistor" / f
        h[p.relative_to(OUTPUTS).as_posix()] = _sha(p)
    return h


def _read(files, category, kind):
    """Count every file read, per folder, to show that each test image is read once per pass."""
    for f in files:
        key = f"{category}/{f.rsplit('/', 1)[0]}"
        FILES_READ[key] = FILES_READ.get(key, 0) + 1
    return files


def _write_csv(df, path, label):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {label}\n")
        df.to_csv(f, index=False, lineterminator="\n")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- scoring passes (save raw scores first) -------------------------------------------
def score_screw():
    SCREW_OUT.mkdir(parents=True, exist_ok=True)
    raw = CHECKPOINTS / "v2" / "pc_512_screw_test_heatmaps.npz"
    if raw.exists():
        sys.exit(f"{raw} exists: the screw test set was already scored with pc_512. Use --report.")
    th = pd.read_csv(OUTPUTS / "v2" / "phase3" / "candidate_thresholds.csv", comment="#").set_index("candidate")
    test = list_test("screw")
    files = _read([t["file"] for t in test], "screw", "test")
    det = PatchCoreDetector.load(CHECKPOINTS / "v2" / "pc_512.pt")
    t0 = time.time()
    s, m = det.predict_many(load_images(files, size=512))
    print(f"[pc_512 screw] scored {len(files)} test images in {time.time() - t0:.0f} s")
    np.savez_compressed(raw, scores=s, heatmaps=m.astype(np.float32), files=np.array(files))
    df = pd.DataFrame({"file": files, "defect": [t["defect"] for t in test], "label": [t["label"] for t in test],
                       "score": s, "pred": (s > th.loc["pc_512", "T_image"]).astype(int)})
    _write_csv(df, SCREW_OUT / "pc_512_scores.csv", SCREW_LABEL)
    _write_csv(pd.DataFrame([{"model": "pc_512", "T_image": th.loc["pc_512", "T_image"],
                              "T_pixel": th.loc["pc_512", "T_pixel"],
                              "source": "outputs/v2/phase3/candidate_thresholds.csv (40 screw val images)"}]),
               SCREW_OUT / "thresholds.csv", SCREW_LABEL)
    del det
    _free()


def build_transistor(split):
    ck = TR["checkpoints"]
    ck.mkdir(parents=True, exist_ok=True)
    info = {}
    train = load_images(_read(split["train"], "transistor", "train"), category="transistor")
    val = load_images(_read(split["val"], "transistor", "val"), category="transistor")
    print(f"transistor train {train.shape}, val {val.shape}")
    builders = {
        "ae_raw": (ck / "autoencoder.pt", lambda: AutoencoderDetector(category="transistor").fit(train, val)),
        "pc_raw": (ck / "patchcore_bank.pt", lambda: PatchCoreDetector(category="transistor").fit(train)),
        "pc_512": (ck / "pc_512.pt", lambda: PatchCoreDetector(size=512, patch_sample=0.25, category="transistor")
                   .fit(load_images(split["train"], size=512, category="transistor"))),
    }
    for name, (path, make) in builders.items():
        if path.exists():
            print(f"[{name}] {path} exists, loaded (not rebuilt)")
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            det = make()
        except torch.cuda.OutOfMemoryError:
            if name != "ae_raw":
                raise
            _free()                                           # same technical fallback as v1
            print("out of GPU memory, retrying with batch size 4")
            det = AutoencoderDetector(category="transistor", batch_size=4).fit(train, val)
        info[name] = {"build_min": (time.time() - t0) / 60, "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                      "ram_peak_mb": psutil.Process().memory_info().peak_wset / 2**20}
        det.save(path)
        if name == "ae_raw":
            info[name]["best_epoch"] = det.best_epoch
            _write_csv(pd.DataFrame(det.history), TR["results"] / "ae_raw_history.csv", TR_LABEL)
        print(f"[{name}] built in {info[name]['build_min']:.1f} min, GPU peak {info[name]['gpu_peak_mb']:.0f} MB, "
              f"saved {path}" + (f", best epoch {det.best_epoch}" if name == "ae_raw" else
                                 f", bank {tuple(det.bank.shape)}"))
        del det
        _free()
    del train
    return val, info


def _load_tr(name):
    ck = TR["checkpoints"]
    path = {"ae_raw": ck / "autoencoder.pt", "pc_raw": ck / "patchcore_bank.pt", "pc_512": ck / "pc_512.pt"}[name]
    det = (AutoencoderDetector if name == "ae_raw" else PatchCoreDetector).load(path)
    assert det.config["category"] == "transistor"
    return det


def score_transistor():
    res, fig = TR["results"], TR["figures"]
    raws = [TR["checkpoints"] / f"{n}_heatmaps.npz" for n in TR_MODELS]
    if any(p.exists() for p in raws):
        sys.exit("transistor test heat-maps exist: the test set was already scored. Use --report.")
    split = get_split("transistor")
    val256, info = build_transistor(split)
    if info:
        _write_csv(pd.DataFrame([{"model": k, **v} for k, v in info.items()]), res / "build_info.csv", TR_LABEL)
    val512 = load_images(_read(split["val"], "transistor", "val"), size=512, category="transistor")

    # thresholds from the 40 validation images (v1 rule), saved BEFORE any test image is loaded
    rows = []
    for name in TR_MODELS:
        det = _load_tr(name)
        s, m = det.predict_many(val512 if name == "pc_512" else val256)
        pd.DataFrame({"file": split["val"], "score": s}).to_csv(res / f"{name}_val_scores.csv", index=False)
        rows.append({"model": name, "T_image": threshold_from_val(s, Q_IMAGE), "T_pixel": threshold_from_val(m, Q_PIXEL),
                     "val_min": float(s.min()), "val_median": float(np.median(s)), "val_max": float(s.max())})
        del det
        _free()
    th = pd.DataFrame(rows)
    th.to_csv(res / "thresholds.csv", index=False)
    print(th.to_string(index=False))
    print("THRESHOLDS FROZEN (transistor)\n")

    test = list_test("transistor")
    files = [t["file"] for t in test]
    for name in TR_MODELS:
        size = 512 if name == "pc_512" else 256
        x = load_images(_read(files, "transistor", "test"), size=size, category="transistor")
        det = _load_tr(name)
        t0 = time.time()
        s, m = det.predict_many(x)
        print(f"[{name} transistor] scored {len(files)} test images in {time.time() - t0:.0f} s")
        np.savez_compressed(TR["checkpoints"] / f"{name}_heatmaps.npz", scores=s, heatmaps=m.astype(np.float32),
                            files=np.array(files))
        T = th.set_index("model").loc[name, "T_image"]
        pd.DataFrame({"file": files, "defect": [t["defect"] for t in test], "label": [t["label"] for t in test],
                      "score": s, "pred": (s > T).astype(int)}).to_csv(res / f"{name}_scores.csv", index=False)
        del det, x
        _free()


# ---- metrics from saved files ---------------------------------------------------------
def localisation(maps, masks):
    """Peak distance to the mask (px at 256), hit (<= 8 px) and the chance level, per defective image."""
    out = []
    for mp, mk in zip(maps, masks):
        dist = distance_transform_edt(mk == 0)
        py, px = np.unravel_index(np.argmax(mp), mp.shape)
        out.append((dist[py, px], dist[py, px] <= RADIUS, float((dist <= RADIUS).mean())))
    return np.array(out)


def evaluate(df, maps, masks, T, T_pix, types):
    """Overall row + per-type rows, the v1 metrics plus hit rate and its chance level."""
    labels, s = df.label.to_numpy(), df.score.to_numpy()
    bad = labels == 1
    loc = localisation(maps[bad], masks[bad])
    overall = {"group": "all", "n": int(bad.sum()), **image_metrics(labels, s, T),
               **pixel_metrics(masks, maps, T_pix), "hit_rate": loc[:, 1].mean(),
               "median_distance_px": float(np.median(loc[:, 0])), "chance_hit_rate": loc[:, 2].mean()}
    overall.pop("T"), overall.pop("T_pix")
    rows, good = [overall], ~bad
    dbad = df.defect.to_numpy()[bad]
    for d in types:
        sel = (df.defect == d).to_numpy()
        both = good | sel
        px = pixel_metrics(masks[sel], maps[sel], T_pix)
        ld = loc[dbad == d]
        rows.append({"group": d, "n": int(sel.sum()), "auroc": roc_auc_score(labels[both], s[both]),
                     "recall": float((s[sel] > T).mean()), "TP": int((s[sel] > T).sum()),
                     "iou": px["iou"], "dice": px["dice"], "hit_rate": ld[:, 1].mean(),
                     "median_distance_px": float(np.median(ld[:, 0])), "chance_hit_rate": ld[:, 2].mean()})
    return pd.DataFrame(rows), loc


def bootstrap(dfs, ious, hits, pairs, label_col="label"):
    """Stratified bootstrap (good and defective resampled separately), the same draws for all models."""
    y = next(iter(dfs.values()))[label_col].to_numpy()
    g_idx, b_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(SEED)
    draws = [(rng.choice(g_idx, len(g_idx)), rng.choice(len(b_idx), len(b_idx))) for _ in range(N_BOOT)]
    rows, aucs = [], {}
    for name, df in dfs.items():
        s, T = df.score.to_numpy(), df.attrs["T"]
        point = {"auroc": roc_auc_score(y, s), "recall": (s[b_idx] > T).mean(), "false_reject_rate": (s[g_idx] > T).mean(),
                 "iou": ious[name].mean(), "hit_rate": hits[name].mean()}
        b = {k: np.empty(N_BOOT) for k in point}
        for j, (g, d) in enumerate(draws):
            bi = b_idx[d]
            b["auroc"][j] = roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(bi))], np.r_[s[g], s[bi]])
            b["recall"][j] = (s[bi] > T).mean()
            b["false_reject_rate"][j] = (s[g] > T).mean()
            b["iou"][j] = ious[name][d].mean()
            b["hit_rate"][j] = hits[name][d].mean()
        aucs[name] = b["auroc"]
        for k, v in point.items():
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            rows.append({"model": name, "metric": k, "estimate": float(v), "ci95_low": lo, "ci95_high": hi})
    diffs = []
    for a, bname in pairs:
        if bname not in aucs:
            continue
        d = aucs[a] - aucs[bname]
        lo, hi = np.percentile(d, [2.5, 97.5])
        est = roc_auc_score(y, dfs[a].score) - roc_auc_score(y, dfs[bname].score)
        diffs.append({"difference": f"AUROC {a} - {bname}", "estimate": est, "ci95_low": lo, "ci95_high": hi,
                      "share_le_0": float((d <= 0).mean()), "n_resamples": N_BOOT})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def _per_image_iou(maps, masks, T_pix):
    pred = maps > T_pix
    return (masks & pred).sum(axis=(1, 2)) / (masks | pred).sum(axis=(1, 2))


def report_screw():
    test = list_test("screw")
    masks = np.stack([load_mask(t["mask"]) for t in test]).astype(bool)
    th = pd.read_csv(SCREW_OUT / "thresholds.csv", comment="#").iloc[0]
    npz = np.load(CHECKPOINTS / "v2" / "pc_512_screw_test_heatmaps.npz")
    df = pd.read_csv(SCREW_OUT / "pc_512_scores.csv", comment="#")
    assert list(df.file) == [t["file"] for t in test] == list(npz["files"])
    types = get_category("screw")["defect_types"]
    tab, loc = evaluate(df, npz["heatmaps"], masks, th.T_image, th.T_pixel, types)
    tab.insert(0, "model", "pc_512")
    v1 = pd.read_csv(RESULTS / "pc_raw_scores.csv")
    v1_maps = np.load(CHECKPOINTS / "pc_raw_heatmaps.npz")["heatmaps"]
    t1 = pd.read_csv(RESULTS / "thresholds.csv").set_index("variant").loc["pc_raw"]
    tab1, loc1 = evaluate(v1, v1_maps, masks, t1.T_image, t1.T_pixel, types)
    tab1.insert(0, "model", "pc_raw (v1, official)")
    tab = pd.concat([tab, tab1])
    _write_csv(tab, SCREW_OUT / "metrics.csv", SCREW_LABEL)
    bad = df.label.to_numpy() == 1
    df.attrs["T"], v1.attrs["T"] = th.T_image, t1.T_image
    ci, diff = bootstrap({"pc_512": df, "pc_raw": v1},
                         {"pc_512": _per_image_iou(npz["heatmaps"][bad], masks[bad], th.T_pixel),
                          "pc_raw": _per_image_iou(v1_maps[bad], masks[bad], t1.T_pixel)},
                         {"pc_512": loc[:, 1], "pc_raw": loc1[:, 1]}, [("pc_512", "pc_raw")])
    _write_csv(ci, SCREW_OUT / "bootstrap_ci.csv", SCREW_LABEL)
    _write_csv(diff, SCREW_OUT / "bootstrap_difference.csv", SCREW_LABEL)
    _print_tables("SCREW TEST, pc_512 vs v1 pc_raw | " + SCREW_LABEL, tab, ci, diff)
    plot_roc({"pc_512": df, "pc_raw (v1)": v1}, SCREW_OUT / "roc.png", SCREW_LABEL)
    plot_per_type(tab, types, SCREW_OUT / "per_type.png", SCREW_LABEL)
    return tab, ci, diff


def report_transistor():
    res, figd = TR["results"], TR["figures"]
    types = TR["defect_types"]
    test = list_test("transistor")
    masks = np.stack([load_mask(t["mask"], category="transistor") for t in test]).astype(bool)
    th = pd.read_csv(res / "thresholds.csv").set_index("model")
    dfs, maps, tabs, ious, hits = {}, {}, [], {}, {}
    for name in TR_MODELS:
        df = pd.read_csv(res / f"{name}_scores.csv")
        npz = np.load(TR["checkpoints"] / f"{name}_heatmaps.npz")
        assert list(df.file) == [t["file"] for t in test] == list(npz["files"])
        tab, loc = evaluate(df, npz["heatmaps"], masks, th.loc[name, "T_image"], th.loc[name, "T_pixel"], types)
        tab.insert(0, "model", name)
        tabs.append(tab)
        bad = df.label.to_numpy() == 1
        df.attrs["T"] = th.loc[name, "T_image"]
        dfs[name], maps[name] = df, npz["heatmaps"]
        ious[name] = _per_image_iou(npz["heatmaps"][bad], masks[bad], th.loc[name, "T_pixel"])
        hits[name] = loc[:, 1]
    tab = pd.concat(tabs)
    _write_csv(tab, res / "metrics.csv", TR_LABEL)
    ci, diff = bootstrap(dfs, ious, hits, [("pc_raw", "ae_raw"), ("pc_512", "pc_raw")])
    _write_csv(ci, res / "bootstrap_ci.csv", TR_LABEL)
    _write_csv(diff, res / "bootstrap_difference.csv", TR_LABEL)
    _print_tables("TRANSISTOR TEST | " + TR_LABEL, tab, ci, diff)
    print("\nthresholds (40 transistor val images):")
    print(th.round(5).to_string())
    print("\nfalse rejects at T_image (of 60 good):",
          {n: int(((d.label == 0) & (d.score > th.loc[n, "T_image"])).sum()) for n, d in dfs.items()})
    plot_roc(dfs, figd / "eval_roc.png", TR_LABEL)
    plot_per_type(tab, types, figd / "eval_per_type.png", TR_LABEL)
    plot_examples(dfs, maps, masks, test, th, figd / "eval_examples.png")
    return tab, ci, diff


def _print_tables(title, tab, ci, diff):
    pd.set_option("display.width", 250)
    print(f"\n== {title} ==")
    cols = ["model", "group", "n", "auroc", "precision", "recall", "f1", "false_reject_rate", "FP", "TP",
            "pixel_auroc", "iou", "dice", "hit_rate", "chance_hit_rate", "median_distance_px"]
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    print("\nbootstrap 95% CIs (stratified, 2000, seed 42):")
    print(ci.round(3).to_string(index=False))
    print("\npaired differences (descriptive):")
    print(diff.round(4).to_string(index=False))


# ---- figures ----------------------------------------------------------------------------
def plot_roc(dfs, path, label):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for name, df in dfs.items():
        fpr, tpr, _ = roc_curve(df.label, df.score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUROC {roc_auc_score(df.label, df.score):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="guessing")
    ax.set_xlabel("false-reject rate (good parts rejected)")
    ax.set_ylabel("recall (defective parts caught)")
    ax.set_title(f"ROC curve\n{label}", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_per_type(tab, types, path, label):
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(tab.model))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, col, lab in ((axes[0], "auroc", "image AUROC vs the good test images"),
                         (axes[1], "recall", "recall at T_image"),
                         (axes[2], "hit_rate", f"hit rate (peak within {RADIUS} px); black = chance")):
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            r = tab[tab.model == m].set_index("group").loc[types]
            x = np.arange(len(types)) + (k - (len(models) - 1) / 2) * w
            ax.bar(x, r[col], w * 0.92, label=m, color=f"C{k}")
            if col == "hit_rate":
                ax.scatter(x, r.chance_hit_rate, marker="_", s=120, linewidths=2.5, color="k", zorder=3)
        ax.set_xticks(range(len(types)), types, fontsize=8, rotation=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=len(models), fontsize=9, frameon=False)
    fig.suptitle(f"Per defect type (10-25 images each: descriptive only) | {label}", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_examples(dfs, maps, masks, test, th, path):
    """Fixed rule: per defect type the first image of the folder; plus the highest-scoring
    false reject of pc_raw (if any). Columns: image + mask, then each model's heat-map."""
    import matplotlib.pyplot as plt
    rows = [(f"{d}", next(i for i, t in enumerate(test) if t["defect"] == d)) for d in TR["defect_types"]]
    pc = dfs["pc_raw"]
    fr = pc[(pc.label == 0) & (pc.score > th.loc["pc_raw", "T_image"])]
    if len(fr):
        rows.append(("pc_raw false reject (highest)", int(fr.score.idxmax())))
    fig, axes = plt.subplots(len(rows), 1 + len(TR_MODELS), figsize=(3.2 * (1 + len(TR_MODELS)), 3.2 * len(rows)))
    for ax, (title, i) in zip(axes, rows):
        img = to_display(load_images([test[i]["file"]], category="transistor")[0])
        ax[0].imshow(img)
        if masks[i].any():
            ax[0].contour(masks[i], levels=[0.5], colors="yellow", linewidths=1)
        ax[0].set_title(f"{title}\n{test[i]['file'].split('/', 1)[1]}", fontsize=8)
        for k, name in enumerate(TR_MODELS, start=1):
            T = th.loc[name, "T_image"]
            s = dfs[name].score[i]
            ax[k].imshow(img)
            ax[k].imshow(maps[name][i], cmap="jet", alpha=0.5)
            ax[k].set_title(f"{name}: score/T {s / T:.2f} -> {'DEFECTIVE' if s > T else 'good'}", fontsize=8,
                            color="red" if s > T else "green")
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"Transistor test examples (fixed rule) with heat-maps | {TR_LABEL}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["screw", "transistor"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    for d in (SCREW_OUT, TR["results"], TR["figures"]):
        d.mkdir(parents=True, exist_ok=True)
    name = f"score_{args.score}" if args.score else "report"
    log_dir = SCREW_OUT if args.score == "screw" else TR["results"]
    sys.stdout = _Tee(sys.__stdout__, open(log_dir / f"phase4_{name}_log.txt", "w", encoding="utf-8"))
    t0 = time.time()
    print(f"phase4.py --{name.replace('_', ' ')} started {datetime.now():%Y-%m-%d %H:%M}")
    code0, earlier0 = _code_hashes(), _earlier_hashes()
    print("sha256 of decisions.md and all code before the run:")
    for k, v in code0.items():
        print(f"  {v}  {k}")
    print(f"hashed {len(earlier0)} earlier files")
    if args.score == "screw":
        score_screw()
    elif args.score == "transistor":
        score_transistor()
    else:
        report_screw()
        report_transistor()
    print("\nfiles read (count per folder):", FILES_READ)
    earlier1 = _earlier_hashes()
    changed = [k for k, v in earlier0.items() if earlier1.get(k) != v]
    print(f"earlier files: {len(earlier0)} hashed, {len(changed)} changed {changed}")
    assert not changed, "an earlier file changed!"
    assert _code_hashes() == code0, "code or decisions.md changed during the run!"
    print("code and decisions.md unchanged during the run")
    print(f"run time: {(time.time() - t0) / 60:.1f} min")
