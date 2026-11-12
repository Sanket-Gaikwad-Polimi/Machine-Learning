"""POST-HOC ANALYSIS OF THE FROZEN v1 RESULTS (v2, phase 1): python analysis.py

Three questions about the v1 test run, answered from the SAVED files only:
  1. Localisation: does the heat-map's maximum point at the real defect?
  2. Threshold trade-off: what would other thresholds have given? (what-if only,
     the official T_image stays as it is)
  3. Uncertainty: how precise are the v1 numbers with only 41 + 119 test images?

Reads ONLY: outputs/results/<variant>_scores.csv, <variant>_val_scores.csv,
thresholds.csv, metrics.csv, the test heat-maps outputs/checkpoints/<variant>_heatmaps.npz
(from the Release) and the ground-truth masks. No model is loaded, nothing is re-scored.
All decisions (8 px radius, false-reject levels, bootstrap settings) are in
decisions.md, section "v2", written before this script was run.

Writes only into outputs/analysis/ (CSV + PNG). Every file of outputs/results,
outputs/figures and outputs/checkpoints is hashed (SHA-256) before and after to
prove that v1 is untouched.
"""
import hashlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.data import DEFECT_TYPES, list_test, load_mask
from src.paths import CHECKPOINTS, FIGURES, OUTPUTS, RESULTS

VARIANTS = ["ae_raw", "ae_bn", "pc_raw", "pc_bn"]
ANALYSIS = OUTPUTS / "analysis"
LABEL = "post-hoc analysis of frozen v1 results"

# Fixed a priori (decisions.md, v2 phase 1)
RADIUS = 8                        # px at 256 px: one PatchCore cell
TARGET_FRR = [0.0, 0.025, 0.05, 0.10]
N_BOOT, SEED = 2000, 42

# Same colour per variant in every figure; autoencoder dashed, PatchCore solid
COLORS = {"ae_raw": "C0", "ae_bn": "C1", "pc_raw": "C2", "pc_bn": "C3"}
STYLES = {"ae_raw": "--", "ae_bn": "--", "pc_raw": "-", "pc_bn": "-"}


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


def _frozen_files():
    """Every v1 file: results, figures (eval_*, diag_*, ...) and checkpoints."""
    return sorted(p for d in (RESULTS, FIGURES, CHECKPOINTS) for p in d.rglob("*") if p.is_file())


def _hashes():
    return {p.relative_to(OUTPUTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in _frozen_files()}


def _write_csv(df, name):
    """CSV whose first line says what it is (read it back with pd.read_csv(..., comment="#"))."""
    with open(ANALYSIS / name, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        df.to_csv(f, index=False, lineterminator="\n")


def load_v1():
    """The saved v1 test results: scores, heat-maps, thresholds, masks."""
    test = list_test()
    files = [t["file"] for t in test]
    masks = np.stack([load_mask(t["mask"]) for t in test]).astype(bool)
    th = pd.read_csv(RESULTS / "thresholds.csv").set_index("variant")
    v1 = {}
    for name in VARIANTS:
        df = pd.read_csv(RESULTS / f"{name}_scores.csv")
        npz = np.load(CHECKPOINTS / f"{name}_heatmaps.npz")
        assert list(df.file) == files and list(npz["files"]) == files, f"{name}: file order differs"
        val = pd.read_csv(RESULTS / f"{name}_val_scores.csv").score.to_numpy()
        v1[name] = {"df": df, "maps": npz["heatmaps"], "val": val,
                    "T": float(th.loc[name, "T_image"]), "T_pix": float(th.loc[name, "T_pixel"])}
    return test, masks, v1


def _iou_per_image(masks, maps, T_pix):
    """IoU of (heat-map > T_pixel) with the mask, per image (as in src/metrics.py)."""
    pred = maps > T_pix
    inter = (masks & pred).sum(axis=(1, 2))
    union = (masks | pred).sum(axis=(1, 2))
    return inter / union


def self_check(masks, v1):
    """Recompute the main v1 numbers from the saved files: they must equal metrics.csv."""
    official = pd.read_csv(RESULTS / "metrics.csv").set_index("variant")
    bad = masks.any(axis=(1, 2))
    for name, r in v1.items():
        y, s = r["df"].label.to_numpy(), r["df"].score.to_numpy()
        mine = {"auroc": roc_auc_score(y, s), "recall": (s[y == 1] > r["T"]).mean(),
                "false_reject_rate": (s[y == 0] > r["T"]).mean(),
                "iou": _iou_per_image(masks[bad], r["maps"][bad], r["T_pix"]).mean()}
        for k, v in mine.items():
            assert abs(v - official.loc[name, k]) < 1e-9, f"{name} {k}: {v} != {official.loc[name, k]}"
    print("self-check: AUROC, recall, false-reject rate and IoU recomputed = metrics.csv (all 4 variants)")


# ---- 1. Localisation --------------------------------------------------------------
def localisation(test, masks, v1):
    bad = [i for i, t in enumerate(test) if t["label"] == 1]
    # distance of every pixel to the nearest defect pixel (0 inside the mask), per image
    dist = {i: distance_transform_edt(~masks[i]) for i in bad}
    chance = {i: float((dist[i] <= RADIUS).mean()) for i in bad}
    rows = []
    for name, r in v1.items():
        for i in bad:
            y, x = np.unravel_index(np.argmax(r["maps"][i]), r["maps"][i].shape)
            d = float(dist[i][y, x])
            rows.append({"variant": name, "file": test[i]["file"], "defect": test[i]["defect"],
                         "detected": int(r["df"].score[i] > r["T"]), "peak_y": int(y), "peak_x": int(x),
                         "distance_px": round(d, 3), "hit": int(d <= RADIUS),
                         "chance": round(chance[i], 5)})
    per_image = pd.DataFrame(rows)

    summary = []
    for name in VARIANTS:
        p = per_image[per_image.variant == name]
        groups = [(d, p[p.defect == d]) for d in DEFECT_TYPES]
        groups += [("all defective", p), ("detected (TP)", p[p.detected == 1])]
        for g, q in groups:
            summary.append({"variant": name, "group": g, "n": len(q), "hits": int(q.hit.sum()),
                            "hit_rate": q.hit.mean() if len(q) else np.nan,
                            "median_distance_px": q.distance_px.median() if len(q) else np.nan,
                            "chance_hit_rate": q.chance.mean() if len(q) else np.nan})
    summary = pd.DataFrame(summary)
    return per_image, summary


def plot_localisation(summary, path):
    import matplotlib.pyplot as plt
    groups = DEFECT_TYPES + ["all defective", "detected (TP)"]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    w = 0.2
    for k, name in enumerate(VARIANTS):
        s = summary[summary.variant == name].set_index("group").loc[groups]
        x = np.arange(len(groups)) + (k - 1.5) * w
        ax.bar(x, s.hit_rate, w * 0.9, color=COLORS[name], label=name,
               hatch="//" if name.startswith("ae") else None, edgecolor="white", linewidth=0.5)
        ax.scatter(x, s.chance_hit_rate, marker="_", s=150, linewidths=2.5, color="k", zorder=3,
                   label="chance level (random peak)" if k == 0 else None)
        for xi, rate, h, n in zip(x, s.hit_rate, s.hits, s.n):
            ax.text(xi, rate + 0.015, f"{h}/{n}", rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(groups)), groups, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel(f"hit rate: heat-map maximum within {RADIUS} px of the defect")
    ax.set_title(f"Does the heat-map maximum point at the defect? (test, 256 px)\n{LABEL}", fontsize=10)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---- 2. Threshold trade-off (hypothetical) ------------------------------------------
def tradeoff(v1):
    rows, alt = [], []
    for name, r in v1.items():
        y, s = r["df"].label.to_numpy(), r["df"].score.to_numpy()
        good, bad = np.sort(s[y == 0])[::-1], s[y == 1]
        for f in TARGET_FRR:
            k = int(np.floor(f * len(good) + 1e-9))      # false rejects allowed
            T = good[k]                                 # the (k+1)-th highest good score
            rows.append({"variant": name, "target_frr": f, "false_rejects_allowed": k,
                         "T_hypothetical": T, "frr": (good > T).mean(), "recall": (bad > T).mean(),
                         "detected": int((bad > T).sum())})
        val = r["val"]
        for rule, T in [("frozen: 99th percentile of val (official)", r["T"]),
                        ("max of val", val.max()),
                        ("val mean + 3 std (ddof 1)", val.mean() + 3 * val.std(ddof=1))]:
            alt.append({"variant": name, "rule": rule, "T": T, "T / T_official": T / r["T"],
                        "val_above_T": int((val > T).sum()), "test_frr": (good > T).mean(),
                        "test_recall": (bad > T).mean(), "test_detected": int((bad > T).sum())})
    return pd.DataFrame(rows), pd.DataFrame(alt)


def plot_roc_pr(v1, path):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for name, r in v1.items():
        y, s = r["df"].label.to_numpy(), r["df"].score.to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        frr0, rec0 = (s[y == 0] > r["T"]).mean(), (s[y == 1] > r["T"]).mean()
        tp, fp = ((s > r["T"]) & (y == 1)).sum(), ((s > r["T"]) & (y == 0)).sum()
        a1.plot(fpr, tpr, STYLES[name], color=COLORS[name], lw=2,
                label=f"{name} (AUROC {roc_auc_score(y, s):.3f})")
        a1.plot(frr0, rec0, "o", ms=8, color=COLORS[name], mec="k")
        a2.plot(rec, prec, STYLES[name], color=COLORS[name], lw=2, label=name)
        a2.plot(rec0, tp / (tp + fp), "o", ms=8, color=COLORS[name], mec="k")
    a1.plot([0, 1], [0, 1], "k:", lw=1, label="guessing")
    a1.plot([], [], "o", color="grey", mec="k", label="frozen T_image (official)")
    a1.set_xlabel("false-reject rate (good test screws rejected)")
    a1.set_ylabel("recall (defective test screws caught)")
    a1.set_title("ROC curve")
    a1.legend(fontsize=8, loc="lower right")
    a2.axhline((119 / 160), color="k", ls=":", lw=1, label="guessing (119/160 defective)")
    a2.set_xlabel("recall (defective test screws caught)")
    a2.set_ylabel("precision")
    a2.set_ylim(0.6, 1.02)
    a2.set_title("precision-recall curve")
    a2.legend(fontsize=8, loc="lower left")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle(f"What-if: every possible threshold on the test scores (the official T stays)\n{LABEL}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---- 3. Uncertainty: stratified, paired bootstrap ------------------------------------
def bootstrap(masks, v1):
    y = v1["ae_raw"]["df"].label.to_numpy()
    good_idx, bad_idx = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    rng = np.random.default_rng(SEED)
    # the same resampled images for every variant (paired)
    samples = [(rng.choice(good_idx, len(good_idx)), rng.choice(bad_idx, len(bad_idx)))
               for _ in range(N_BOOT)]
    boot, rows = {}, []
    for name, r in v1.items():
        s = r["df"].score.to_numpy()
        iou = np.full(len(s), np.nan)
        iou[bad_idx] = _iou_per_image(masks[bad_idx], r["maps"][bad_idx], r["T_pix"])
        point = {"auroc": roc_auc_score(y, s), "recall": (s[bad_idx] > r["T"]).mean(),
                 "false_reject_rate": (s[good_idx] > r["T"]).mean(), "iou": iou[bad_idx].mean()}
        b = {k: np.empty(N_BOOT) for k in point}
        for j, (g, d) in enumerate(samples):
            idx = np.r_[g, d]
            b["auroc"][j] = roc_auc_score(y[idx], s[idx])
            b["recall"][j] = (s[d] > r["T"]).mean()
            b["false_reject_rate"][j] = (s[g] > r["T"]).mean()
            b["iou"][j] = iou[d].mean()
        boot[name] = b
        for k, v in point.items():
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            rows.append({"variant": name, "metric": k, "estimate": v, "ci95_low": lo, "ci95_high": hi})
    diff = boot["pc_raw"]["auroc"] - boot["ae_raw"]["auroc"]
    point_diff = (pd.DataFrame(rows).query("metric == 'auroc'").set_index("variant").estimate)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    paired = pd.DataFrame([{"difference": "AUROC pc_raw - ae_raw",
                            "estimate": point_diff["pc_raw"] - point_diff["ae_raw"],
                            "ci95_low": lo, "ci95_high": hi,
                            "share_of_resamples_le_0": float((diff <= 0).mean()), "n_resamples": N_BOOT}])
    return pd.DataFrame(rows), paired


def plot_auroc_ci(ci, paired, path):
    import matplotlib.pyplot as plt
    a = ci[ci.metric == "auroc"].set_index("variant").loc[VARIANTS]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, name in enumerate(VARIANTS):
        ax.errorbar(k, a.estimate[name], yerr=[[a.estimate[name] - a.ci95_low[name]],
                                               [a.ci95_high[name] - a.estimate[name]]],
                    fmt="o", ms=8, capsize=6, lw=2, color=COLORS[name])
        ax.text(k + 0.08, a.estimate[name], f"{a.estimate[name]:.3f}\n"
                f"[{a.ci95_low[name]:.3f}, {a.ci95_high[name]:.3f}]", va="center", fontsize=8)
    p = paired.iloc[0]
    ax.set_xticks(range(len(VARIANTS)), VARIANTS)
    ax.set_xlim(-0.4, len(VARIANTS) - 0.2)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("image AUROC on the 160 test images")
    ax.set_title(f"AUROC with 95% bootstrap CI ({N_BOOT} stratified resamples)\n"
                 f"pc_raw - ae_raw = {p.estimate:.3f} [{p.ci95_low:.3f}, {p.ci95_high:.3f}]\n{LABEL}",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(ANALYSIS / "analysis_log.txt", "w", encoding="utf-8"))
    t0 = time.time()
    print(f"analysis.py started {datetime.now():%Y-%m-%d %H:%M} | {LABEL}")
    before = _hashes()
    print(f"hashed {len(before)} v1 files in outputs/results, outputs/figures, outputs/checkpoints")
    missing = [f"{v}_heatmaps.npz" for v in VARIANTS if not (CHECKPOINTS / f"{v}_heatmaps.npz").exists()]
    if missing:
        sys.exit(f"missing in outputs/checkpoints: {missing} (download the Release, see README.md)")

    test, masks, v1 = load_v1()
    self_check(masks, v1)
    pd.set_option("display.width", 200)

    print("\n== 1. LOCALISATION (heat-map maximum within 8 px of the defect, 256 px) ==")
    per_image, loc = localisation(test, masks, v1)
    _write_csv(per_image, "localisation_per_image.csv")
    _write_csv(loc, "localisation_summary.csv")
    print(loc.pivot(index="group", columns="variant", values="hit_rate")
          .loc[DEFECT_TYPES + ["all defective", "detected (TP)"], VARIANTS].round(3).to_string())
    print("\nhits / n:")
    print(loc.assign(hn=loc.hits.astype(str) + "/" + loc.n.astype(str))
          .pivot(index="group", columns="variant", values="hn")
          .loc[DEFECT_TYPES + ["all defective", "detected (TP)"], VARIANTS].to_string())
    print("\nmedian distance (px at 256) and chance level (all defective):")
    print(loc[loc.group == "all defective"].set_index("variant")
          [["median_distance_px", "chance_hit_rate"]].round(3).to_string())
    plot_localisation(loc, ANALYSIS / "analysis_localisation.png")

    print("\n== 2. THRESHOLD TRADE-OFF (hypothetical: the official T stays) ==")
    trade, alt = tradeoff(v1)
    _write_csv(trade, "threshold_tradeoff.csv")
    _write_csv(alt, "threshold_alternatives.csv")
    print("recall at a test false-reject rate of at most ...")
    print(trade.pivot(index="target_frr", columns="variant", values="recall")[VARIANTS].round(3).to_string())
    print("\nvalidation-only threshold rules (what-if on the test set):")
    print(alt.round(4).to_string(index=False))
    plot_roc_pr(v1, ANALYSIS / "analysis_roc_pr.png")

    print(f"\n== 3. UNCERTAINTY (bootstrap, {N_BOOT} stratified paired resamples, seed {SEED}) ==")
    ci, paired = bootstrap(masks, v1)
    _write_csv(ci, "bootstrap_ci.csv")
    _write_csv(paired, "bootstrap_auroc_difference.csv")
    print(ci.round(3).to_string(index=False))
    print()
    print(paired.round(4).to_string(index=False))
    plot_auroc_ci(ci, paired, ANALYSIS / "analysis_auroc_ci.png")

    after = _hashes()
    check = pd.DataFrame([{"file": f, "sha256_before": h, "sha256_after": after.get(f, ""),
                           "unchanged": h == after.get(f)} for f, h in before.items()])
    _write_csv(check, "v1_hash_check.csv")
    assert set(after) == set(before) and check.unchanged.all(), "a v1 file changed!"
    print(f"\nv1 untouched: all {len(before)} files in outputs/results, outputs/figures and "
          "outputs/checkpoints have identical sha256 before and after")
    print(f"run time: {time.time() - t0:.0f} s")
