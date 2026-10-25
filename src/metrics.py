"""Thresholds and evaluation metrics, the same for both models.

  threshold_from_val()   threshold T from the 40 good VALIDATION images only
  image_metrics()        AUROC, precision, recall, F1, confusion matrix at T
  pixel_metrics()        pixel AUROC, IoU, Dice, pixel precision/recall at T_pix
  per_defect_table()     recall and AUROC for each defect type
  plot_roc()             ROC curves of one or more models in one figure

Every function takes plain numpy arrays (or a pandas table), so it does not
matter which model produced the scores and heat-maps.

Golden rule: thresholds come ONLY from validation (good) images. Test images
are used once, at the end, to compute these numbers, never to choose anything.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.paths import FIGURES

# Quantiles decided a priori (before seeing any test result):
#  - image level: T = 99th percentile of the 40 validation scores, i.e. we accept
#    rejecting about 1% of good screws;
#  - pixel level: T_pix = 99.9th percentile of all validation heat-map pixels.
#    It is stricter because defects are tiny (median 0.29% of the image, see EDA):
#    at 99% the false pixels alone would outnumber the defect pixels.
Q_IMAGE = 0.99
Q_PIXEL = 0.999


def threshold_from_val(val_values, q=Q_IMAGE):
    """Threshold = q-quantile of validation scores (or heat-map pixels, any shape)."""
    return float(np.quantile(np.ravel(val_values), q))


def image_metrics(labels, scores, T):
    """Image-level metrics. Decision: "defective" if score > T.

    false_reject_rate  = good screws wrongly rejected / all good screws
    missed_defect_rate = defective screws let through / all defective screws
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores > T).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"auroc": float(roc_auc_score(labels, scores)), "T": float(T),
            "precision": precision, "recall": recall, "f1": f1,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "false_reject_rate": fp / (fp + tn) if fp + tn else 0.0,
            "missed_defect_rate": fn / (fn + tp) if fn + tp else 0.0}


def pixel_metrics(masks, heatmaps, T_pix, max_pixels=2_000_000, seed=0):
    """Pixel-level metrics. masks (0/1) and heatmaps are N x H x W.

    pixel AUROC: computed on a random subsample of max_pixels pixels when there
                 are more, to keep memory low (the result barely changes).
    IoU, Dice:   per image, averaged over DEFECTIVE images only (a good image has
                 an empty mask, so its IoU is not defined).
    pixel precision/recall: over all pixels of all images.
    """
    masks = np.asarray(masks).astype(bool)
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    flat_m, flat_h = masks.ravel(), heatmaps.ravel()
    if flat_m.size > max_pixels:
        idx = np.random.default_rng(seed).choice(flat_m.size, max_pixels, replace=False)
        flat_m, flat_h = flat_m[idx], flat_h[idx]
    auroc = float(roc_auc_score(flat_m, flat_h))

    pred = heatmaps > T_pix
    ious, dices = [], []
    for m, p in zip(masks, pred):
        if m.any():
            inter, union = (m & p).sum(), (m | p).sum()
            ious.append(inter / union)
            dices.append(2 * inter / (m.sum() + p.sum()))
    tp = (masks & pred).sum()
    return {"pixel_auroc": auroc, "T_pix": float(T_pix),
            "iou": float(np.mean(ious)) if ious else float("nan"),
            "dice": float(np.mean(dices)) if dices else float("nan"),
            "pixel_precision": float(tp / pred.sum()) if pred.sum() else 0.0,
            "pixel_recall": float(tp / masks.sum()) if masks.sum() else 0.0}


def per_defect_table(df, T):
    """One row per defect type: count, recall at T, AUROC against test/good.

    df needs columns: file, defect ("good" or a type), label (0/1), score.
    The "good" row shows how many good screws were wrongly rejected.
    """
    good = df[df.defect == "good"]
    rows = [{"defect": "good", "n": len(good), "flagged": int((good.score > T).sum()),
             "recall": np.nan, "auroc_vs_good": np.nan}]
    for d in sorted(set(df.defect) - {"good"}):
        bad = df[df.defect == d]
        both = pd.concat([good, bad])
        rows.append({"defect": d, "n": len(bad), "flagged": int((bad.score > T).sum()),
                     "recall": float((bad.score > T).mean()),
                     "auroc_vs_good": float(roc_auc_score(both.label, both.score))})
    return pd.DataFrame(rows)


def plot_roc(labels, scores_by_model, path):
    """ROC curves of several models in one figure, e.g. {"AE": s1, "PatchCore": s2}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(labels, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="guessing")
    ax.set_xlabel("false positive rate (good screws rejected)")
    ax.set_ylabel("true positive rate (defects caught)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    # Self-test on FAKE numbers only (no model, no real images): python -m src.metrics
    rng = np.random.default_rng(0)

    # 1) Sanity case: perfectly separated scores must give AUROC 1.0.
    perfect = image_metrics([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], T=0.5)
    print("perfect case: AUROC =", perfect["auroc"], "| F1 =", perfect["f1"])

    # 2) Fake image scores: good ~ N(0, 1), defective ~ N(2, 1).
    val_scores = rng.normal(0, 1, 40)
    T = threshold_from_val(val_scores)
    defects = ["good"] * 41 + [d for d, n in [("manipulated_front", 24), ("scratch_head", 24),
                                              ("scratch_neck", 25), ("thread_side", 23),
                                              ("thread_top", 23)] for _ in range(n)]
    labels = np.array([int(d != "good") for d in defects])
    scores = rng.normal(0, 1, len(labels)) + 2 * labels
    print(f"\nimage threshold T (from 40 fake val scores): {T:.3f}")
    for k, v in image_metrics(labels, scores, T).items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")

    df = pd.DataFrame({"file": [f"img{i}" for i in range(len(labels))],
                       "defect": defects, "label": labels, "score": scores})
    print("\nper defect type:")
    print(per_defect_table(df, T).round(3).to_string(index=False))

    # 3) Fake heat-maps: 64 x 64, defect = small square, heat-map = noise + 2 on the defect.
    def fake_maps(n, n_defective):
        masks = np.zeros((n, 64, 64), dtype=np.uint8)
        for i in range(n_defective):
            r, c = rng.integers(0, 56, 2)
            masks[i, r:r + 8, c:c + 8] = 1
        return masks, rng.normal(0, 1, masks.shape) + 2 * masks

    _, val_maps = fake_maps(40, 0)
    T_pix = threshold_from_val(val_maps, Q_PIXEL)
    test_masks, test_maps = fake_maps(160, 119)
    print(f"\npixel threshold T_pix (from fake val heat-maps): {T_pix:.3f}")
    for k, v in pixel_metrics(test_masks, test_maps, T_pix).items():
        print(f"  {k:16s} {v:.3f}")

    # 4) ROC figure with two fake "models".
    path = FIGURES / "metrics_selftest_roc.png"
    plot_roc(labels, {"fake strong": scores, "fake weak": scores / 2 + rng.normal(0, 1, len(labels))}, path)
    print("\nsaved", path)
