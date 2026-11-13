"""EDA of the MVTec AD transistor data (v2, phase 4a): python eda_transistor.py

Describes the data only, as the v1 EDA did for screw: image size and colour, one example
per defect type with its mask, defect areas, brightness of train/good vs test/good.
It looks at test images and masks, but NO model is run and nothing is chosen from it
(decisions.md, v2 Phase 4). Outputs: outputs/transistor/figures/eda_transistor.png and
outputs/transistor/results/eda_*.csv.
"""
import numpy as np
import pandas as pd
from PIL import Image

from src.categories import get_category
from src.data import get_split, list_test, load_image, load_mask

CAT = "transistor"
LABEL = "transistor EDA (data description only, no model)"


def _write_csv(df, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        df.to_csv(f, index=False, lineterminator="\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cat = get_category(CAT)
    for d in (cat["results"], cat["figures"]):
        d.mkdir(parents=True, exist_ok=True)
    data, types = cat["data"], cat["defect_types"]
    split, test = get_split(CAT), list_test(CAT)

    # 1. size and colour of every image
    files = sorted(p for p in data.rglob("*.png") if "ground_truth" not in p.parts)
    fmt = pd.Series([f"{im.size[0]}x{im.size[1]} {im.mode}" for im in (Image.open(p) for p in files)]).value_counts()
    print(f"images: {len(files)} ({len(split['train'])} train + {len(split['val'])} val + {len(test)} test)")
    print("size and colour mode:", fmt.to_dict())
    counts = pd.Series([t["defect"] for t in test]).value_counts().reindex(["good"] + types)
    print("test images per type:", counts.to_dict())

    # 2. defect areas (% of the image), at 1024 and at 256 after nearest-neighbour resize
    rows = [{"defect": t["defect"], "file": t["file"],
             "area_pct_1024": 100 * load_mask(t["mask"], size=1024, category=CAT).mean(),
             "area_pct_256": 100 * load_mask(t["mask"], size=256, category=CAT).mean()}
            for t in test if t["label"] == 1]
    area = pd.DataFrame(rows)
    _write_csv(area.round(4), cat["results"] / "eda_defect_area.csv")
    stats = area.groupby("defect")[["area_pct_1024", "area_pct_256"]].agg(["min", "median", "max"]).loc[types]
    stats.loc["all"] = area[["area_pct_1024", "area_pct_256"]].agg(["min", "median", "max"]).T.stack().values
    pd.set_option("display.width", 200)
    print("\ndefect area, % of the image:")
    print(stats.round(3).to_string())

    # 3. brightness per channel (images at 256 px, as the models see them), mean of each image
    groups = {"train/good (213)": split["train"] + split["val"],
              "test/good (60)": [t["file"] for t in test if t["label"] == 0],
              "test defective (40)": [t["file"] for t in test if t["label"] == 1]}
    brows = []
    for g, fs in groups.items():
        means = np.stack([load_image(f, category=CAT).mean(axis=(1, 2)) for f in fs])   # N x 3
        for c, ch in enumerate("RGB"):
            brows.append({"group": g, "channel": ch, "mean": means[:, c].mean(), "std": means[:, c].std(ddof=1)})
        brows.append({"group": g, "channel": "all", "mean": means.mean(), "std": means.mean(axis=1).std(ddof=1)})
    bright = pd.DataFrame(brows)
    _write_csv(bright.round(4), cat["results"] / "eda_brightness.csv")
    print("\nbrightness (mean of each 256 px image, 0-1), mean +- std over images:")
    print(bright.pivot(index="group", columns="channel", values="mean")[["R", "G", "B", "all"]]
          .loc[list(groups)].round(4).to_string())
    print(bright.pivot(index="group", columns="channel", values="std")[["R", "G", "B", "all"]]
          .loc[list(groups)].round(4).rename(columns=lambda c: f"std {c}").to_string())

    # 4. figure: one example per type (fixed rule: the first file of each folder), mask overlay,
    #    defect areas, brightness
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.05])
    ex = [next(t for t in test if t["defect"] == d) for d in ["good"] + types]
    for k, t in enumerate(ex):
        img = load_image(t["file"], size=1024, category=CAT).transpose(1, 2, 0)
        m = load_mask(t["mask"], size=1024, category=CAT)
        a0, a1 = fig.add_subplot(gs[0, k]), fig.add_subplot(gs[1, k])
        a0.imshow(img)
        a0.set_title(f"{t['defect']}\n{t['file'].split('/', 1)[1]}", fontsize=9)
        a1.imshow(img)
        if m.any():
            a1.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.55)
            a1.contour(m, levels=[0.5], colors="yellow", linewidths=0.8)
            a1.set_title(f"mask: {100 * m.mean():.2f}% of the image", fontsize=9)
        else:
            a1.set_title("good: no mask", fontsize=9)
        for a in (a0, a1):
            a.axis("off")
    ax = fig.add_subplot(gs[2, 0:3])
    ax.boxplot([area[area.defect == d].area_pct_1024 for d in types], tick_labels=types)
    for k, d in enumerate(types, start=1):
        v = area[area.defect == d].area_pct_1024
        ax.scatter(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, s=12, alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("defect area, % of the image (1024 px mask, log)")
    ax.set_title("defect size per type (10 images each)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax = fig.add_subplot(gs[2, 3:5])
    w = 0.25
    for k, g in enumerate(groups):
        b = bright[(bright.group == g) & (bright.channel != "all")]
        ax.bar(np.arange(3) + (k - 1) * w, b["mean"], w * 0.9, yerr=b["std"], capsize=3, label=g)
    ax.set_xticks(range(3), ["R", "G", "B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean pixel value per image (0-1), +- std")
    ax.set_title("brightness per colour channel (256 px)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"MVTec AD transistor: one example per type (first file of each folder), defect sizes, "
                 f"brightness | {LABEL}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = cat["figures"] / "eda_transistor.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print("\nsaved", out)
