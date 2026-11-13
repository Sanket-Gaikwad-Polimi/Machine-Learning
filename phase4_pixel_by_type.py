"""Per-type pixel AUROC on transistor, computed after Phase 4c from the SAVED heat-maps (no re-scoring).

Checks the part of Phase 4b prediction 5 that 4c left open ("misplaced dominates pixel AUROC").
Reads outputs/checkpoints/transistor/<model>_heatmaps.npz and the masks; no model is loaded.
  - pixel AUROC per defect type: the pixels of that type's 10 test images (mask = positive, the rest
    negative) plus all pixels of the 60 good test images (negative), at 256 px
  - pixel AUROC of all 100 test images (must equal the Phase 4c value), and without the 10 misplaced
    images (all 60 good kept)
  - each type's share of all defect pixels of the test set
Pixel AUROC as in src/metrics.pixel_metrics (random sample of 2 million pixels when there are more, seed 0).
Only the saved heat-map files and the ground-truth masks (labels) are read; no test image, no model.
Output: outputs/transistor/results/pixel_auroc_by_type.csv
"""
import hashlib

import numpy as np
import pandas as pd

from src.categories import get_category
from src.data import list_test, load_mask
from src.metrics import pixel_metrics

LABEL = "computed after 4c from saved heat-maps, no re-scoring"
TR = get_category("transistor")
MODELS = ("pc_raw", "ae_raw", "pc_512")

if __name__ == "__main__":
    npz_files = {n: TR["checkpoints"] / f"{n}_heatmaps.npz" for n in MODELS}
    sha0 = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in npz_files.items()}
    test = list_test("transistor")                       # file names only; no image is opened
    masks = np.stack([load_mask(t["mask"], category="transistor") for t in test]).astype(bool)
    print(f"read: {sum(t['mask'] is not None for t in test)} ground-truth masks (labels), 3 saved heat-map files; "
          "no test image, no model")
    defect = np.array([t["defect"] for t in test])
    good = defect == "good"
    share = {d: 100 * masks[defect == d].sum() / masks.sum() for d in TR["defect_types"]}
    print(f"defect pixels (256 px): {int(masks.sum())}; share per type (%):",
          {d: round(v, 1) for d, v in share.items()})
    official = pd.read_csv(TR["results"] / "metrics.csv", comment="#").query("group == 'all'").set_index("model")
    rows = []
    for name in MODELS:
        npz = np.load(npz_files[name])
        assert list(npz["files"]) == [t["file"] for t in test]
        maps = npz["heatmaps"]
        groups = {d: good | (defect == d) for d in TR["defect_types"]}
        groups["all 100 test images"] = np.ones(len(test), bool)
        groups["all except misplaced"] = defect != "misplaced"
        for g, sel in groups.items():
            rows.append({"label": LABEL, "model": name, "group": g, "n_images": int(sel.sum()),
                         "defect_pixel_share_of_test_pct": share.get(g, np.nan),
                         "pixel_auroc": pixel_metrics(masks[sel], maps[sel], T_pix=np.inf)["pixel_auroc"]})
        diff = rows[-2]["pixel_auroc"] - official.loc[name, "pixel_auroc"]
        print(f"[{name}] recomputed overall pixel AUROC {rows[-2]['pixel_auroc']:.6f} vs Phase 4c "
              f"{official.loc[name, 'pixel_auroc']:.6f}: difference {diff:.1e}")
        assert abs(diff) < 1e-9
    df = pd.DataFrame(rows)
    assert {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in npz_files.items()} == sha0
    print("saved heat-map files unchanged (sha256 before = after)")
    with open(TR["results"] / "pixel_auroc_by_type.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {LABEL}\n")
        df.to_csv(f, index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(f"\npixel AUROC ({LABEL}):")
    print(df.pivot(index="group", columns="model", values="pixel_auroc")
          .loc[TR["defect_types"] + ["all 100 test images", "all except misplaced"], ["pc_raw", "ae_raw", "pc_512"]]
          .round(3).to_string())
