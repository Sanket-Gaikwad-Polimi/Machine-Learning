"""Per-category settings (v2, phase 3a).

One entry per MVTec AD category. Every value was decided by reasoning, before any
model was run on the category, never from scores; the reasons are in decisions.md
(v2, Phase 3a). screw reproduces v1 exactly.

  get_category(name)            the settings + folders of one category (clear error if unknown)
  require_background_mask(name) stops if a background mask is not allowed for the category
"""
from src.paths import category_dirs

CATEGORIES = {
    "screw": {
        "color": "L",                     # greyscale, as in v1
        "n_val": 40,
        "ae_augment": "rot_flip",         # v1: random rotation 0-360 deg + flips
        "background_mask_allowed": True,  # plain background; a rotated screw is still good
        "target_brightness": 0.72,        # v1: mean grey value of the 280 train images
        "defect_types": ["manipulated_front", "scratch_head", "scratch_neck",
                         "thread_side", "thread_top"],
        "split_ready": True,              # outputs/results/split.json exists since v1
    },
    "transistor": {
        "color": "RGB",                   # case and leads differ in colour
        "n_val": 40,                      # 173 train / 40 val of 213 good images (seed 42)
        "ae_augment": "none",             # a rotated / shifted transistor is itself a defect
        "background_mask_allowed": False, # a missing / misplaced part shows on the background
        "target_brightness": None,        # set from its 173 train images when the split is made
        "defect_types": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
        "split_ready": False,             # the split is created at the start of phase 4, not now
    },
}


def get_category(name):
    """Settings of one category plus its folders (data, results, figures, checkpoints, split_file)."""
    if name not in CATEGORIES:
        raise ValueError(f"unknown category {name!r}; known: {', '.join(CATEGORIES)}")
    dirs = category_dirs(name)
    return {"name": name, **CATEGORIES[name], **dirs,
            "channels": 3 if CATEGORIES[name]["color"] == "RGB" else 1,
            "split_file": dirs["results"] / "split.json"}


def require_background_mask(name):
    """Call before using any background mask: stops for categories where it could hide defects."""
    if not get_category(name)["background_mask_allowed"]:
        raise ValueError(f"a background mask is not allowed for {name!r} "
                         "(it could hide missing or misplaced parts, see decisions.md v2 Phase 3a)")


if __name__ == "__main__":
    # Self-check: python -m src.categories (reads no image)
    for n in CATEGORIES:
        c = get_category(n)
        print(f"{n:10s} {c['color']:3s} channels={c['channels']} n_val={c['n_val']} augment={c['ae_augment']:8s} "
              f"background_mask={c['background_mask_allowed']!s:5s} split_ready={c['split_ready']!s:5s} "
              f"split_file={c['split_file']}")
