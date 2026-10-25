"""Loading the MVTec AD screw images and masks.

What this file gives the rest of the project:
  get_split()        280 train / 40 val good images (made once, saved to split.json)
  list_test()        every test image with its defect type, label and mask path
  load_image()       one greyscale image -> float array H x W in [0, 1]
  load_mask()        one mask -> 0/1 array H x W (all zeros for good images)
  load_images()      many images -> array N x H x W
  to_tensor()        array(s) -> PyTorch tensor with 3 channels (+ ImageNet normalisation)

Both models receive plain numpy arrays and call to_tensor() themselves,
so metrics.py and evaluate.py never need to know which model is used.
"""
import json

import numpy as np
from PIL import Image

from src.paths import DATA_DIR, RESULTS

IMG_SIZE = 256          # default working resolution (images are 1024 x 1024)
SEED = 42               # fixed seed for the train/val split
N_VAL = 40              # held-out good images used ONLY for the threshold
SPLIT_FILE = RESULTS / "split.json"
DEFECT_TYPES = ["manipulated_front", "scratch_head", "scratch_neck",
                "thread_side", "thread_top"]

# ImageNet statistics, needed because the pretrained backbone was trained on them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_split():
    """Return {"train": [...], "val": [...]} as paths relative to screw/.

    The first call shuffles train/good with a fixed seed and saves the result;
    every later call just reloads split.json, so all scripts use the same split.
    """
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    files = sorted(p.relative_to(DATA_DIR).as_posix()
                   for p in (DATA_DIR / "train" / "good").glob("*.png"))
    order = np.random.default_rng(SEED).permutation(len(files))
    split = {"seed": SEED,
             "train": sorted(files[i] for i in order[N_VAL:]),
             "val": sorted(files[i] for i in order[:N_VAL])}
    SPLIT_FILE.write_text(json.dumps(split, indent=2))
    return split


def list_test():
    """One dict per test image: file, defect ("good" or a type), label (0/1), mask."""
    items = []
    for folder in sorted((DATA_DIR / "test").iterdir()):
        for p in sorted(folder.glob("*.png")):
            defect = folder.name
            mask = None if defect == "good" else \
                f"ground_truth/{defect}/{p.stem}_mask.png"
            items.append({"file": p.relative_to(DATA_DIR).as_posix(),
                          "defect": defect,
                          "label": int(defect != "good"),
                          "mask": mask})
    return items


def load_image(rel_path, size=IMG_SIZE):
    """Greyscale image as float32 H x W in [0, 1], resized with bilinear filtering."""
    img = Image.open(DATA_DIR / rel_path).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(rel_path, size=IMG_SIZE):
    """Mask as uint8 H x W with 0 = normal, 1 = defect.

    Nearest-neighbour resizing keeps the mask strictly binary (no grey edges).
    A good image has no mask file, so it gets an all-zero mask.
    """
    if rel_path is None:
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(DATA_DIR / rel_path).convert("L")
    m = m.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def load_images(rel_paths, size=IMG_SIZE):
    """Stack several images into one float32 array N x H x W."""
    return np.stack([load_image(p, size) for p in rel_paths])


def to_tensor(images, imagenet=False):
    """numpy H x W or N x H x W in [0, 1] -> torch tensor N x 3 x H x W.

    The grey channel is repeated 3 times because pretrained networks expect RGB.
    imagenet=True also applies ImageNet normalisation (for PatchCore's backbone).
    """
    import torch
    x = torch.as_tensor(np.asarray(images, dtype=np.float32))
    if x.ndim == 2:
        x = x[None]                                   # single image -> batch of 1
    x = x[:, None].repeat(1, 3, 1, 1)                 # N x 3 x H x W
    if imagenet:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


if __name__ == "__main__":
    # Self-check: python -m src.data
    split = get_split()
    print("train:", len(split["train"]), "| val:", len(split["val"]),
          "| overlap:", len(set(split["train"]) & set(split["val"])))

    test = list_test()
    counts = {}
    for t in test:
        counts[t["defect"]] = counts.get(t["defect"], 0) + 1
    print("test images:", len(test), counts)
    missing = [t["mask"] for t in test if t["mask"] and not (DATA_DIR / t["mask"]).exists()]
    print("missing masks:", len(missing))

    img = load_image(split["train"][0])
    print("image:", img.shape, img.dtype, f"min={img.min():.2f} max={img.max():.2f}")
    first_defect = next(t for t in test if t["label"] == 1)
    mask = load_mask(first_defect["mask"])
    print("mask:", first_defect["mask"], mask.shape, "values:", np.unique(mask).tolist(),
          f"defect area={100 * mask.mean():.2f}%")
    x = to_tensor(img, imagenet=True)
    print("tensor:", tuple(x.shape), f"mean={x.mean().item():.2f}")
