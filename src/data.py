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

# Brightness normalisation (optional, OFF by default).
# EDA finding: test/good images are darker (mean ~0.665) than train/good (~0.72)
# and defective test images (~0.73), which can cause false alarms on good screws.
# Plan, decided a priori and never tuned on test results:
#  - load_image(..., normalize_brightness=True) shifts each image so its mean
#    equals TARGET_BRIGHTNESS, a fixed value taken from the 280 TRAIN images only;
#  - the robustness step checks score stability on the 40 VALIDATION images
#    under brightness changes;
#  - both models are reported with and without normalisation.
# Resolution: main results at IMG_SIZE = 256; a larger size only as a reported
# comparison, never chosen by test scores.
TARGET_BRIGHTNESS = 0.72  # mean pixel value of the 280 train/good images (EDA, rounded)

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


def load_image(rel_path, size=IMG_SIZE, normalize_brightness=False):
    """Greyscale image as float32 H x W in [0, 1], resized with bilinear filtering.

    normalize_brightness=True shifts the image so its mean is TARGET_BRIGHTNESS
    (then clips to [0, 1]), removing the overall light level difference.
    """
    img = Image.open(DATA_DIR / rel_path).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    img = np.asarray(img, dtype=np.float32) / 255.0
    if normalize_brightness:
        img = shift_brightness(img)
    return img


def shift_brightness(images):
    """Move the mean of each image (H x W or N x H x W) to TARGET_BRIGHTNESS, clip to [0, 1].

    Used by load_image and by the models, so both apply exactly the same rule.
    """
    images = np.asarray(images, dtype=np.float32)
    means = images.mean(axis=(-2, -1), keepdims=True)
    return np.clip(images - means + TARGET_BRIGHTNESS, 0.0, 1.0)


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


def load_images(rel_paths, size=IMG_SIZE, normalize_brightness=False):
    """Stack several images into one float32 array N x H x W."""
    return np.stack([load_image(p, size, normalize_brightness) for p in rel_paths])


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
