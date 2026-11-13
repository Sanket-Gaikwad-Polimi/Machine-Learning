"""Background mask (v2, phase 3b): keep only the object and a margin around it.

  object_mask_1024(img)          dilated screw region of one 1024 px image (bool 1024 x 1024)
  background_masks(files, size)  the masks of several image files, resized to `size`
  mask_to_size(mask, size)       1024 px mask -> size x size (nearest-neighbour, like load_mask)
  apply_mask(maps, masks)        heat-map outside the mask -> that heat-map's minimum

Rules fixed in decisions.md (v2, Phase 3b): the screw rule of src/synthetic.py at 1024 px,
then every pixel within Euclidean distance 32 px of the screw (8 px at 256). Computed from
each image itself, no label. Only for categories that allow it (src/categories.py).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.categories import get_category, require_background_mask
from src.data import IMG_SIZE

DILATE = 32                 # px at 1024


def object_mask_1024(img, category="screw"):
    """img: float 1024 x 1024 in [0, 1] -> bool mask: screw region + 32 px (Euclidean)."""
    require_background_mask(category)
    from src.synthetic import screw_region
    return distance_transform_edt(~screw_region(img)) <= DILATE


def mask_to_size(mask, size=IMG_SIZE):
    """bool 1024 px -> bool size x size with nearest-neighbour, like data.load_mask."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(m) > 127


def background_masks(rel_paths, size=IMG_SIZE, category="screw"):
    """The background masks of image files (paths relative to the category's data folder)."""
    require_background_mask(category)
    data = get_category(category)["data"]
    out = []
    for f in rel_paths:
        img = np.asarray(Image.open(data / f).convert("L"), dtype=np.float32) / 255.0
        out.append(mask_to_size(object_mask_1024(img, category), size))
    return np.stack(out)


def apply_mask(maps, masks):
    """Heat-maps N x H x W: pixels outside the mask get that heat-map's minimum (over the whole
    map, before masking). Returns a new array; masks must have the heat-maps' size."""
    maps = np.asarray(maps, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    assert maps.shape == masks.shape, f"heat-maps {maps.shape} vs masks {masks.shape}"
    low = maps.reshape(len(maps), -1).min(axis=1)[:, None, None]
    return np.where(masks, maps, low).astype(np.float32)
