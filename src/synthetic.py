"""Synthetic defects on the 40 VALIDATION screws (v2, phase 2).

  make_synthetic(files)   -> images N x 256 x 256, masks N x 256 x 256, parameter table
  screw_region(img)       the screw (foreground) of one 1024 px image
  load_clean(files)       the same files through the same 1024 -> 256 pipeline, no defect

Every rule and range below is fixed in decisions.md (v2, Phase 2) before anything was
scored. Defects are drawn at the original 1024 px and then resized exactly like
data.load_image / data.load_mask, so the models see them like a real photo.
Nothing is stored: the same seed always gives the same 160 images.
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from src.data import DATA_DIR, IMG_SIZE

SEED = 42
TYPES = ["scratch", "spot", "cutpaste", "thread_smear"]
FG_DELTA = 0.30             # screw = darker than median grey - 0.30
EDGE = 12                   # defect centre at least 12 px inside the screw
THREAD_T = (0.55, 0.95)     # threaded part along the axis (0 = head, 1 = tip)
AREA = (0.001, 0.01)        # mask area at 256 px, share of the image (from the EDA)
MIN_CHANGE = 0.03           # largest grey change inside the 256 px mask
MAX_DRAWS = 500


# ---- 1024 <-> 256, exactly like src/data.py -------------------------------------------
def _load_1024(rel_path):
    return np.asarray(Image.open(DATA_DIR / rel_path).convert("L"), dtype=np.float32) / 255.0


def _resize(img01, size):
    """float 1024 px in [0, 1] -> 8 bit -> PIL bilinear to size -> float, like load_image."""
    u8 = np.round(np.clip(img01, 0, 1) * 255).astype(np.uint8)
    small = Image.fromarray(u8).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def _to_256(img01):
    """float 1024 px in [0, 1] -> 8 bit -> PIL bilinear to 256 -> float, like load_image."""
    return _resize(img01, IMG_SIZE)


def _mask_to_256(mask):
    """bool 1024 px -> 0/1 at 256 with nearest-neighbour, like load_mask."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def load_clean(files):
    """The validation images through this pipeline without a defect (must equal load_images)."""
    return np.stack([_to_256(_load_1024(f)) for f in files])


# ---- where the screw is ---------------------------------------------------------------
def screw_region(img):
    """Screw pixels of a 1024 px image: darker than median - 0.30, cleaned, largest region."""
    fg = img < np.median(img) - FG_DELTA
    fg = ndi.binary_closing(fg, iterations=6)
    fg = ndi.binary_opening(fg, iterations=3)
    lab, n = ndi.label(fg)
    fg = lab == 1 + int(np.argmax(ndi.sum(fg, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(fg)


def axis_position(fg):
    """Position t along the screw's main axis for every pixel: 0 at the head, 1 at the tip."""
    ys, xs = np.nonzero(fg)
    c = np.array([ys.mean(), xs.mean()])
    _, vecs = np.linalg.eigh(np.cov(np.stack([ys, xs])))
    u = vecs[:, -1]                                          # longest direction
    yy, xx = np.mgrid[:fg.shape[0], :fg.shape[1]]
    proj = (yy - c[0]) * u[0] + (xx - c[1]) * u[1]
    lo, hi = proj[fg].min(), proj[fg].max()
    t = (proj - lo) / (hi - lo)
    if (fg & (t > 0.8)).sum() > (fg & (t < 0.2)).sum():      # head = the wider end
        t = 1 - t
    return t


# ---- the 4 defect types (each returns the changed image and the 1024 px shape) --------
def _pick(rng, region):
    ys, xs = np.nonzero(region)
    k = rng.integers(len(ys))
    return float(ys[k]), float(xs[k])


def _ellipse_r(shape, cy, cx, a, b, ang):
    """Normalised ellipse radius (1 on the border) for every pixel."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    return np.sqrt((u / a) ** 2 + (v / b) ** 2)


def _scratch(rng, img, fg, inner, ctx):
    cy, cx = _pick(rng, inner)
    length, width = rng.uniform(60, 250), rng.uniform(2, 6)
    ang = rng.uniform(0, np.pi)
    delta = rng.choice([-1, 1]) * rng.uniform(0.06, 0.15)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    d = np.array([np.sin(ang), np.cos(ang)])
    y0, x0 = cy - d[0] * length / 2, cx - d[1] * length / 2
    s = np.clip((yy - y0) * d[0] + (xx - x0) * d[1], 0, length)        # closest point on the segment
    dist = np.hypot(yy - (y0 + s * d[0]), xx - (x0 + s * d[1]))
    line = dist <= width / 2
    if (line & fg).sum() < 0.9 * line.sum():
        return None
    profile = ndi.gaussian_filter(np.clip(width / 2 + 0.5 - dist, 0, 1), 1)
    out = img + delta * profile * fg
    params = dict(length=length, width=width, angle_deg=np.degrees(ang), delta=delta)
    return out, dist <= width / 2 + 4, params


def _spot(rng, img, fg, inner, ctx):
    cy, cx = _pick(rng, inner)
    a, b, ang = rng.uniform(15, 50), rng.uniform(15, 50), rng.uniform(0, np.pi)
    delta = rng.choice([-1, 1]) * rng.uniform(0.08, 0.20)
    r = _ellipse_r(img.shape, cy, cx, a, b, ang)
    weight = np.clip(1 - r, 0, 1) ** 0.5                                # 1 in the centre, 0 at the border
    out = img + delta * weight * fg
    return out, r <= 1, dict(a=a, b=b, angle_deg=np.degrees(ang), delta=delta)


def _cutpaste(rng, img, fg, inner, ctx):
    h, w = int(rng.uniform(30, 90)), int(rng.uniform(30, 90))
    ty, tx = _pick(rng, inner)
    sy, sx = _pick(rng, inner)
    if np.hypot(ty - sy, tx - sx) < 1.5 * max(h, w):
        return None
    ty0, tx0, sy0, sx0 = int(ty) - h // 2, int(tx) - w // 2, int(sy) - h // 2, int(sx) - w // 2
    H, W = img.shape
    if min(ty0, tx0, sy0, sx0) < 0 or max(ty0, sy0) + h > H or max(tx0, sx0) + w > W:
        return None
    rect = np.zeros(img.shape, bool)
    rect[ty0:ty0 + h, tx0:tx0 + w] = True
    src = img.copy()
    src[ty0:ty0 + h, tx0:tx0 + w] = img[sy0:sy0 + h, sx0:sx0 + w]
    alpha = ndi.gaussian_filter(rect.astype(np.float32), 3) * fg
    out = (1 - alpha) * img + alpha * src
    return out, rect, dict(h=h, w=w, source_y=sy, source_x=sx)


def _thread_smear(rng, img, fg, inner, ctx):
    region = inner & (ctx["t"] >= THREAD_T[0]) & (ctx["t"] <= THREAD_T[1])
    cy, cx = _pick(rng, region)
    a, b, ang = rng.uniform(20, 50), rng.uniform(20, 50), rng.uniform(0, np.pi)
    sigma = rng.uniform(3, 8)
    r = _ellipse_r(img.shape, cy, cx, a, b, ang)
    weight = np.clip(1.5 * (1 - r), 0, 1) * fg                          # flat core, soft border
    out = (1 - weight) * img + weight * ndi.gaussian_filter(img, sigma)
    return out, r <= 1, dict(a=a, b=b, angle_deg=np.degrees(ang), sigma=sigma)


MAKERS = {"scratch": _scratch, "spot": _spot, "cutpaste": _cutpaste, "thread_smear": _thread_smear}


# ---- all 160 ----------------------------------------------------------------------------
def make_one(img, fg, inner, ctx, clean256, kind, rng):
    """Draw until area, on-screw and visibility rules hold; returns image, mask (256), info."""
    for draw in range(1, MAX_DRAWS + 1):
        res = MAKERS[kind](rng, img, fg, inner, ctx)
        if res is None:
            continue
        out, shape, params = res
        mask = _mask_to_256(shape & fg)
        area = float(mask.mean())
        if not AREA[0] <= area <= AREA[1]:
            continue
        small = _to_256(out)
        change = np.abs(small - clean256)
        if change[mask == 1].max() < MIN_CHANGE:
            continue
        outside = ndi.binary_dilation(mask, iterations=2) == 0
        info = dict(type=kind, draws=draw, area_pct=100 * area,
                    max_change=float(change[mask == 1].max()),
                    mean_change_in_mask=float(change[mask == 1].mean()),
                    changed_px_outside=int((change[outside] > 1.5 / 255).sum()), **params)
        return small, mask, info, out
    raise RuntimeError(f"{kind}: no valid defect after {MAX_DRAWS} draws")


def make_synthetic(files, image_sizes=(), bg_mask_sizes=()):
    """One defect of each type on each file: images, masks (N x 256 x 256) and a table.

    v2 phase 3b, technical options (off by default; they change no random draw and no
    256 px image). If either is given, a 4th value is returned:
      {"images": {size: N x size x size}, "bg_masks": {size: N x size x size}}
    image_sizes=(512,): the same defective 1024 px images resized to 512 exactly like
      data.load_image (8 bit, PIL bilinear);
    bg_mask_sizes=(256, 512): the background mask of each DEFECTIVE 1024 px image
      (src/masks.py), computed from the image itself, as for a real image.
    """
    images, masks, rows = [], [], []
    big = {s: [] for s in image_sizes}
    bg = {s: [] for s in bg_mask_sizes}
    for i, f in enumerate(files):
        img = _load_1024(f)
        fg = screw_region(img)
        inner = ndi.binary_erosion(fg, iterations=EDGE)
        ctx = {"t": axis_position(fg)}
        clean256 = _to_256(img)
        for k, kind in enumerate(TYPES):
            rng = np.random.default_rng([SEED, i, k])
            small, mask, info, out = make_one(img, fg, inner, ctx, clean256, kind, rng)
            images.append(small)
            masks.append(mask)
            rows.append({"file": f, "val_index": i, **info})
            for s in big:
                big[s].append(_resize(out, s))
            if bg:
                from src.masks import mask_to_size, object_mask_1024
                # the defective image as a real photo would be: 8-bit rounded, then the mask rule
                obj = object_mask_1024(np.round(np.clip(out, 0, 1) * 255).astype(np.float32) / 255.0)
                for s in bg:
                    bg[s].append(mask_to_size(obj, s))
    result = (np.stack(images), np.stack(masks), pd.DataFrame(rows))
    if big or bg:
        return result + ({"images": {s: np.stack(v) for s, v in big.items()},
                          "bg_masks": {s: np.stack(v) for s, v in bg.items()}},)
    return result


if __name__ == "__main__":
    # Self-check: python -m src.synthetic (2 images only, no model)
    from src.data import get_split, load_images
    files = get_split()["val"][:2]
    clean = load_clean(files)
    print("1024 -> 256 pipeline equals load_images:", np.array_equal(clean, load_images(files)))
    x, m, info = make_synthetic(files)
    print(x.shape, m.shape, x.dtype, m.dtype)
    print(info[["file", "type", "draws", "area_pct", "max_change", "changed_px_outside"]].round(3).to_string())
