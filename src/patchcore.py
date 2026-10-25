"""Model 2: PatchCore-style detector with pretrained features (no training).

The four ideas in plain words:
  - Feature map: a network pretrained on ImageNet turns the image into a grid
    (here 32 x 32) of vectors; each vector describes a small patch of the image
    (edges, texture, shape) with 1536 numbers.
  - Memory bank: we store the patch vectors of all 280 good training screws.
    It is the model's "memory" of what good patches look like.
  - Nearest-neighbour score: for each patch of a new image we find the closest
    vector in the bank. Far from everything = never seen on a good screw = anomalous.
    Image score = the worst (largest) patch distance.
  - Coreset subsampling: the full bank (286,720 vectors) is too big, so we keep
    10% of them, chosen greedily: always add the vector farthest from those already
    kept, so the small bank still covers every kind of good patch.

Choices fixed in advance (never tuned on test images):
  - backbone wide_resnet50_2 (IMAGENET1K_V1), frozen, eval mode;
  - input 256 px, grey -> 3 channels + ImageNet normalisation;
  - features layer2 (512 x 32 x 32) + layer3 (1024 x 16 x 16), each 3 x 3 averaged,
    layer3 upsampled to 32 x 32, concatenated -> 1536 dims per patch;
  - bank from the 280 TRAIN images, coreset 10% (seed 42, random projection to 128
    dims used only to choose the coreset);
  - heat-map: 32 x 32 patch scores upsampled (bilinear) to 256 x 256, Gaussian sigma 4
    (same as the autoencoder); no data augmentation.

Shared interface (same as AutoencoderDetector):
  det = PatchCoreDetector(); det.fit(train_images)
  score, heatmap = det.predict(image); scores, heatmaps = det.predict_many(images)
  det.save(path); det = PatchCoreDetector.load(path)

Run: python -m src.patchcore [--rebuild] [--normalize-brightness]
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import wide_resnet50_2
from tqdm import tqdm

from src.data import IMG_SIZE, SEED, get_split, load_images, shift_brightness, to_tensor
from src.paths import CHECKPOINTS, FIGURES

SMOOTH_SIGMA = 4
EMB_DIM = 512 + 1024        # layer2 + layer3 channels


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PatchCoreDetector:
    def __init__(self, size=IMG_SIZE, normalize_brightness=False, coreset_ratio=0.10,
                 proj_dim=128, seed=SEED, device=None):
        self.config = dict(size=size, normalize_brightness=normalize_brightness,
                           coreset_ratio=coreset_ratio, proj_dim=proj_dim, seed=seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(seed)
        self.backbone = wide_resnet50_2(weights="IMAGENET1K_V1").eval().requires_grad_(False)
        self.backbone.to(self.device)
        self.bank = None            # M x 1536 memory bank
        self.val_scores = None

    # ---- features ---------------------------------------------------------
    @torch.inference_mode()
    def _embed(self, images):
        """numpy B x H x W in [0, 1] -> patch features B x 1536 x 32 x 32 (CPU)."""
        x = np.asarray(images, dtype=np.float32)
        if self.config["normalize_brightness"]:
            x = shift_brightness(x)
        b = self.backbone
        t = to_tensor(x, imagenet=True).to(self.device)
        t = b.maxpool(b.relu(b.bn1(b.conv1(t))))
        f2 = b.layer2(b.layer1(t))                            # B x 512 x 32 x 32
        f3 = b.layer3(f2)                                     # B x 1024 x 16 x 16
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)         # 3 x 3 local average
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1).cpu()               # B x 1536 x 32 x 32

    # ---- memory bank ------------------------------------------------------
    @torch.inference_mode()
    def _coreset(self, feats):
        """Indices of a greedy coreset: always add the patch farthest from all chosen ones."""
        n, n_keep = len(feats), int(len(feats) * self.config["coreset_ratio"])
        g = torch.Generator().manual_seed(self.config["seed"])
        k = self.config["proj_dim"]
        proj = (torch.randn(feats.shape[1], k, generator=g) / k ** 0.5).to(self.device)
        z = torch.cat([c.to(self.device) @ proj for c in feats.split(16384)])   # N x 128
        z2 = (z ** 2).sum(dim=1)
        chosen = torch.empty(n_keep, dtype=torch.long)
        i = int(torch.randint(n, (1,), generator=g))
        min_d = (z2 - 2 * z @ z[i] + z2[i]).clamp_min(0)      # squared distance to chosen set
        chosen[0] = i
        for j in tqdm(range(1, n_keep), desc="coreset", mininterval=5):
            i = int(torch.argmax(min_d))                      # farthest remaining patch
            chosen[j] = i
            min_d = torch.minimum(min_d, (z2 - 2 * z @ z[i] + z2[i]).clamp_min(0))
        return chosen

    def fit(self, train_images, val_images=None, batch_size=8):
        """Build the bank from good TRAIN images. Val images (optional) are only scored."""
        _set_seed(self.config["seed"])
        n = len(train_images)
        t0 = time.time()
        feats = None
        for s in range(0, n, batch_size):
            e = self._embed(train_images[s:s + batch_size])
            if feats is None:                                 # preallocate once: N*1024 x 1536
                hw = e.shape[2] * e.shape[3]
                feats = torch.empty(n * hw, EMB_DIM, dtype=torch.float32)
            feats[s * hw:(s + len(e)) * hw] = e.permute(0, 2, 3, 1).reshape(-1, EMB_DIM)
        print(f"feature extraction: {time.time() - t0:.1f} s | full bank "
              f"{tuple(feats.shape)} = {feats.numel() * 4 / 2**30:.2f} GB")
        t1 = time.time()
        self.bank = feats[self._coreset(feats)].to(self.device)
        del feats
        print(f"coreset: {time.time() - t1:.1f} s | bank {tuple(self.bank.shape)} = "
              f"{self.bank.numel() * 4 / 2**30:.3f} GB")
        if val_images is not None:
            self.val_scores, _ = self.predict_many(val_images)
        return self

    # ---- scoring ----------------------------------------------------------
    @torch.inference_mode()
    def _nn_dist(self, q, bank):
        """Distance of each query row to its nearest bank row, queries in chunks
        so the distance matrix stays <= ~256 MB."""
        b2 = (bank ** 2).sum(dim=1)
        rows = max(1, (256 * 2**20 // 4) // len(bank))
        out = []
        for c in q.to(bank.device).split(rows):
            d2 = (c ** 2).sum(dim=1, keepdim=True) - 2 * c @ bank.T + b2
            out.append(d2.min(dim=1).values.clamp_min(0).sqrt().cpu())
        return torch.cat(out)

    def predict_many(self, images, batch_size=8):
        """Scores (N,) and heat-maps (N x H x W, float32). Higher = more anomalous."""
        images = np.asarray(images, dtype=np.float32)
        size = images.shape[-1]
        scores, heatmaps = [], []
        for s in range(0, len(images), batch_size):
            e = self._embed(images[s:s + batch_size])         # B x 1536 x h x w
            bsz, _, h, w = e.shape
            q = e.permute(0, 2, 3, 1).reshape(-1, EMB_DIM)
            try:
                d = self._nn_dist(q, self.bank)
            except torch.cuda.OutOfMemoryError:               # fall back to the CPU
                torch.cuda.empty_cache()
                d = self._nn_dist(q, self.bank.cpu())
            patch = d.reshape(bsz, 1, h, w)                   # 32 x 32 patch scores
            scores.append(patch.reshape(bsz, -1).max(dim=1).values.numpy())
            up = F.interpolate(patch, size=(size, size), mode="bilinear", align_corners=False)
            heatmaps += [gaussian_filter(m, SMOOTH_SIGMA) for m in up[:, 0].numpy()]
        return np.concatenate(scores), np.stack(heatmaps).astype(np.float32)

    def predict(self, image):
        """One H x W image -> (score, heat-map H x W)."""
        scores, heatmaps = self.predict_many(np.asarray(image)[None])
        return float(scores[0]), heatmaps[0]

    # ---- save / load ------------------------------------------------------
    def save(self, path=CHECKPOINTS / "patchcore_bank.pt"):
        torch.save({"bank": self.bank.cpu(), "config": self.config}, path)

    @classmethod
    def load(cls, path=CHECKPOINTS / "patchcore_bank.pt", device=None):
        ckpt = torch.load(path, map_location="cpu")
        det = cls(**ckpt["config"], device=device)
        det.bank = ckpt["bank"].to(det.device)
        return det


def _save_figure(det, val_files, val_images, path, n=5):
    """5 VALIDATION images: image | heat-map | overlay, one shared colour scale."""
    import matplotlib.pyplot as plt
    scores, maps = det.predict_many(val_images[:n])
    vmax = float(maps.max())
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    for row, f, img, s, m in zip(axes, val_files, val_images, scores, maps):
        row[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"original {f.split('/')[-1]}", fontsize=9)
        im = row[1].imshow(m, cmap="jet", vmin=0, vmax=vmax)
        row[1].set_title(f"heat-map, score {s:.3f}", fontsize=9)
        row[2].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[2].imshow(m, cmap="jet", vmin=0, vmax=vmax, alpha=0.4)
        row[2].set_title("overlay", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.5, label="distance to nearest good patch")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or load the PatchCore bank (no test images used).")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if a saved bank exists")
    parser.add_argument("--normalize-brightness", action="store_true")
    args = parser.parse_args()

    tag = "pc_bn" if args.normalize_brightness else "pc"      # bn = brightness-normalised
    ckpt = CHECKPOINTS / ("patchcore_bank_bn.pt" if args.normalize_brightness else "patchcore_bank.pt")
    split = get_split()
    val_images = load_images(split["val"])

    if ckpt.exists() and not args.rebuild:
        det = PatchCoreDetector.load(ckpt)
        print("loaded", ckpt, "| bank", tuple(det.bank.shape))
    else:
        det = PatchCoreDetector(normalize_brightness=args.normalize_brightness)
        b = det.backbone
        with torch.inference_mode():                          # print the shapes once
            t = to_tensor(val_images[:1], imagenet=True).to(det.device)
            f2 = b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(t))))))
            print("device:", det.device, "| layer2:", tuple(f2.shape), "| layer3:",
                  tuple(b.layer3(f2).shape), "| embedding:", tuple(det._embed(val_images[:1]).shape))
        det.fit(load_images(split["train"]))
        det.save(ckpt)
        print(f"saved {ckpt} ({ckpt.stat().st_size / 2**20:.0f} MB)")

    t0 = time.time()
    val_scores, _ = det.predict_many(val_images)
    print(f"scoring 40 validation images: {time.time() - t0:.1f} s")
    print(f"40 validation scores: min {val_scores.min():.3f}  median {np.median(val_scores):.3f}"
          f"  max {val_scores.max():.3f}")
    _save_figure(det, split["val"], val_images, FIGURES / f"{tag}_val_heatmaps.png")
    if torch.cuda.is_available():
        print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        torch.cuda.empty_cache()
    print("figure:", FIGURES / f"{tag}_val_heatmaps.png")
