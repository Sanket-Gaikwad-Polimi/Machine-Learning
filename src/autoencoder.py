"""Model 1: convolutional autoencoder.

Idea: the network squeezes a 256 x 256 image into a small code and rebuilds it.
Trained only on good screws, it rebuilds good screws well; a defect is something
it never saw, so it is rebuilt badly there.

Choices fixed in advance (never tuned on test images):
  - input: 1 grey channel in [0, 1], NO ImageNet normalisation;
  - bottleneck: 32 channels x 8 x 8 = 2048 numbers for 65536 pixels, so the
    network cannot simply copy its input;
  - loss: mean squared error; Adam, lr 1e-3, batch 8 (4 GB GPU);
  - augmentation (training only): random rotation + flips, corners filled with
    the image's median grey (background), because screws lie at any angle;
  - heat-map: per-pixel squared error, Gaussian-smoothed with sigma = 4 px;
  - image score: maximum of the smoothed heat-map;
  - the 40 validation images (good) are used only to monitor the loss and keep
    the weights of the epoch with the lowest validation loss.

Shared interface (PatchCore has the same one):
  det = AutoencoderDetector(); det.fit(train_images, val_images)
  score, heatmap = det.predict(image)            one H x W image
  scores, heatmaps = det.predict_many(images)    N x H x W images
  det.save(path); det = AutoencoderDetector.load(path)

v2: category="screw" (default, identical to v1) or another category from
src/categories.py. in_ch = 1 (greyscale) or 3 (colour) and the augmentation
("rot_flip" or "none") come from the category unless given; no layer is renamed,
so the v1 checkpoints load unchanged.

v2 phase 3b options (all off by default, so v1 is unchanged):
  loss="mse_ssim"   training loss 0.5 MSE + 0.5 (1 - mean SSIM), heat-map 1 - local SSIM
                    (11 x 11 Gaussian window, sigma 1.5, reflect padding), Gaussian sigma 4;
  predict_many(images, masks=...)   background mask: heat-map outside the mask = its minimum,
                    score = maximum inside the mask (src/masks.py).

Run: python -m src.autoencoder [--retrain] [--normalize-brightness] [--category screw]
"""
import argparse
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate
from scipy.ndimage import gaussian_filter

from src.categories import get_category, require_background_mask
from src.data import IMG_SIZE, SEED, get_split, load_images, shift_brightness, to_display
from src.paths import CHECKPOINTS

LATENT_CH = 32
SMOOTH_SIGMA = 4
SSIM_WIN, SSIM_SIGMA, SSIM_C1, SSIM_C2 = 11, 1.5, 0.01 ** 2, 0.03 ** 2   # v2 phase 3b, data range 1


def ssim_map(x, y):
    """Local SSIM of two tensors N x C x H x W in [0, 1] -> N x C x H x W (same size).

    11 x 11 Gaussian window (sigma 1.5) per channel, reflect padding, C1 = 0.01^2,
    C2 = 0.03^2. 1 = locally identical; lower = locally different structure.
    """
    r = SSIM_WIN // 2
    g = torch.exp(-(torch.arange(SSIM_WIN, dtype=x.dtype, device=x.device) - r) ** 2 / (2 * SSIM_SIGMA ** 2))
    g = g / g.sum()
    c = x.shape[1]
    w = (g[:, None] * g[None, :]).expand(c, 1, SSIM_WIN, SSIM_WIN).contiguous()

    def blur(z):
        return nn.functional.conv2d(nn.functional.pad(z, (r, r, r, r), mode="reflect"), w, groups=c)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    return ((2 * mx * my + SSIM_C1) * (2 * cxy + SSIM_C2)) / ((mx ** 2 + my ** 2 + SSIM_C1) * (vx + vy + SSIM_C2))


def _loss(rec, x, kind):
    """v1 "mse", or v2 "mse_ssim" = 0.5 MSE + 0.5 (1 - mean SSIM)."""
    mse = nn.functional.mse_loss(rec, x)
    if kind == "mse":
        return mse
    return 0.5 * mse + 0.5 * (1 - ssim_map(rec, x).mean())


def _down(c_in, c_out):
    """Conv with stride 2: halves height and width."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


def _up(c_in, c_out):
    """Transposed conv with stride 2: doubles height and width."""
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


class ConvAutoencoder(nn.Module):
    """in_ch x 256 x 256 -> (encoder) -> 32 x 8 x 8 -> (decoder) -> in_ch x 256 x 256 (v1: in_ch 1)."""

    def __init__(self, latent_ch=LATENT_CH, in_ch=1):
        super().__init__()
        self.encoder = nn.Sequential(                       # 256 -> 128 -> 64 -> 32 -> 16 -> 8
            _down(in_ch, 32), _down(32, 64), _down(64, 128), _down(128, 256), _down(256, 256),
            nn.Conv2d(256, latent_ch, 1))                   # bottleneck
        self.decoder = nn.Sequential(                       # 8 -> 16 -> 32 -> 64 -> 128 -> 256
            nn.Conv2d(latent_ch, 256, 1), nn.LeakyReLU(0.2),
            _up(256, 256), _up(256, 128), _up(128, 64), _up(64, 32),
            nn.ConvTranspose2d(32, in_ch, 4, stride=2, padding=1),
            nn.Sigmoid())                                   # output in [0, 1]

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _augment(batch, mode="rot_flip"):
    """mode "rot_flip" (screw, v1): random rotation (0-360 deg) and flips; corners get the
    image's median grey (per channel for colour). mode "none" (e.g. transistor): unchanged."""
    if mode == "none":
        return batch
    out = []
    for img in batch:                                       # img: C x H x W
        if random.random() < 0.5:
            img = img.flip(-1)
        if random.random() < 0.5:
            img = img.flip(-2)
        fill = [img.median().item()] if len(img) == 1 else [c.median().item() for c in img]
        out.append(rotate(img, random.uniform(0, 360), fill=fill))
    return torch.stack(out)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AutoencoderDetector:
    def __init__(self, size=IMG_SIZE, normalize_brightness=False, epochs=120, batch_size=8,
                 lr=1e-3, seed=SEED, latent_ch=LATENT_CH, category="screw", in_ch=None, augment=None,
                 loss="mse", device=None):
        cat = get_category(category)
        self.config = dict(size=size, normalize_brightness=normalize_brightness, epochs=epochs,
                           batch_size=batch_size, lr=lr, seed=seed, latent_ch=latent_ch,
                           category=category, in_ch=in_ch or cat["channels"],
                           augment=augment or cat["ae_augment"], loss=loss)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(seed)
        self.model = ConvAutoencoder(latent_ch, self.config["in_ch"]).to(self.device)
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}
        self.best_epoch = None

    def _prepare(self, images):
        """numpy (N x) H x W (colour: (N x) 3 x H x W) in [0, 1] -> tensor N x C x H x W
        (brightness shift if enabled)."""
        x = np.asarray(images, dtype=np.float32)
        grey = self.config["in_ch"] == 1
        if x.ndim == (2 if grey else 3):
            x = x[None]
        if self.config["normalize_brightness"]:
            x = shift_brightness(x, self.config["category"])
        x = torch.from_numpy(x)
        return x[:, None] if grey else x

    def _reconstruct(self, x):
        """Tensor N x C x H x W -> reconstructions on the CPU, in small batches."""
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self.model(b.to(self.device)).cpu()
                              for b in x.split(self.config["batch_size"])])

    def fit(self, train_images, val_images=None):
        c = self.config
        _set_seed(c["seed"])
        x_train = self._prepare(train_images)
        x_val = self._prepare(val_images) if val_images is not None else None
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"])
        best_loss, best_state = float("inf"), None
        for epoch in range(1, c["epochs"] + 1):
            t0 = time.time()
            self.model.train()
            total = 0.0
            for idx in torch.randperm(len(x_train)).split(c["batch_size"]):
                batch = _augment(x_train[idx].to(self.device), c["augment"])
                loss = _loss(self.model(batch), batch, c["loss"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / len(x_train)
            val_loss = (_loss(self._reconstruct(x_val), x_val, c["loss"]).item()
                        if x_val is not None else float("nan"))
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            monitored = val_loss if x_val is not None else train_loss
            if monitored < best_loss:                          # keep the best weights
                best_loss, self.best_epoch = monitored, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            print(f"epoch {epoch:3d}/{c['epochs']}  train {train_loss:.5f}  val {val_loss:.5f}"
                  f"  ({time.time() - t0:.1f} s){'  *best' if self.best_epoch == epoch else ''}")
        self.model.load_state_dict(best_state)
        return self

    def predict_many(self, images, masks=None):
        """Scores (N,) and smoothed heat-maps (N x H x W, float32).
        v1 ("mse"): squared error; v2 "mse_ssim": 1 - local SSIM. Colour: mean over the 3 channels.
        masks (N x H x W bool, optional, v2): heat-map outside = its minimum, score = max inside."""
        x = self._prepare(images)
        rec = self._reconstruct(x)
        err = (x - rec) ** 2 if self.config.get("loss", "mse") == "mse" else 1 - ssim_map(rec, x)
        err = (err[:, 0] if self.config["in_ch"] == 1 else err.mean(dim=1)).numpy()
        heatmaps = np.stack([gaussian_filter(e, SMOOTH_SIGMA) for e in err]).astype(np.float32)
        if masks is not None:
            from src.masks import apply_mask
            require_background_mask(self.config["category"])
            heatmaps = apply_mask(heatmaps, masks)
            scores = np.array([m[k].max() for m, k in zip(heatmaps, np.asarray(masks, bool))], dtype=np.float32)
            return scores, heatmaps
        return heatmaps.reshape(len(heatmaps), -1).max(axis=1), heatmaps

    def predict(self, image):
        """One H x W (colour: 3 x H x W) image -> (score, heat-map H x W). Higher = more anomalous."""
        scores, heatmaps = self.predict_many(np.asarray(image)[None])
        return float(scores[0]), heatmaps[0]

    def reconstruct(self, images):
        """Rebuilt images N x H x W (colour: N x 3 x H x W), for plots."""
        rec = self._reconstruct(self._prepare(images))
        return (rec[:, 0] if self.config["in_ch"] == 1 else rec).numpy()

    def save(self, path=CHECKPOINTS / "autoencoder.pt"):
        torch.save({"config": self.config, "state_dict": self.model.state_dict(),
                    "history": self.history, "best_epoch": self.best_epoch}, path)

    @classmethod
    def load(cls, path=CHECKPOINTS / "autoencoder.pt", device=None):
        ckpt = torch.load(path, map_location="cpu")
        det = cls(**ckpt["config"], device=device)
        det.model.load_state_dict(ckpt["state_dict"])
        det.history, det.best_epoch = ckpt["history"], ckpt["best_epoch"]
        return det


def _save_figures(det, val_files, val_images, tag, figures, n_train):
    import matplotlib.pyplot as plt
    h = det.history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label=f"train ({n_train} good)")
    ax.plot(h["epoch"], h["val_loss"], label=f"validation ({len(val_files)} good)")
    ax.axvline(det.best_epoch, color="gray", ls="--", lw=1, label=f"best epoch {det.best_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Autoencoder training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / f"{tag}_loss_curve.png", dpi=100)
    plt.close(fig)

    n = 5                                                    # VALIDATION images only
    recon = det.reconstruct(val_images[:n])
    scores, maps = det.predict_many(val_images[:n])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, f, img, rec, s, m in zip(axes, val_files, val_images, recon, scores, maps):
        row[0].imshow(to_display(img), cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"original {f.split('/')[-1]}", fontsize=9)
        row[1].imshow(to_display(rec), cmap="gray", vmin=0, vmax=1)
        row[1].set_title("reconstruction", fontsize=9)
        row[2].imshow(m, cmap="jet")
        row[2].set_title(f"smoothed error, score {s:.4f}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures / f"{tag}_val_reconstructions.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load the autoencoder (no test images used).")
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    parser.add_argument("--normalize-brightness", action="store_true")
    parser.add_argument("--category", default="screw", help="MVTec category (default screw = v1)")
    args = parser.parse_args()

    cat = get_category(args.category)
    for d in (cat["results"], cat["figures"], cat["checkpoints"]):
        d.mkdir(parents=True, exist_ok=True)
    tag = "ae_bn" if args.normalize_brightness else "ae"     # bn = brightness-normalised
    ckpt = cat["checkpoints"] / ("autoencoder_bn.pt" if args.normalize_brightness else "autoencoder.pt")
    split = get_split(args.category)
    train_images = load_images(split["train"], category=args.category)
    val_images = load_images(split["val"], category=args.category)

    if ckpt.exists() and not args.retrain:
        det = AutoencoderDetector.load(ckpt)
        print("loaded", ckpt)
    else:
        det = AutoencoderDetector(normalize_brightness=args.normalize_brightness, category=args.category)
        print(det.model)
        print("device:", det.device, "| parameters:",
              f"{sum(p.numel() for p in det.model.parameters()):,}")
        t0 = time.time()
        try:
            det.fit(train_images, val_images)
        except torch.cuda.OutOfMemoryError:                  # fall back to a smaller batch
            torch.cuda.empty_cache()
            print("out of GPU memory, retrying with batch size 4")
            det = AutoencoderDetector(normalize_brightness=args.normalize_brightness, batch_size=4,
                                      category=args.category)
            det.fit(train_images, val_images)
        print(f"training time: {(time.time() - t0) / 60:.1f} min")
        det.save(ckpt)
        print("saved", ckpt)

    h = det.history
    assert det.config["category"] == args.category, f"{ckpt} belongs to {det.config['category']}"
    with open(cat["results"] / f"{tag}_history.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for e, tr, va in zip(h["epoch"], h["train_loss"], h["val_loss"]):
            f.write(f"{e},{tr:.6f},{va:.6f}\n")
    _save_figures(det, split["val"], val_images, tag, cat["figures"], len(split["train"]))

    b = det.best_epoch - 1
    print(f"epoch 1: train {h['train_loss'][0]:.5f} val {h['val_loss'][0]:.5f} | "
          f"last: train {h['train_loss'][-1]:.5f} val {h['val_loss'][-1]:.5f} | "
          f"best epoch {det.best_epoch}: val {h['val_loss'][b]:.5f}")
    val_scores, _ = det.predict_many(val_images)
    print(f"40 validation scores: min {val_scores.min():.5f}  median {np.median(val_scores):.5f}"
          f"  max {val_scores.max():.5f}")
    if torch.cuda.is_available():
        print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        torch.cuda.empty_cache()
    print("figures:", cat["figures"] / f"{tag}_loss_curve.png", "|",
          cat["figures"] / f"{tag}_val_reconstructions.png")
