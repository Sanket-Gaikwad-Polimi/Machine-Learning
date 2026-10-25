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

Run: python -m src.autoencoder [--retrain] [--normalize-brightness]
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

from src.data import IMG_SIZE, SEED, get_split, load_images, shift_brightness
from src.paths import CHECKPOINTS, FIGURES, RESULTS

LATENT_CH = 32
SMOOTH_SIGMA = 4


def _down(c_in, c_out):
    """Conv with stride 2: halves height and width."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


def _up(c_in, c_out):
    """Transposed conv with stride 2: doubles height and width."""
    return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2))


class ConvAutoencoder(nn.Module):
    """1 x 256 x 256 -> (encoder) -> 32 x 8 x 8 -> (decoder) -> 1 x 256 x 256."""

    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        self.encoder = nn.Sequential(                       # 256 -> 128 -> 64 -> 32 -> 16 -> 8
            _down(1, 32), _down(32, 64), _down(64, 128), _down(128, 256), _down(256, 256),
            nn.Conv2d(256, latent_ch, 1))                   # bottleneck
        self.decoder = nn.Sequential(                       # 8 -> 16 -> 32 -> 64 -> 128 -> 256
            nn.Conv2d(latent_ch, 256, 1), nn.LeakyReLU(0.2),
            _up(256, 256), _up(256, 128), _up(128, 64), _up(64, 32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid())                                   # output in [0, 1]

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _augment(batch):
    """Random rotation (0-360 deg) and flips; corners get the image's median grey."""
    out = []
    for img in batch:                                       # img: 1 x H x W
        if random.random() < 0.5:
            img = img.flip(-1)
        if random.random() < 0.5:
            img = img.flip(-2)
        out.append(rotate(img, random.uniform(0, 360), fill=[img.median().item()]))
    return torch.stack(out)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AutoencoderDetector:
    def __init__(self, size=IMG_SIZE, normalize_brightness=False, epochs=120, batch_size=8,
                 lr=1e-3, seed=SEED, latent_ch=LATENT_CH, device=None):
        self.config = dict(size=size, normalize_brightness=normalize_brightness, epochs=epochs,
                           batch_size=batch_size, lr=lr, seed=seed, latent_ch=latent_ch)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(seed)
        self.model = ConvAutoencoder(latent_ch).to(self.device)
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}
        self.best_epoch = None

    def _prepare(self, images):
        """numpy (N x) H x W in [0, 1] -> tensor N x 1 x H x W (brightness shift if enabled)."""
        x = np.asarray(images, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if self.config["normalize_brightness"]:
            x = shift_brightness(x)
        return torch.from_numpy(x)[:, None]

    def _reconstruct(self, x):
        """Tensor N x 1 x H x W -> reconstructions on the CPU, in small batches."""
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
                batch = _augment(x_train[idx].to(self.device))
                loss = nn.functional.mse_loss(self.model(batch), batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / len(x_train)
            val_loss = (nn.functional.mse_loss(self._reconstruct(x_val), x_val).item()
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

    def predict_many(self, images):
        """Scores (N,) and smoothed squared-error heat-maps (N x H x W, float32)."""
        x = self._prepare(images)
        err = ((x - self._reconstruct(x)) ** 2)[:, 0].numpy()
        heatmaps = np.stack([gaussian_filter(e, SMOOTH_SIGMA) for e in err]).astype(np.float32)
        return heatmaps.reshape(len(heatmaps), -1).max(axis=1), heatmaps

    def predict(self, image):
        """One H x W image -> (score, heat-map H x W). Higher = more anomalous."""
        scores, heatmaps = self.predict_many(np.asarray(image)[None])
        return float(scores[0]), heatmaps[0]

    def reconstruct(self, images):
        """Rebuilt images N x H x W, for plots."""
        return self._reconstruct(self._prepare(images))[:, 0].numpy()

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


def _save_figures(det, val_files, val_images, tag):
    import matplotlib.pyplot as plt
    h = det.history
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label="train (280 good)")
    ax.plot(h["epoch"], h["val_loss"], label="validation (40 good)")
    ax.axvline(det.best_epoch, color="gray", ls="--", lw=1, label=f"best epoch {det.best_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Autoencoder training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{tag}_loss_curve.png", dpi=100)
    plt.close(fig)

    n = 5                                                    # VALIDATION images only
    recon = det.reconstruct(val_images[:n])
    scores, maps = det.predict_many(val_images[:n])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row, f, img, rec, s, m in zip(axes, val_files, val_images, recon, scores, maps):
        row[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"original {f.split('/')[-1]}", fontsize=9)
        row[1].imshow(rec, cmap="gray", vmin=0, vmax=1)
        row[1].set_title("reconstruction", fontsize=9)
        row[2].imshow(m, cmap="jet")
        row[2].set_title(f"smoothed error, score {s:.4f}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{tag}_val_reconstructions.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load the autoencoder (no test images used).")
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    parser.add_argument("--normalize-brightness", action="store_true")
    args = parser.parse_args()

    tag = "ae_bn" if args.normalize_brightness else "ae"     # bn = brightness-normalised
    ckpt = CHECKPOINTS / ("autoencoder_bn.pt" if args.normalize_brightness else "autoencoder.pt")
    split = get_split()
    train_images, val_images = load_images(split["train"]), load_images(split["val"])

    if ckpt.exists() and not args.retrain:
        det = AutoencoderDetector.load(ckpt)
        print("loaded", ckpt)
    else:
        det = AutoencoderDetector(normalize_brightness=args.normalize_brightness)
        print(det.model)
        print("device:", det.device, "| parameters:",
              f"{sum(p.numel() for p in det.model.parameters()):,}")
        t0 = time.time()
        try:
            det.fit(train_images, val_images)
        except torch.cuda.OutOfMemoryError:                  # fall back to a smaller batch
            torch.cuda.empty_cache()
            print("out of GPU memory, retrying with batch size 4")
            det = AutoencoderDetector(normalize_brightness=args.normalize_brightness, batch_size=4)
            det.fit(train_images, val_images)
        print(f"training time: {(time.time() - t0) / 60:.1f} min")
        det.save(ckpt)
        print("saved", ckpt)

    h = det.history
    with open(RESULTS / f"{tag}_history.csv", "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for e, tr, va in zip(h["epoch"], h["train_loss"], h["val_loss"]):
            f.write(f"{e},{tr:.6f},{va:.6f}\n")
    _save_figures(det, split["val"], val_images, tag)

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
    print("figures:", FIGURES / f"{tag}_loss_curve.png", "|", FIGURES / f"{tag}_val_reconstructions.png")
