from pathlib import Path
import torch, torchvision, sklearn, scipy, matplotlib
from PIL import Image

print("Python OK | torch", torch.__version__, "| torchvision", torchvision.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU computation OK:", (x @ x).shape)

img = Image.open(Path("screw/train/good/000.png"))
print("Dataset image:", img.mode, img.size)          # expect: L (1024, 1024)

m = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")   # downloads ~130 MB once
print("Pretrained backbone loaded OK")