import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.inception import InceptionScore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# Compute Inception Score
# -----------------------------------------
def compute_is(fake_dir):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])

    is_metric = InceptionScore().to(DEVICE)

    files = [f for f in os.listdir(fake_dir)
             if f.lower().endswith(("png", "jpg", "jpeg"))]

    print("[INFO] Computing Inception Score...")

    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(fake_dir, f)).convert("RGB")
            img = transform(img) * 255       # Convert to 0–255
            img = img.to(torch.uint8)        # torch_fidelity requires uint8
            img = img.unsqueeze(0).to(DEVICE)
            is_metric.update(img)

    mean, std = is_metric.compute()
    return mean.item(), std.item()


# -----------------------------------------
# Main
# -----------------------------------------
if __name__ == "__main__":
    FAKE_DIR = "generated_images"

    is_mean, is_std = compute_is(FAKE_DIR)

    print("\n===================================")
    print(f"Inception Score: {is_mean} ± {is_std}")
    print("===================================\n")
