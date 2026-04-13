import os
import torch
from torchvision import utils, transforms
from PIL import Image
from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# IMPORTANT: Fix import for Generator
# ------------------------------
# If your Generator is inside main.py, keep this.
# Else update path accordingly.
from main import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NZ = 100
RESULTS_FILE = "results/fid_is_results.txt"


# ------------------------------
# Dataset WITHOUT classes
# ------------------------------
class ImageFolderNoClass(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img


# ------------------------------
# Save evaluation results
# ------------------------------
def save_results(fid_score, is_score, is_std, n_images):
    os.makedirs("results", exist_ok=True)

    with open(RESULTS_FILE, "a") as f:
        f.write("=====================================\n")
        f.write(f"Date/Time: {datetime.now()}\n")
        f.write(f"Generated Images: {n_images}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"Inception Score: {is_score:.4f} ± {is_std:.4f}\n")
        f.write("=====================================\n\n")

    print(f"[OK] Results saved → {RESULTS_FILE}")


# ------------------------------
# Load Generator
# ------------------------------
def load_generator(ckpt="generator.pth"):
    G = Generator(nz=NZ).to(DEVICE)
    G.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    G.eval()
    print("[OK] Generator loaded.")
    return G


# ------------------------------
# Generate Fake Images
# ------------------------------
def generate_fake_images(G, out_dir="generated_images", n_images=5000, batch_size=100):
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    while saved < n_images:
        cur = min(batch_size, n_images - saved)

        with torch.no_grad():
            z = torch.randn(cur, NZ, 1, 1).to(DEVICE)
            imgs = G(z).cpu()

        imgs = (imgs * 0.5 + 0.5)  # scale [-1,1] → [0,1]

        for i in range(cur):
            utils.save_image(imgs[i], f"{out_dir}/{saved + i:06d}.png")

        saved += cur

    print(f"[OK] Generated {n_images} fake images → {out_dir}")


# ------------------------------
# Compute FID
# ------------------------------
def compute_fid(real_dir, fake_dir):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    real_ds = ImageFolderNoClass(real_dir, transform)
    fake_ds = ImageFolderNoClass(fake_dir, transform)

    real_loader = DataLoader(real_ds, batch_size=32, shuffle=False)
    fake_loader = DataLoader(fake_ds, batch_size=32, shuffle=False)

    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)

    print("[INFO] Computing FID...")

    with torch.no_grad():
        for imgs in real_loader:
            fid.update(imgs.to(DEVICE), real=True)
        for imgs in fake_loader:
            fid.update(imgs.to(DEVICE), real=False)

    return fid.compute().item()


# ------------------------------
# Compute Inception Score
# ------------------------------
def compute_is(fake_dir):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])

    is_metric = InceptionScore().to(DEVICE)
    files = [f for f in os.listdir(fake_dir) if f.lower().endswith("png")]

    print("[INFO] Computing Inception Score...")

    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(fake_dir, f)).convert("RGB")
            img = transform(img).unsqueeze(0).to(DEVICE)
            is_metric.update(img)

    mean, std = is_metric.compute()
    return mean.item(), std.item()


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    REAL_DIR = "../archive/images"   # <-- update if needed
    FAKE_DIR = "generated_images"
    N_IMAGES = 5000

    # Load generator
    G = load_generator("generator.pth")

    # Generate images
    generate_fake_images(G, out_dir=FAKE_DIR, n_images=N_IMAGES)

    # FID
    fid_score = compute_fid(REAL_DIR, FAKE_DIR)
    print("FID:", fid_score)

    # IS
    is_mean, is_std = compute_is(FAKE_DIR)
    print(f"IS: {is_mean} ± {is_std}")

    # Save results
    save_results(fid_score, is_mean, is_std, N_IMAGES)
