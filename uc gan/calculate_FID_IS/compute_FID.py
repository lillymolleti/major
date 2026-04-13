import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Dataset without labels
# ---------------------------------------------------------
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
        img = Image.open(os.path.join(self.folder, self.files[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------
# Compute FID
# ---------------------------------------------------------
def compute_fid(real_dir, fake_dir):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    real_set = ImageFolderNoClass(real_dir, transform)
    fake_set = ImageFolderNoClass(fake_dir, transform)

    real_loader = DataLoader(real_set, batch_size=32, shuffle=False)
    fake_loader = DataLoader(fake_set, batch_size=32, shuffle=False)

    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)

    print("[INFO] Computing FID...")

    with torch.no_grad():
        for imgs in real_loader:
            fid.update(imgs.to(DEVICE), real=True)
        for imgs in fake_loader:
            fid.update(imgs.to(DEVICE), real=False)

    return fid.compute().item()


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------
def save_result(fid_score):
    os.makedirs("results", exist_ok=True)
    with open("results/fid_result.txt", "a") as f:
        f.write(f"FID: {fid_score:.4f}\n")
    print("[OK] Saved → results/fid_result.txt")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    REAL_DIR = "../archive/images"          # change this
    FAKE_DIR = "generated_images"           # change this

    score = compute_fid(REAL_DIR, FAKE_DIR)
    print("FID =", score)

    save_result(score)
