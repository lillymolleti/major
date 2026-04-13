import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image


# ================================
#       PURE DCGAN GENERATOR
# ================================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


# ================================
#       PURE DCGAN DISCRIMINATOR
# ================================
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


# ================================
#           DATASET
# ================================
class AnimeImages(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.files = [f for f in os.listdir(root) if f.lower().endswith(valid_exts)]

        print("Found", len(self.files), "images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0


# ================================
#            TRAINING
# ================================
def train():
    root = "archive/images"
    epochs = 100
    batch_size = 128
    nz = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = AnimeImages(root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, nz, 1, 1).to(device)

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)

            real_labels = torch.ones(b, device=device)
            fake_labels = torch.zeros(b, device=device)

            # ---- Train Discriminator ----
            z = torch.randn(b, nz, 1, 1, device=device)
            fake = G(z).detach()

            d_loss_real = criterion(D(real), real_labels)
            d_loss_fake = criterion(D(fake), fake_labels)
            d_loss = d_loss_real + d_loss_fake

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---- Train Generator ----
            fake = G(z)
            g_loss = criterion(D(fake), real_labels)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss:.3f}  G Loss: {g_loss:.3f}")

        # Save sample grid
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            samples = (samples * 0.5 + 0.5)
            utils.save_image(samples, f"epoch_{epoch+1}.png", nrow=4)

    # Save Models
    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "discriminator.pth")

    with open("models.pkl", "wb") as f:
        pickle.dump({
            "generator": G.state_dict(),
            "discriminator": D.state_dict()
        }, f)

    print("Saved generator.pth, discriminator.pth, models.pkl")

    return G


# ================================
#     GENERATE IMAGES FUNCTION
# ================================
def generate_and_save(G, out_dir="generated", n_images=10000, batch_size=50, nz=100, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    G.eval()
    saved = 0

    while saved < n_images:
        cur = min(batch_size, n_images - saved)

        with torch.no_grad():
            z = torch.randn(cur, nz, 1, 1, device=device)
            imgs = G(z).cpu()

        for i in range(cur):
            utils.save_image(imgs[i], os.path.join(out_dir, f"{saved+i:06d}.png"))

        saved += cur

    print(f"Saved {n_images} images to {out_dir}")


# ================================
#               MAIN
# ================================
if __name__ == "__main__":
    G = train()
    # generate_and_save(G, out_dir="generated_images", n_images=10000)
