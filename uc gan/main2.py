import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from torch.nn.utils import spectral_norm


# ===============================
#            USE Block+
# ===============================
class USE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 2, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        w = self.gap(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        x = x * w
        return self.upsample(x)


# ===============================
#          CMHSA Block
# ===============================
class CMHSA(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key   = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).reshape(B, self.num_heads, self.head_dim, N)
        k = self.key(x).reshape(B, self.num_heads, self.head_dim, N)
        v = self.value(x).reshape(B, self.num_heads, self.head_dim, N)

        attn = torch.einsum("bhdk,bhdm->bhkm", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhkm,bhdm->bhdk", attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.out_proj(out)


# ===============================
#           Generator
# ===============================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU()
        )

        self.use = USE(ngf * 8)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU()
        )

        self.cmhsa = CMHSA(ngf * 4)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.init(z)
        x = self.use(x)
        x = self.deconv1(x)
        x = self.cmhsa(x)
        x = self.deconv2(x)
        x = self.final(x)
        return x


# ===============================
#   Optimized Discriminator
# ===============================
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()

        # SpectralNorm + NO BatchNorm + LeakyReLU(0.3) + PatchGAN
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.3),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.3),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.3),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.3),

            # PatchGAN output (no Sigmoid!)
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1))
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)  # patch output


# ===============================
#       Anime Dataset Loader
# ===============================
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


# ===============================
#             Training
# ===============================
def train():
    root = "./archive/images"
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

    # NO SIGMOID → Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(16, nz, 1, 1).to(device)

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)

            real_labels = torch.ones(b, device=device)
            fake_labels = torch.zeros(b, device=device)


            # ---- Train Discriminator ----
            z = torch.randn(b, nz, 1, 1).to(device)
            fake = G(z).detach()

            d_real = D(real).mean(dim=1)
            d_fake = D(fake).mean(dim=1)

            d_loss_real = criterion(d_real, real_labels)
            d_loss_fake = criterion(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ---- Train Generator ----
            fake = G(z)
            d_fake = D(fake).mean(dim=1)
            g_loss = criterion(d_fake, real_labels)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss:.3f}  G Loss: {g_loss:.3f}")

        # Save sample grid each epoch
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            samples = (samples * 0.5 + 0.5)
            utils.save_image(samples, f"epoch2_{epoch+1}.png", nrow=4)

    # Save models
    torch.save(G.state_dict(), "generator2.pth")
    torch.save(D.state_dict(), "discriminator2.pth")

    with open("models2.pkl", "wb") as f:
        pickle.dump({
            "generator": G.state_dict(),
            "discriminator": D.state_dict()
        }, f)

    print("Saved generator2.pth, discriminator2.pth, models2.pkl")

    return G


# ===============================
#       Fake Image Generator
# ===============================
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


# ===============================
#              MAIN
# ===============================
if __name__ == "__main__":
    G = train()
