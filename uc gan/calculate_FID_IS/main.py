# # ---- Code Cell ----



# # ---- Code Cell ----
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, utils
# import matplotlib.pyplot as plt
# from einops import rearrange
# import os


# # ---- Code Cell ----
# class USE(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.gap = nn.AdaptiveAvgPool2d(1)

#         self.fc1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#         self.upsample = nn.ConvTranspose2d(
#             in_channels, in_channels,
#             kernel_size=4, stride=2, padding=1
#         )

#     def forward(self, x):
#         w = self.gap(x)
#         w = self.relu(self.fc1(w))
#         w = self.sigmoid(self.fc2(w))
#         x = x * w
#         return self.upsample(x)


# # ---- Code Cell ----
# class CMHSA(nn.Module):
#     def __init__(self, in_channels, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = in_channels // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.query = nn.Conv2d(in_channels, in_channels, 1)
#         self.key   = nn.Conv2d(in_channels, in_channels, 1)
#         self.value = nn.Conv2d(in_channels, in_channels, 1)

#         self.dropout = nn.Dropout(dropout)
#         self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W

#         q = self.query(x).reshape(B, self.num_heads, self.head_dim, N)
#         k = self.key(x).reshape(B, self.num_heads, self.head_dim, N)
#         v = self.value(x).reshape(B, self.num_heads, self.head_dim, N)

#         attn = torch.einsum("bhdk,bhdm->bhkm", q, k) * self.scale
#         attn = torch.softmax(attn, dim=-1)
#         attn = self.dropout(attn)

#         out = torch.einsum("bhkm,bhdm->bhdk", attn, v)
#         out = out.reshape(B, C, H, W)

#         return x + self.out_proj(out)


# # ---- Code Cell ----
# class Generator(nn.Module):
#     def __init__(self, nz=100, ngf=64, nc=3):
#         super().__init__()

#         self.init = nn.Sequential(
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU()
#         )

#         self.use = USE(ngf * 8)

#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU()
#         )

#         self.cmhsa = CMHSA(ngf * 4)

#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU()
#         )

#         self.final = nn.Sequential(
#             nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         x = self.init(z)
#         x = self.use(x)
#         x = self.deconv1(x)
#         x = self.cmhsa(x)
#         x = self.deconv2(x)
#         x = self.final(x)
#         return x


# # ---- Code Cell ----
# class Discriminator(nn.Module):
#     def __init__(self, ndf=64, nc=3):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(ndf * 8, 1, 4, 1, 0),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.main(x).view(-1)


# # ---- Code Cell ----
# #from google.colab import drive
# #drive.mount('/content/drive')


# # ---- Code Cell ----
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import os

# class AnimeImages(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform

#         valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#         self.files = [
#             f for f in os.listdir(root)
#             if f.lower().endswith(valid_exts)
#         ]

#         print("Found", len(self.files), "images")

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.root, self.files[index])
#         img = Image.open(img_path).convert("RGB")

#         if self.transform:
#             img = self.transform(img)

#         return img, 0   # GAN doesn't use label


# # ---- Code Cell ----
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# dataset = AnimeImages("./archive (1)/images", transform=transform)
# loader = DataLoader(dataset, batch_size=128, shuffle=True)



# # ---- Code Cell ----
# device = "cuda" if torch.cuda.is_available() else "cpu"

# G = Generator().to(device)
# D = Discriminator().to(device)

# opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# criterion = nn.BCELoss()


# # ---- Code Cell ----
# epochs = 100
# nz = 100
# fixed_noise = torch.randn(16, nz, 1, 1).to(device)

# for epoch in range(epochs):
#     for i, (real, _) in enumerate(loader):
#         real = real.to(device)
#         b = real.size(0)

#         real_labels = torch.ones(b).to(device)
#         fake_labels = torch.zeros(b).to(device)

#         # ---- Train Discriminator ----
#         z = torch.randn(b, nz, 1, 1).to(device)
#         fake = G(z).detach()

#         loss_real = criterion(D(real), real_labels)
#         loss_fake = criterion(D(fake), fake_labels)
#         d_loss = loss_real + loss_fake

#         opt_D.zero_grad()
#         d_loss.backward()
#         opt_D.step()

#         # ---- Train Generator ----
#         fake = G(z)
#         g_loss = criterion(D(fake), real_labels)

#         opt_G.zero_grad()
#         g_loss.backward()
#         opt_G.step()

#     print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss:.3f}  G Loss: {g_loss:.3f}")

#     # Save samples every epoch
#     with torch.no_grad():
#         samples = G(fixed_noise).cpu()
#         samples = (samples * 0.5 + 0.5)
#         utils.save_image(samples, f"epoch_{epoch+1}.png", nrow=4)


# # ---- Code Cell ----
# img = plt.imread(f"epoch_{epoch+1}.png")
# plt.figure(figsize=(6,6))
# plt.imshow(img)
# plt.axis("off")

# import os
# import torch
# from torchvision.utils import save_image

# def generate_and_save(netG, out_dir='../output_images', n_images=10000, batch_size=50,
#                       nz=100, device='cuda'):
#     os.makedirs(out_dir, exist_ok=True)
#     netG.eval()
#     imgs_saved = 0
#     while imgs_saved < n_images:
#         current_batch = min(batch_size, n_images - imgs_saved)
#         with torch.no_grad():
#             z = torch.randn(current_batch, nz, 1, 1, device=device)
#             gen = netG(z).cpu()  # (B, C, H, W)
#         for i in range(gen.size(0)):
#             idx = imgs_saved + i
#             save_image(gen[i], os.path.join(out_dir, f'image_{idx:06d}.png'))
#         imgs_saved += current_batch
#     netG.train()
#     print(f"Saved {n_images} images to {out_dir}")

# Example usage (adjust device, batch_size as needed)
# generate_and_save(netG, out_dir='../output_images', n_images=10000, batch_size=50,
#                   nz=100, device=device)



import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image


# ===============================
#            USE Block
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
#          Discriminator
# ===============================
class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


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
    root = "./archive/images"  # 👈 CHANGE THIS TO YOUR FOLDER
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

            real_labels = torch.ones(b).to(device)
            fake_labels = torch.zeros(b).to(device)

            # ---- Train Discriminator ----
            z = torch.randn(b, nz, 1, 1).to(device)
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

        # Save sample grid each epoch
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
            samples = (samples * 0.5 + 0.5)
            utils.save_image(samples, f"epoch_{epoch+1}.png", nrow=4)

    # ===========================
    #     SAVE MODELS
    # ===========================
    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "discriminator.pth")

    with open("models.pkl", "wb") as f:
        pickle.dump({
            "generator": G.state_dict(),
            "discriminator": D.state_dict()
        }, f)

    print("Saved generator.pth, discriminator.pth, models.pkl")

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

    # Example: generate 10k images
    # generate_and_save(G, out_dir="generated_images", n_images=10000)




