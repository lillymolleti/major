import os
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from PIL import Image

# ===============================
# USE Block (unchanged)
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
# CMHSA Attention Block (unchanged)
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
# Generator (Improved + Fixed Residual)
# ===============================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, cond_dim=0):
        super().__init__()
        self.nz = nz
        self.cond_dim = cond_dim
        self.nz_total = nz + (cond_dim if cond_dim else 0)

        # Initial dense block
        self.init = nn.Sequential(
            nn.ConvTranspose2d(self.nz_total, ngf * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.use = USE(ngf * 8)

        # First upsampling
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # CMHSA attention + dropout
        self.cmhsa = CMHSA(ngf * 4)
        self.attn_drop = nn.Dropout2d(0.1)

        # Second upsampling
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # Fixed residual skip connection: matches spatial size of deconv1
        self.res_skip = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)

        # Weight init
        self.apply(self._weights_init)

    def forward(self, z, cond=None):
        # z: (B, nz, 1, 1)
        if self.cond_dim and cond is not None:
            # cond: (B, cond_dim) -> expand to (B, cond_dim, 1, 1)
            if cond.dim() == 2:
                cond = cond.view(cond.size(0), cond.size(1), 1, 1)
            z = torch.cat([z, cond], dim=1)
        elif self.cond_dim and cond is None:
            # pad zeros
            zeros = torch.zeros(z.size(0), self.cond_dim, 1, 1, device=z.device, dtype=z.dtype)
            z = torch.cat([z, zeros], dim=1)

        x = self.init(z)
        x = self.use(x)
        x1 = self.deconv1(x)
        x1 = self.attn_drop(self.cmhsa(x1)) + self.res_skip(x)  # fixed
        x2 = self.deconv2(x1)
        out = self.final(x2)
        return out

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)

# ===============================
# Discriminator (PatchGAN + MinibatchStdDev)
# ===============================
class MinibatchStdDev(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, unbiased=False).mean().view(1,1,1,1)
        std = std.expand(x.size(0),1,x.size(2),x.size(3))
        return torch.cat([x,std],1)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc + (cond_dim if cond_dim else 0), ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            MinibatchStdDev(),

            spectral_norm(nn.Conv2d(ndf*8+1, 1, 4, 1, 1))
        )

    def forward(self, x):
        # if conditioned, expect cond to be concatenated before calling or handled externally
        return self.main(x).view(x.size(0), -1)

# ===============================
# Dataset
# ===============================
class AnimeImages(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.files = sorted([f for f in os.listdir(root) if f.lower().endswith(valid_exts)])
        print("Found", len(self.files), "images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0
