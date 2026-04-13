import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from main3 import Generator, Discriminator, AnimeImages
from tqdm import tqdm
import argparse
import re
# -----------------------
# Config
# -----------------------
ROOT = "../archive/images"
EPOCHS = 100
BATCH_SIZE = 128
NZ = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume training")
args = parser.parse_args()

# -----------------------
# Simple augmentation (flip)
# -----------------------
def augment(x):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, [-1])
    return x

# -----------------------
# Initialize models
# -----------------------
G = Generator(nz=NZ).to(DEVICE)
D = Discriminator().to(DEVICE)
#optimizers
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.0,0.9))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.0,0.9))


ema_G = Generator(nz=NZ).to(DEVICE)

start_epoch = 0

# -------- RESUME LOGIC --------
ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("EMA_G_epoch")]

if ckpts:
    def extract_epoch(name):
        return int(re.search(r"epoch(\d+)", name).group(1))

    ckpts = sorted(ckpts, key=extract_epoch)
    last_ckpt = ckpts[-1]
    start_epoch = extract_epoch(last_ckpt)

    print(f"🔁 Resuming from epoch {start_epoch}")

    G.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"G_epoch{start_epoch}.pth"),
        map_location=DEVICE
    ))
    D.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, f"D_epoch{start_epoch}.pth"),
        map_location=DEVICE
    ))
    ema_G.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, last_ckpt),
        map_location=DEVICE
    ))
else:
    ema_G.load_state_dict(G.state_dict())

# -----------------------
# Dataset
# -----------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = AnimeImages(ROOT, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------
# EMA helpers
# -----------------------
def update_ema(model, ema_model, beta=0.999):
    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1-beta)

# -----------------------
# Hinge Loss
# -----------------------
def d_loss_fn(real, fake):
    loss_real = torch.mean(torch.relu(1.0 - real))
    loss_fake = torch.mean(torch.relu(1.0 + fake))
    return loss_real + loss_fake

def g_loss_fn(fake):
    return -torch.mean(fake)

# -----------------------
# Training Loop
# -----------------------
fixed_noise = torch.randn(16, NZ,1,1, device=DEVICE)
for epoch in range(start_epoch, EPOCHS):
    epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for real, _ in epoch_bar:
        real = real.to(DEVICE)
        b = real.size(0)

        # ---- Train Discriminator ----
        z = torch.randn(b, NZ,1,1, device=DEVICE)
        fake = G(z).detach()
        real_aug = augment(real)
        fake_aug = augment(fake)
        d_real = D(real_aug)
        d_fake = D(fake_aug)
        d_loss = d_loss_fn(d_real, d_fake)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ---- Train Generator ----
        z = torch.randn(b, NZ,1,1, device=DEVICE)
        fake = G(z)
        fake_aug = augment(fake)
        g_loss = g_loss_fn(D(fake_aug))
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # EMA update
        update_ema(G, ema_G)

        # Update progress bar status
        epoch_bar.set_postfix({
            "D_loss": f"{d_loss:.4f}",
            "G_loss": f"{g_loss:.4f}"
        })

    # End of epoch logging
    print(f"Epoch [{epoch+1}/{EPOCHS}]  D Loss: {d_loss:.4f}  G Loss: {g_loss:.4f}")

    # Save sample images
    with torch.no_grad():
        samples = ema_G(fixed_noise).cpu()
        samples = (samples * 0.5 + 0.5).clamp(0,1)
        utils.save_image(samples, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.png"), nrow=4)
   
    # Save checkpoints every 5 epochs AND final epoch
    if (epoch+1) % 5 == 0 or (epoch+1) == EPOCHS:
        torch.save(G.state_dict(), os.path.join(CHECKPOINT_DIR, f"G_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(CHECKPOINT_DIR, f"D_epoch{epoch+1}.pth"))
        torch.save(ema_G.state_dict(), os.path.join(CHECKPOINT_DIR, f"EMA_G_epoch{epoch+1}.pth"))