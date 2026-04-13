import os
import numpy as np
from PIL import Image
import torch
import clip
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime


# ---------------------------
#      Load CLIP model
# ---------------------------
def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


# ---------------------------
#  Load images & get CLIP embeddings
# ---------------------------
def get_clip_embeddings(folder, model, preprocess, device):
    features = []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(("png", "jpg", "jpeg"))]

    print(f"Loading {len(files)} images from: {folder}")

    for file in tqdm(files):
        img_path = os.path.join(folder, file)
        img = Image.open(img_path).convert("RGB")

        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat.cpu().numpy().flatten()
            features.append(feat)

    return np.array(features)


# ---------------------------
#  Compute FID-CLIP
# ---------------------------
def calculate_fid_clip(real_feats, fake_feats):
    mu_real = np.mean(real_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)

    cov_real = np.cov(real_feats, rowvar=False)
    cov_fake = np.cov(fake_feats, rowvar=False)

    diff = mu_real - mu_fake

    cov_sqrt, _ = np.linalg.sqrtm(cov_real.dot(cov_fake), disp=False)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    return float(diff.dot(diff) + np.trace(cov_real + cov_fake - 2 * cov_sqrt))


# ---------------------------
#        Save results
# ---------------------------
def save_fid_clip_result(fid_clip, real_folder, fake_folder):
    os.makedirs("results", exist_ok=True)
    out_path = "results/fid_clip_results.txt"

    with open(out_path, "a") as f:
        f.write("=======================================\n")
        f.write(f"Date/Time: {datetime.now()}\n")
        f.write(f"Real images folder : {real_folder}\n")
        f.write(f"Fake images folder : {fake_folder}\n")
        f.write(f"FID-CLIP Score     : {fid_clip:.4f}\n")
        f.write("=======================================\n\n")

    print(f"[OK] Saved → {out_path}")


# ---------------------------
#              MAIN
# ---------------------------
if __name__ == "__main__":
    real_folder = "../archive/images"      # change if needed
    fake_folder = "./generated_images"     # change if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading CLIP model...")
    model, preprocess = load_clip(device)

    print("\nExtracting real image features...")
    real_feats = get_clip_embeddings(real_folder, model, preprocess, device)

    print("\nExtracting generated image features...")
    fake_feats = get_clip_embeddings(fake_folder, model, preprocess, device)

    print("\nCalculating FID-CLIP...")
    fid_clip = calculate_fid_clip(real_feats, fake_feats)

    print("\n=======================================")
    print(f"   FID-CLIP Score: {fid_clip:.4f}")
    print("   (Lower = Better)")
    print("=======================================")

    save_fid_clip_result(fid_clip, real_folder, fake_folder)

