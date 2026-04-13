def main():

    import os
    import shutil
    import subprocess
    import torch
    from torchvision import utils
    from tqdm import tqdm
    from torch_fidelity import calculate_metrics
    from main3 import Generator
    import sys
    from multiprocessing import freeze_support
    from PIL import Image
    from train import BATCH_SIZE, NZ, EPOCHS, ROOT, CHECKPOINT_DIR

    # -------------------------------
    # Config
    # -------------------------------
    ROOT = "../archive/images"
    EPOCHS = 100
    BATCH_SIZE = 128
    NZ = 100
    CHECKPOINT_DIR = "./checkpoints"

    N_GEN_IMAGES = 10000       
    QUICK_FID_NUM = 1000       
    GEN_BATCH = 50
    SAMPLES_DIR = "samples"
    GENERATED_DIR = "generated_images"
    QUICK_DIR = "generated_quick"
    FID_FILE = "fid_scores.txt"

    SRC_REAL_DIR = ROOT                 # original real images
    REAL_RESIZED_DIR = "C:/Users/molle/OneDrive/Desktop/uc gan/optimizeddiscriminatormodel/real_resized_dir"
    IMAGE_SIZE = 64


    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(QUICK_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------------------------------
    # Device
    # -------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    
    # -------------------------------
    # Train the model
    # -------------------------------
    print("▶️ Starting training...")
    subprocess.run([sys.executable, "train.py", "--resume"], check=True)

    # -------------------------------
    # Load the last EMA checkpoint
    # -------------------------------
    ema_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("EMA_G_")])
    if not ema_files:
        raise FileNotFoundError("❌ No EMA_G_ checkpoint found!")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, ema_files[-1])
    print(f"Loaded checkpoint: {checkpoint_path}")

    G = Generator(nz=NZ).to(DEVICE)
    G.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    G.eval()

    # -------------------------------
    # Generate images
    # -------------------------------
    print("▶️ Generating images...")
    saved = 0

    with torch.no_grad():
        for _ in tqdm(range(0, N_GEN_IMAGES, GEN_BATCH)):
            cur_batch = min(GEN_BATCH, N_GEN_IMAGES - saved)
            z = torch.randn(cur_batch, NZ, 1, 1, device=DEVICE)
            imgs = G(z)
            imgs = (imgs * 0.5 + 0.5).clamp(0,1)

            for i in range(cur_batch):
                utils.save_image(imgs[i], os.path.join(GENERATED_DIR, f"{saved+i:05d}.png"))
            
            saved += cur_batch

    # -------------------------------
    # Create quick-FID subset folder
    # -------------------------------
    print(f"▶️ Preparing {QUICK_FID_NUM} images for quick FID...")
    for i in range(QUICK_FID_NUM):
        src = os.path.join(GENERATED_DIR, f"{i:05d}.png")
        dst = os.path.join(QUICK_DIR, f"{i:05d}.png")
        shutil.copy(src, dst)

    
    os.makedirs(REAL_RESIZED_DIR, exist_ok=True)

    for fname in os.listdir(SRC_REAL_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        src = os.path.join(SRC_REAL_DIR, fname)
        dst = os.path.join(REAL_RESIZED_DIR, fname)

        img = Image.open(src).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img.save(dst)

    print("✅ Real images resized to 64x64")

    # -------------------------------
    # Create quick-FID subset folder
    # -------------------------------
    print(f"▶️ Preparing {QUICK_FID_NUM} images for quick FID...")
    os.makedirs(QUICK_DIR, exist_ok=True)

    for i in range(QUICK_FID_NUM):
        src = os.path.join(GENERATED_DIR, f"{i:05d}.png")
        dst = os.path.join(QUICK_DIR, f"{i:05d}.png")

        img = Image.open(src).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img.save(dst)

    # -------------------------------
    # Quick FID
    # -------------------------------
    print("▶️ Calculating quick FID...")
    quick_metrics = calculate_metrics(
        input1=QUICK_DIR,
        input2=REAL_RESIZED_DIR,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        batch_size=50,
        num_workers=0   # Windows fix
    )
    quick_fid = quick_metrics["frechet_inception_distance"]
    print(f"Quick FID: {quick_fid:.2f}")

    # -------------------------------
    # Full FID
    # -------------------------------
    print("▶️ Calculating full FID (this may take long)...")
    full_metrics = calculate_metrics(
        input1=GENERATED_DIR,
        input2=REAL_RESIZED_DIR,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        batch_size=50,
        num_workers=0
    )
    full_fid = full_metrics["frechet_inception_distance"]
    print(f"Full FID: {full_fid:.2f}")

    # -------------------------------
    # Save scores
    # -------------------------------
    with open(FID_FILE, "w") as f:
        f.write(f"Quick FID: {quick_fid:.2f}\n")
        f.write(f"Full FID: {full_fid:.2f}\n")

    print("✅ Done! FID scores saved.")


if __name__ == "__main__":
    freeze_support()
    main()
