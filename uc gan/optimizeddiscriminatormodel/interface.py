import os
import time
import uuid

import gradio as gr
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from main3 import Generator  # your DCGAN generator class

# ---------------------------
# Config
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NZ = 100
COND_DIM = 10
GEN_PTH = "checkpoints/EMA_G_epoch100.pth"

# ---------------------------
# Load Generator
# ---------------------------
G = Generator(nz=NZ, cond_dim=COND_DIM).to(DEVICE)

# Safe checkpoint loader: if a tensor in the checkpoint has a smaller first
# dimension than the model (e.g. original nz vs nz+cond_dim), expand the
# checkpoint tensor by copying existing values and init'ing the new slice.
def safe_load_checkpoint(model, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    model_sd = model.state_dict()
    new_sd = {}
    for k, v in model_sd.items():
        if k in sd:
            ck = sd[k]
            if ck.shape == v.shape:
                new_sd[k] = ck
            else:
                # try to expand along dim 0 when compatible on remaining dims
                if isinstance(ck, torch.Tensor) and ck.ndim == v.ndim:
                    compatible = True
                    for d in range(1, ck.ndim):
                        if ck.shape[d] != v.shape[d]:
                            compatible = False
                            break
                    if compatible and ck.shape[0] < v.shape[0]:
                        new_t = torch.zeros(v.shape, dtype=ck.dtype, device=ck.device)
                        new_t[: ck.shape[0]] = ck
                        # init new slice with same normal init used elsewhere
                        try:
                            torch.nn.init.normal_(new_t[ck.shape[0]:], 0.0, 0.02)
                        except Exception:
                            pass
                        new_sd[k] = new_t
                    else:
                        # incompatible shapes: skip this key
                        print(f"Skipping incompatible key: {k} ckpt {ck.shape} vs model {v.shape}")
                else:
                    print(f"Skipping non-tensor or ndim mismatch for key: {k}")
        else:
            # key missing in checkpoint; keep current model value
            new_sd[k] = v

    # load adjusted state dict non-strictly
    model.load_state_dict(new_sd, strict=False)


# attempt to load
try:
    safe_load_checkpoint(G, GEN_PTH, map_location=DEVICE)
except FileNotFoundError:
    print(f"Checkpoint not found: {GEN_PTH} — continuing with random init")
G.eval()  # evaluation mode




def _parse_feature_text(s, dim):
    if not s:
        return [0.0] * dim
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            continue
    # pad or truncate
    if len(vals) < dim:
        vals += [0.0] * (dim - len(vals))
    return vals[:dim]


def generate_faces(num_faces=1, feature_text="", progress=gr.Progress()):
    # prepare noise
    z = torch.randn(num_faces, NZ, 1, 1, device=DEVICE)
    # parse features and create condition batch
    vals = _parse_feature_text(feature_text, COND_DIM)
    cond = torch.tensor(vals, device=DEVICE, dtype=torch.float32).unsqueeze(0).repeat(num_faces, 1)

    with torch.no_grad():
        imgs = G(z, cond)
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

    grid = make_grid(imgs, nrow=min(10, num_faces))

    # create a unique filename to avoid caching issues
    filename = os.path.abspath(f"temp_{uuid.uuid4().hex}.png")
    save_image(grid, filename)

    # load as PIL image for Gradio Image output
    pil_img = Image.open(filename).convert("RGB")

    return pil_img, filename

'''# ---------------------------
# Gradio Interface
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Anime Face Generator")
    
    with gr.Row():
        num_faces_slider = gr.Slider(1, 12, value=4, step=1, label="Number of Faces")
        generate_button = gr.Button("Generate", elem_id="green_button")

    output_img = gr.Image(label="Generated Faces")
    download_btn = gr.File(label="Download Grid")

    generate_button.click(
        generate_faces,
        inputs=[num_faces_slider],
        outputs=[output_img, download_btn]
    )

# ---------------------------
# Custom CSS for green button
# ---------------------------
css = """
#green_button {
    background-color: #4CAF50 !important; 
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
}
"""
demo.launch(inline_templates={"style": css})'''

css = """
#green_button {
    background-color: #4CAF50 !important; 
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 🎨 Anime Face Generator")
    
    with gr.Row():
        num_faces_slider = gr.Slider(1, 120, value=4, step=1, label="Number of Faces")
        feature_input = gr.Textbox(label=f"Feature vector (comma-separated, length {COND_DIM})", lines=1)
        generate_button = gr.Button("Generate", elem_id="green_button")

    output_img = gr.Image(label="Generated Faces")
    download_btn = gr.File(label="Download Grid")

    generate_button.click(
        generate_faces,
        inputs=[num_faces_slider, feature_input],
        outputs=[output_img, download_btn]
    )

if __name__ == "__main__":
    demo.launch()
