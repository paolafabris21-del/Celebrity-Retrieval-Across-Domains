"""
ensemble_final.py
=================
Final submission script — Celebrity Retrieval Across Domains
Group: Model Minds — University of Trento, M.Sc. Data Science, Spring 2026
Final score: 786 / 1000
 
Triple ensemble: FaceNet (vggface2) + FaceNet (casia-webface) + CLIP ViT-L/14
Weights: 0.6 / 0.1 / 0.3
 
Usage:
    1. Set data_folder below to the folder containing query/ and gallery/
    2. Run: python ensemble_final.py
    3. Results are saved to results.json
 
Requirements:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install transformers facenet-pytorch Pillow
    CUDA install required for GPU support. For CPU-only, use: pip install torch torchvision
"""
 
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms.functional as TF
 
# ── Config ─────────────────────────────────────────────────────────────────────
data_folder    = "/home/disi"          # <-- change to your path
query_folder   = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")
 
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
 
# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
 
 
# ── Load images ────────────────────────────────────────────────────────────────
def load_images_from_folder(folder):
    images, filenames = [], []
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() in EXTENSIONS:
            images.append(Image.open(os.path.join(folder, fname)).convert("RGB"))
            filenames.append(fname)
    return images, filenames
 
 
print("Loading images...")
query_images,   query_filenames   = load_images_from_folder(query_folder)
gallery_images, gallery_filenames = load_images_from_folder(gallery_folder)
print(f"  {len(query_images)} query / {len(gallery_images)} gallery")
 
 
# ── FaceNet features ───────────────────────────────────────────────────────────
def facenet_features(images, pretrained, batch_size=32):
    mtcnn  = MTCNN(image_size=160, margin=20, keep_all=False,
                   device=device, post_process=True)
    resnet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    all_features = []
    no_face = 0
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        crops = []
        for img in batch:
            face_tensor = mtcnn(img)
            if face_tensor is not None:
                crops.append(face_tensor)
            else:
                no_face += 1
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top  = (h - side) // 2
                crop = img.crop((left, top, left+side, top+side)).resize((160, 160))
                t = TF.to_tensor(crop)
                t = (t - 0.5) / 0.5
                crops.append(t)
        batch_tensor = torch.stack(crops).to(device)
        with torch.no_grad():
            embs = resnet(batch_tensor)
            embs = F.normalize(embs, p=2, dim=1)
        all_features.append(embs.cpu())
        print(f"  FaceNet-{pretrained} {min(i+batch_size, len(images))}/{len(images)}", end="\r")
    print(f"\n  No-face fallbacks: {no_face}/{len(images)}")
    del resnet, mtcnn
    torch.cuda.empty_cache()
    return torch.cat(all_features, dim=0)
 
 
# ── CLIP features ──────────────────────────────────────────────────────────────
def clip_features(images, batch_size=32):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        pv = clip_proc(images=batch, return_tensors="pt",
                       padding=True)['pixel_values'].to(device)
        with torch.no_grad():
            f = clip_model.vision_model(pixel_values=pv).pooler_output
            f = clip_model.visual_projection(f)
        feats.append(F.normalize(f, p=2, dim=1).cpu())
        print(f"  CLIP {min(i+batch_size, len(images))}/{len(images)}", end="\r")
    print()
    del clip_model
    torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)
 
 
# ── Min-max normalization ──────────────────────────────────────────────────────
def minmax(s):
    mn = s.min(dim=1, keepdim=True).values
    mx = s.max(dim=1, keepdim=True).values
    return (s - mn) / (mx - mn + 1e-8)
 
 
# ── Extract features ───────────────────────────────────────────────────────────
print("\nExtracting FaceNet vggface2...")
vgg_q = facenet_features(query_images,   "vggface2")
vgg_g = facenet_features(gallery_images, "vggface2")
 
print("\nExtracting FaceNet casia-webface...")
cas_q = facenet_features(query_images,   "casia-webface")
cas_g = facenet_features(gallery_images, "casia-webface")
 
print("\nExtracting CLIP features...")
clip_q = clip_features(query_images)
clip_g = clip_features(gallery_images)
 
 
# ── Ensemble ───────────────────────────────────────────────────────────────────
print("\nComputing ensemble...")
sim_ensemble = (0.6 * minmax(torch.matmul(vgg_q,  vgg_g.T))
              + 0.1 * minmax(torch.matmul(cas_q,  cas_g.T))
              + 0.3 * minmax(torch.matmul(clip_q, clip_g.T)))
 
 
# ── Ranking ────────────────────────────────────────────────────────────────────
TOP_K = min(10, len(gallery_filenames))
_, top_k_indices = torch.topk(sim_ensemble, k=TOP_K, dim=1)
 
results = {}
for i, qname in enumerate(query_filenames):
    results[qname] = [gallery_filenames[idx] for idx in top_k_indices[i].tolist()]
 
 
# ── Save results ───────────────────────────────────────────────────────────────
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results.json")
