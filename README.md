# Celebrity Retrieval Across Domains — Model Minds

Final submission for the Intro to Machine Learning Competition Project  
**University of Trento — M.Sc. Data Science — Spring 2026**  
**Group:** Model Minds  
**Final score:** 786 / 1000  

## Authors
- Alice Annibaletti
- Paola Fabris
- Giorgia Mazzarello
- Alessia Ronchetti

## Task
Given real photographs of celebrities as query images and synthetic images as a gallery,
the system retrieves the 10 most similar gallery images for each query, ranked by
descending similarity. Performance is evaluated as a weighted sum of Top-1, Top-5,
and Top-10 accuracy (max score: 1000).

## Final Approach
Triple ensemble of three pretrained frozen encoders:

| Model | Pretraining | Embedding dim | Weight |
|-------|-------------|---------------|--------|
| FaceNet (InceptionResnetV1) | VGGFace2 | 512-d | 0.6 |
| FaceNet (InceptionResnetV1) | CASIA-WebFace | 512-d | 0.1 |
| CLIP ViT-L/14 | OpenAI (image-text) | 768-d (projected) | 0.3 |

Each model produces L2-normalised embeddings. For each encoder a query×gallery
cosine similarity matrix is computed, normalised to [0,1] via min-max scaling,
and combined via weighted summation. The top-10 gallery images per query are
selected with `torch.topk`.

Face detection and alignment for FaceNet is handled by MTCNN (160×160 px crop,
margin=20, largest face only). When MTCNN fails to detect a face, a centre-square
crop resized to 160×160 is used as fallback.

## Requirements
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers facenet-pytorch Pillow
```

## Usage
Note: Before running, open ensemble_final.py and set data_folder (line 1 of the Config section) to the path of your local folder containing query/ and gallery/. The default value /home/disi is the path used on the competition VM.

1. Set `data_folder` in `ensemble_final.py` to the folder containing `query/` and `gallery/`
2. Run:
```bash
python ensemble_final.py
```
Results are saved to `results.json` as a dictionary mapping each query filename
to its ordered list of 10 gallery filenames.

## Results Summary

| Method | Final Score |
|--------|-------------|
| CLIP ViT-L/14 — CLS token (zero-shot) | 287 |
| CLIP ViT-L/14 — pooler output (zero-shot) | 655 |
| CLIP + ArcFace fine-tuning (5 epochs) | 670 |
| CLIP + ArcFace fine-tuning (15 epochs) | 710 |
| DINOv2 (zero-shot) | 119 |
| CLIP + DINOv2 ensemble (0.5/0.5) | 643 |
| FaceNet vggface2 — 160×160 | 744 |
| FaceNet casia-webface + CLIP (0.7/0.3) | 748 |
| FaceNet vggface2 + CLIP (0.7/0.3) | 784 |
| **Triple ensemble (0.6/0.1/0.3)** | **786** |

## Hardware
- GPU: NVIDIA Tesla V100-PCIE (16 GB VRAM)
- OS: Ubuntu 20.04
- CUDA: 12.4
