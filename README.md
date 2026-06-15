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
| CLIP ViT-L/14 | OpenAI (image-text) | 768-d | 0.3 |

Each model produces L2-normalised embeddings. For each encoder a query×gallery
cosine similarity matrix is computed, normalised to [0,1] via min-max scaling,
and combined via weighted summation. The top-10 gallery images per query are
selected with `torch.topk`.

Face detection and alignment for FaceNet is handled by MTCNN (160×160 px crop,
margin=20, largest face only). When MTCNN fails to detect a face, a centre-square
crop resized to 160×160 is used as fallback.

## Repository Structure
