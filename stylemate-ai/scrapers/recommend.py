import os
# Force CPU-only to avoid MPS/CUDA segfaults
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Disable MPS backend if present
import torch
torch.backends.mps.is_available = lambda : False
torch.backends.mps.is_built = lambda : False
# Limit threads
torch.set_num_threads(1)

import sys, json
import faiss
from PIL import Image
from clip_model import model, preprocess

# Move model to CPU explicitly
device = torch.device('cpu')
model = model.to(device)

# ─ CONFIG ──────────────────────────────────────────────────────────────────────
INDEX_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', 'product.index'))
METAS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', 'product_metas.json'))

# ─ DEBUG PRINTS ─────────────────────────────────────────────────────────────────
print(f"Loading FAISS index from: {INDEX_FILE}")
# ─ LOAD INDEX & METAS ───────────────────────────────────────────────────────────
index = faiss.read_index(INDEX_FILE)
print("Index loaded successfully.")

with open(METAS_FILE, 'r', encoding='utf-8') as f:
    metas = json.load(f)
print(f"Loaded {len(metas)} metadata entries.")


def recommend(image_path: str, k: int = 5) -> list:
    """
    Given a local image path, returns the top-k matching products.
    """
    print(f"Embedding input image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    q = q.cpu().numpy().astype('float32')
    print("Query vector ready. Performing search...")

    # search
    D, I = index.search(q, k)
    print(f"Search completed. Top {k} scores: {D[0].tolist()}")
    results = []
    for score, idx in zip(D[0], I[0]):
        meta = metas[idx]
        results.append({
            **meta,
            'score': float(score)
        })
    return results

# CLI example
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <path_to_image> [k]")
        sys.exit(1)
    path = sys.argv[1]
    k    = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    recs = recommend(path, k)
    print(json.dumps(recs, indent=2))
