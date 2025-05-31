# stylemate-ai/flask_app.py

import os
import io
import json
import torch
from flask_cors import CORS
from flask import Flask, request, jsonify, abort
import faiss
from PIL import Image
from clip_model import model, preprocess

# ─── FORCE CPU ONLY ────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # disable CUDA/MPS
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built     = lambda: False
torch.set_num_threads(1)           # avoid BLAS contention

# ─── LOAD & PREPARE CLIP MODEL ─────────────────────────────────────────────────
device = torch.device("cpu")
model.to(device)
model.eval()

# ─── CONFIG: PATHS TO EACH BRAND’S INDEX + METADATA ────────────────────────────
BASE_DIR = os.path.dirname(__file__)

# ─── Drmers: FAISS index + metas JSON (these live at the root of stylemate-ai/)
DRMERS_INDEX = os.path.join(BASE_DIR, "product.index")
DRMERS_METAS = os.path.join(BASE_DIR, "product_metas.json")

# ─── Galore: FAISS index + metas JSON (these live under stylemate-ai/data/)
GALORE_INDEX = os.path.join(BASE_DIR, "data", "galore.index")
GALORE_METAS = os.path.join(BASE_DIR, "data", "galore_metas.json")

# ─── Container to hold both (faiss_index, metas_list) pairs ──────────────────
brand_indices = []


# ─── LOAD Drmers INDEX + METADATA ─────────────────────────────────────────────
if not os.path.exists(DRMERS_INDEX):
    raise RuntimeError(f"Missing Drmers index at: {DRMERS_INDEX}")
if not os.path.exists(DRMERS_METAS):
    raise RuntimeError(f"Missing Drmers metadata file at: {DRMERS_METAS}")

dr_index = faiss.read_index(DRMERS_INDEX)
with open(DRMERS_METAS, "r", encoding="utf-8") as f:
    dr_metas = json.load(f)

brand_indices.append((dr_index, dr_metas))


# ─── LOAD Galore INDEX + METADATA ─────────────────────────────────────────────
if not os.path.exists(GALORE_INDEX):
    raise RuntimeError(f"Missing Galore index at: {GALORE_INDEX}")
if not os.path.exists(GALORE_METAS):
    raise RuntimeError(f"Missing Galore metadata file at: {GALORE_METAS}")

ga_index = faiss.read_index(GALORE_INDEX)
with open(GALORE_METAS, "r", encoding="utf-8") as f:
    ga_metas = json.load(f)

brand_indices.append((ga_index, ga_metas))


# ─── FLASK APP SETUP ─────────────────────────────────────────────────────────
app = Flask(__name__)
# Allow your React dev server (http://localhost:5173) to hit this endpoint
CORS(app, origins=["http://localhost:5173"])


def embed_image_bytes(data: bytes):
    """
    Embed raw image bytes via CLIP. Returns a (1 × D) numpy array (dtype=float32),
    normalized so that inner‐product == cosine‐similarity.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy().astype("float32")   # shape: (1, D)


@app.route("/recommend", methods=["POST"])
def recommend_api():
    """
    POST /recommend?k=5
    - Expect a multipart form‐file under key="file".
    - Optional query parameter 'k' (default=5) controls how many neighbors each brand returns,
      then we merge both brands' results and return the top‐k overall.
    """
    if "file" not in request.files:
        abort(400, description="No file part named 'file'. Please upload an image using key='file'.")

    file = request.files["file"]
    try:
        raw = file.read()
        q_vec = embed_image_bytes(raw)   # shape = (1, D)
    except Exception as e:
        abort(400, description=f"Invalid image or embedding error: {e}")

    # parse k (how many results PER BRAND to fetch) from ?k=
    try:
        k = int(request.args.get("k", 5))
        if k <= 0:
            raise ValueError()
    except ValueError:
        abort(400, description="Query parameter 'k' must be a positive integer.")

    # Collect all matches from each brand
    all_results = []
    for faiss_idx, metas in brand_indices:
        # This search returns two arrays of shape (1, k): distances and indices
        distances, indices = faiss_idx.search(q_vec, k)
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            entry = metas[idx].copy()
            entry["score"] = float(score)
            all_results.append(entry)

    # Merge both brands' candidates, sort by descending score, take overall top‐k
    all_results.sort(key=lambda x: x["score"], reverse=True)
    topk = all_results[:k]

    return jsonify(topk)


if __name__ == "__main__":
    # Launch Flask on http://127.0.0.1:8000 (debug mode)
    app.run(host="127.0.0.1", port=8000, debug=True)
