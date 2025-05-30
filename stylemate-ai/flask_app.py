import os, io, json
# ─── FORCE CPU-ONLY ENVIRONMENT ────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
# disable MPS backend (macOS GPU)
from flask_cors import CORS        # ← new import

torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
# limit threads to avoid BLAS contention
torch.set_num_threads(1)

from flask import Flask, request, jsonify, abort
import faiss
from PIL import Image
from clip_model import model, preprocess

# ─── MODEL SETUP ───────────────────────────────────────────────────────────────
device = torch.device("cpu")
model.to(device)
model.eval()

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE_DIR, "product.index")
METAS_FILE = os.path.join(BASE_DIR, "product_metas.json")

# ─── LOAD FAISS INDEX & METADATA ────────────────────────────────────────────────
index = faiss.read_index(INDEX_FILE)
with open(METAS_FILE, "r", encoding="utf-8") as f:
    metas = json.load(f)

# ─── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


def embed_image_bytes(data: bytes):
    """Embed a JPEG/PNG image in-memory via CLIP and return a normalized vector."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy().astype("float32")


@app.route('/recommend', methods=['POST'])
def recommend_api():
    if 'file' not in request.files:
        abort(400, description="No file part named 'file'.")
    file = request.files['file']
    try:
        data = file.read()
        q = embed_image_bytes(data)
    except Exception as e:
        abort(400, description=f"Invalid image: {e}")

    k = int(request.args.get('k', 5))
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        entry = metas[idx].copy()
        entry['score'] = float(score)
        results.append(entry)
    return jsonify(results)


if __name__ == '__main__':
    # debug mode, single-threaded
    app.run(host='127.0.0.1', port=8000, debug=True)
