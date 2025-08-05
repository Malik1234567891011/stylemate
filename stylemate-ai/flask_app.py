import os
import io
import json
import time
import uuid
import torch
import faiss
from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
from PIL import Image
from clip_model import model, preprocess

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False
torch.set_num_threads(1)

device = torch.device("cpu")
model.to(device)
model.eval()

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_FILE = os.path.join(BASE_DIR, "user_data.json")
brand_indices = []

BRANDS = [
    {
        "name": "Drmers",
        "index": os.path.join(BASE_DIR, "product.index"),
        "meta": os.path.join(BASE_DIR, "product_metas.json"),
    },
    {
        "name": "Supermade",
        "index": os.path.join(BASE_DIR, "data", "supermade.index"),
        "meta": os.path.join(BASE_DIR, "data", "supermade_metas.json"),
    },
    {
        "name": "Gymshark",
        "index": os.path.join(BASE_DIR, "data", "gymshark.index"),
        "meta": os.path.join(BASE_DIR, "data", "gymshark_metas.json"),
    },
]


# Load brand indices
for brand in BRANDS:
    if not os.path.exists(brand["index"]) or not os.path.exists(brand["meta"]):
        raise RuntimeError(f"Missing files for {brand['name']}")
    idx = faiss.read_index(brand["index"])
    with open(brand["meta"], "r", encoding="utf-8") as f:
        metas = json.load(f)
    brand_indices.append((idx, metas))

# Ensure folders/files exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Vectorize image
def embed_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy().astype("float32")

def load_user_vectors():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_user_vector(vec, timestamp, filename=None, clothing_type=None):
    data = load_user_vectors()
    data.append({
        "vector": vec.tolist(),
        "timestamp": timestamp,
        "filename": filename or "",
        "type": clothing_type or ""
    })
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# 🔥 Hybrid Upload Endpoint
@app.route("/upload_fit", methods=["POST"])
def upload_fit():
    print("✅ Received upload request")

    if "file" not in request.files:
        print("❌ No file in request")
        abort(400, description="Missing file part")

    file = request.files["file"]
    save_image = request.form.get("saveImage", "false").lower() == "true"
    clothing_type = request.form.get("type", "").strip()
    print(f"👕 Clothing type: {clothing_type}, Save image: {save_image}")

    try:
        buffer = file.read()
        print(f"📦 Received {len(buffer)} bytes")
        vec = embed_image_bytes(buffer)
        print("🧠 Embedding successful")

        timestamp = time.time()
        filename = ""

        if save_image:
            ext = file.filename.rsplit(".", 1)[-1]
            filename = f"{uuid.uuid4()}.{ext}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            with open(filepath, "wb") as f_out:
                f_out.write(buffer)
            print(f"💾 Saved file to {filepath}")

        save_user_vector(vec[0], timestamp, filename, clothing_type)
        print("✅ Vector saved to user_data.json")
        return jsonify({"status": "ok", "timestamp": timestamp})

    except Exception as e:
        print("🔥 ERROR during upload:", e)
        abort(400, description=f"Failed to embed/upload: {e}")


# 🔄 Overwrite closet (used for delete)
@app.route("/closet", methods=["PUT"])
def overwrite_closet():
    try:
        updated = request.get_json()
        if not isinstance(updated, list):
            abort(400, description="Closet data must be a list")
        for entry in updated:
            if "vector" not in entry or "timestamp" not in entry or "filename" not in entry:
                abort(400, description="Missing required keys in closet item")
        with open(DATA_FILE, "w") as f:
            json.dump(updated, f)
        return jsonify({"status": "updated", "count": len(updated)})
    except Exception as e:
        abort(400, description=f"Failed to update closet: {e}")

# Recommend from image
@app.route("/recommend", methods=["POST"])
def recommend_api():
    if "file" not in request.files:
        abort(400, description="Missing file part")
    file = request.files["file"]
    try:
        q_vec = embed_image_bytes(file.read())
    except Exception as e:
        abort(400, description=f"Embed error: {e}")

    k = int(request.args.get("k", 5))
    page = int(request.args.get("page", 1))
    if k <= 0 or page <= 0:
        abort(400, description="Invalid k or page")

    all_results = []
    for faiss_idx, metas in brand_indices:
        distances, indices = faiss_idx.search(q_vec, 100)
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            entry = metas[idx].copy()
            entry["score"] = float(score)
            all_results.append(entry)

    all_results.sort(key=lambda x: x["score"], reverse=True)
    start = (page - 1) * k
    end = start + k
    return jsonify(all_results[start:end])

# Recommend based on average
@app.route("/recommend_expand", methods=["GET"])
def recommend_expand():
    vectors = load_user_vectors()
    if not vectors:
        abort(400, description="No user wardrobe vectors found")

    # optional type filtering
    requested_type = request.args.get("type")
    if requested_type:
        vectors = [x for x in vectors if x.get("type") == requested_type]

    if not vectors:
        abort(400, description="No matching vectors for selected type")

    vecs = torch.tensor([x["vector"] for x in vectors], dtype=torch.float32)
    avg_vec = vecs.mean(dim=0).numpy().reshape(1, -1)

    k = int(request.args.get("k", 5))
    if k <= 0:
        abort(400, description="Invalid k")

    all_results = []
    for faiss_idx, metas in brand_indices:
        distances, indices = faiss_idx.search(avg_vec.astype("float32"), k)
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            entry = metas[idx].copy()
            entry["score"] = float(score)
            all_results.append(entry)

    all_results.sort(key=lambda x: x["score"])
    return jsonify(all_results[:k])


@app.route("/recommend_mode", methods=["GET"])
def recommend_by_mode():
    vectors = load_user_vectors()
    if not vectors:
        abort(400, description="No user wardrobe vectors found")

    mode = request.args.get("mode", "aesthetic")
    k = int(request.args.get("k", 5))
    clothing_type = request.args.get("type")
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)

    # Filter vectors by clothing type
    if clothing_type:
        vectors = [v for v in vectors if v.get("type") == clothing_type]
        if not vectors:
            abort(400, description="No matching vectors for selected type")

    if mode == "single":
        target_timestamp = request.args.get("timestamp", type=float)
        match = next((v for v in vectors if v["timestamp"] == target_timestamp), None)
        if not match:
            abort(400, description="No item with the given timestamp found")
        vec = torch.tensor(match["vector"]).reshape(1, -1).numpy().astype("float32")
    else:
        vecs = torch.tensor([v["vector"] for v in vectors])
        avg_vec = vecs.mean(dim=0).numpy()
        vec = avg_vec.reshape(1, -1).astype("float32")

    all_results = []
    for faiss_idx, metas in brand_indices:
        distances, indices = faiss_idx.search(vec, 100)
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            entry = metas[idx].copy()
            entry["score"] = float(score)
            all_results.append(entry)

    # Sort and filter
    all_results.sort(key=lambda x: x["score"], reverse=(mode == "single"))

    if min_price is not None:
        all_results = [x for x in all_results if "price" in x and x["price"] >= min_price]
    if max_price is not None:
        all_results = [x for x in all_results if "price" in x and x["price"] <= max_price]

    return jsonify(all_results[:k])


# Get Closet
@app.route("/closet", methods=["GET"])
def get_closet():
    return jsonify(load_user_vectors())

# Serve Uploaded Images (for those saved)
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# Start app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
