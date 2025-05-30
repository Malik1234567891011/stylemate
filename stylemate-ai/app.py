# stylemate-ai/app.py

import os, io, json
import faiss
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from PIL import Image
from clip_model import model, preprocess

# ─── FORCE CPU ────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")
model.to(device)
model.eval()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE_DIR, "product.index")
METAS_FILE = os.path.join(BASE_DIR, "product_metas.json")

# ─── LOAD FAISS & METADATA ────────────────────────────────────────────────────
index = faiss.read_index(INDEX_FILE)
with open(METAS_FILE, "r", encoding="utf-8") as f:
    metas = json.load(f)

# ─── FASTAPI SETUP ────────────────────────────────────────────────────────────
app = FastAPI(title="StyleMate Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or your React dev URL
    allow_methods=["POST"],
    allow_headers=["*"],
)

def embed_image_bytes(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(x)
        q = q / q.norm(dim=-1, keepdim=True)
    return q.cpu().numpy().astype("float32")

@app.post("/recommend")
async def recommend(file: UploadFile = File(...), k: int = 5):
    """
    Upload an image (JPEG/PNG) and get top-k product recommendations.
    """
    try:
        content = await file.read()
        q = embed_image_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # vector search
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        entry = metas[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return JSONResponse(results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
