#!/usr/bin/env python3
import os
import sys

# ─── Make sure the project root (stylemate-ai/) is on sys.path ────────────────
# If this file lives at stylemate-ai/scrapers/build_product_vectors.py,
# then `root_dir` will be stylemate-ai/.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ───────────────────────────────────────────────────────────────────────────────
import json
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from clip_model import model, preprocess   # ← now this works

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def fetch_image(url: str) -> Image.Image:
    """Download an image from a URL and return a PIL Image (RGB)."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def embed_image(img: Image.Image) -> list:
    """
    Run the global CLIP model on a PIL image and return a normalized vector.
    Assumes that `clip_model.model` and `clip_model.preprocess` are already imported.
    """
    x = preprocess(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().tolist()

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────

def build_product_vectors(input_path: str, output_path: str):
    """
    1. Loads your scraped JSON (list of { title, price, url, image_url, … } entries).
    2. Downloads each `image_url`, runs it through CLIP, and gathers the resulting vector.
    3. Writes out a new JSON array of { meta: {title,price,url}, vector: [...] } to `output_path`.
    """
    if not os.path.isfile(input_path):
        print(f"❌ Products file not found: {input_path}")
        return

    # Load the scraped list of products
    with open(input_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    product_vectors = []
    print(f"🔍 Embedding {len(products)} products from: {input_path}\n")

    for idx, prod in enumerate(products, start=1):
        image_url = prod.get("image_url", "")
        meta = {
            "title": prod.get("title"),
            "price": prod.get("price"),
            "url": prod.get("url"),
        }

        if not image_url:
            print(f"⚠️  [{idx}] Skipping (no image_url): {meta['title']}")
            continue

        try:
            img = fetch_image(image_url)
            vec = embed_image(img)
            product_vectors.append({
                "meta": meta,
                "vector": vec
            })
            print(f"✅  [{idx}] Embedded: {meta['title']}")
        except Exception as e:
            print(f"❌  [{idx}] Failed to embed {meta['title']}: {e}")

    # Write out the embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n💾 Saving {len(product_vectors)} vectors to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(product_vectors, out_f, indent=2, ensure_ascii=False)

    print("🎉 Done!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed all product images (from a scraped JSON) into CLIP vectors."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help=(
            "Path to the scraped JSON file (e.g. data/drmers_products.json or data/galore_products.json). "
            "Each entry must include at least { 'title', 'price', 'url', 'image_url', … }."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help=(
            "Where to write the JSON of embedded products. "
            "This script will produce an array of { 'meta': {title,price,url}, 'vector': … }."
        )
    )
    args = parser.parse_args()

    # Force CLIP to CPU (in case GPU/MPS is enabled)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA GPUs
    torch.backends.mps.is_available = lambda: False  # disable MPS if on macOS
    torch.backends.mps.is_built = lambda: False

    # Put the CLIP model on CPU
    device = torch.device("cpu")
    model.to(device).eval()

    build_product_vectors(
        input_path=args.input,
        output_path=args.output
    )
