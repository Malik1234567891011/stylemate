import os
import json
import requests
from io import BytesIO
from PIL import Image
import torch
from clip_model import model, preprocess

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
# Path to the JSON output from your scraper
PRODUCTS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'drmers_products.json')
)

# Where to write the product embeddings
OUTPUT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'product_vectors.json')
)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert('RGB')


def embed_image(img: Image.Image) -> list:
    """Run CLIP model on a PIL image and return a normalized vector."""
    x = preprocess(img).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        emb = model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().tolist()

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
def build_product_vectors():
    # load scraped products
    if not os.path.isfile(PRODUCTS_FILE):
        print(f"❌ Products file not found: {PRODUCTS_FILE}")
        return

    with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        products = json.load(f)

    product_vectors = []
    print(f"🔍 Embedding {len(products)} products...")

    for idx, prod in enumerate(products, 1):
        url = prod.get('image_url')
        meta = {k: prod[k] for k in ('title','price','url')}

        if not url:
            print(f"⚠️  [{idx}] No image URL for {meta['title']}")
            continue

        try:
            img = fetch_image(url)
            vec = embed_image(img)
            product_vectors.append({
                'meta': meta,
                'vector': vec
            })
            print(f"✅  [{idx}] Embedded: {meta['title']}")
        except Exception as e:
            print(f"❌  [{idx}] Failed to embed {meta['title']}: {e}")

    # write out
    print(f"💾 Saving {len(product_vectors)} vectors to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(product_vectors, f, indent=2, ensure_ascii=False)
    print("🎉 Done!")


if __name__ == '__main__':
    build_product_vectors()
