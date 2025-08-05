#!/usr/bin/env python3
# ───────────────────────────────────────────────────────────────────────────────
# Disable all GPU/MPS backends before importing torch/faiss
# (This avoids MPS/CUDA‐related segfaults on Mac, etc.)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prevent MPS backend from initializing:
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built     = lambda: False

# ───────────────────────────────────────────────────────────────────────────────
import sys
import importlib
import argparse
import json
import numpy as np
import faiss
import requests
from io import BytesIO
from PIL import Image

# ─── MAKE SURE PROJECT ROOT IS ON sys.path ────────────────────────────────────
# so that `import clip_model` (and others in root) works from here.
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# ─── IMPORT CLIP MODEL TO EMBED IMAGES ────────────────────────────────────────
import clip_model  # your clip_model.py lives at project root; it defines `model` and `preprocess`

# ── UTILITY: fetch an image from its URL → PIL.Image ──────────────────────────
def fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# ── UTILITY: given a PIL image, return a normalized embed as a Python list ──
def embed_image(img: Image.Image) -> list:
    x = clip_model.preprocess(img).unsqueeze(0)  # 1×3×224×224
    with torch.no_grad():
        emb = clip_model.model.encode_image(x)    # 1×D
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().tolist()

# ── UTILITY: build & write a Faiss index (inner product on L2‐normalized vectors) ─
def build_faiss_index_from_vectors(vectors_list: list, metas_list: list, index_path: str, metas_path: str):
    # Convert to numpy float32
    vectors_np = np.array(vectors_list, dtype="float32")
    # Normalize for cosine search
    faiss.normalize_L2(vectors_np)
    # Create a flat Inner‐Product index
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    # Write the index to disk
    faiss.write_index(index, index_path)
    # Write metadata JSON
    with open(metas_path, "w", encoding="utf-8") as mf:
        json.dump(metas_list, mf, indent=2, ensure_ascii=False)
    print(f"✅ Indexed {len(metas_list)} entries → {index_path}")

# ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────────────
def run_full_pipeline(scraper_module: str, output_folder: str):
    """
    1) Dynamically import `scrapers/{scraper_module}.py` and call `scrape()`.
    2) Write scraped products → <brand>_products.json
    3) Embed each product image → <brand>_product_vectors.json
    4) Build a Faiss index + write <brand>.index + <brand>_metas.json
    """
    # 1) Import the scraper module
    try:
        scraper = importlib.import_module(f"scrapers.{scraper_module}")
    except ImportError as e:
        print(f"❌ Error: could not import scraper `scrapers/{scraper_module}.py`: {e}")
        sys.exit(1)

    if not hasattr(scraper, "scrape"):
        print(f"❌ Error: `{scraper_module}.py` must define a function `scrape()`.")
        sys.exit(1)

    print(f"\n🔍 Running scraper `scrapers/{scraper_module}.py` …")
    products = scraper.scrape()
    brand_name = scraper_module.replace("_scraper", "")  # e.g. "galore_scraper" → "galore"

    # 2) Write scraped products to JSON
    os.makedirs(output_folder, exist_ok=True)
    products_path = os.path.join(output_folder, f"{brand_name}_products.json")
    print(f"   • Writing scraped data ({len(products)} items) → {products_path}")
    with open(products_path, "w", encoding="utf-8") as pf:
        json.dump(products, pf, indent=2, ensure_ascii=False)

    # 3) Embed images and collect (meta, vector)
    embeddings = []
    metas = []
    print(f"\n🔍 Embedding {len(products)} images …")
    # Ensure model is on CPU
    device = torch.device("cpu")
    clip_model.model.to(device).eval()

    for idx, prod in enumerate(products, start=1):
        title = prod.get("title", "<no-title>")
        image_url = prod.get("image", "")
        meta = { "title": title, "price": prod.get("price"), "url": prod.get("url") }

        if not image_url:
            print(f"⚠️  [{idx}] No image_url for {title!r}; skipping.")
            continue

        try:
            img = fetch_image(image_url)
            vec = embed_image(img)
            embeddings.append(vec)
            metas.append(meta)
            print(f"   ✅  [{idx}] Embedded: {title!r}")
        except Exception as e:
            print(f"   ❌  [{idx}] Failed to embed {title!r}: {e}")

    # 3b) Write out product_vectors.json
    vectors_path = os.path.join(output_folder, f"{brand_name}_product_vectors.json")
    print(f"\n💾 Saving {len(embeddings)} vectors → {vectors_path}")
    out_list = []
    for i in range(len(embeddings)):
        out_list.append({ "meta": metas[i], "vector": embeddings[i] })
    with open(vectors_path, "w", encoding="utf-8") as vf:
        json.dump(out_list, vf, indent=2, ensure_ascii=False)

    # 4) Build & save Faiss index + metas
    index_path = os.path.join(output_folder, f"{brand_name}.index")
    metas_path = os.path.join(output_folder, f"{brand_name}_metas.json")
    print(f"\n🔨 Building Faiss index for {brand_name} …")
    build_faiss_index_from_vectors(embeddings, metas, index_path, metas_path)

    print("\n🎉 Pipeline complete!\n")
    print(f"  ↳ Scraped JSON →     {products_path}")
    print(f"  ↳ Vector JSON →      {vectors_path}")
    print(f"  ↳ Faiss index →      {index_path}")
    print(f"  ↳ Metadata JSON →    {metas_path}\n")


# ─── COMMAND‐LINE INTERFACE ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full scrape→embed→index pipeline for a given brand."
    )
    parser.add_argument(
        "--scraper", "-s", required=True,
        help="Name of the scraper under `scrapers/` (omit `.py`), e.g. `drmers_scraper` or `galore_scraper`."
    )
    parser.add_argument(
        "--outdir", "-o",
        default=os.path.abspath(os.path.join(root_dir, "data")),
        help="Output directory for JSON + index files (default: <project>/data)"
    )
    args = parser.parse_args()
    run_full_pipeline(args.scraper, args.outdir)
