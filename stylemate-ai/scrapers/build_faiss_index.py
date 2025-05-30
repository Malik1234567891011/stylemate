import os, json
import numpy as np
import faiss

# ─ CONFIG ──────────────────────────────────────────────────────────────────────
VECTORS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '..', 'product_vectors.json'))
INDEX_FILE   = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '..', 'product.index'))
METAS_FILE   = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '..', 'product_metas.json'))

# ─ LOAD VECTORS & METADATA ──────────────────────────────────────────────────────
with open(VECTORS_FILE, 'r', encoding='utf-8') as f:
    items = json.load(f)

# extract numpy arrays
vectors = np.array([it['vector'] for it in items], dtype='float32')
metas   = [it['meta'] for it in items]

# normalize for inner-product (cosine) search
faiss.normalize_L2(vectors)

# ─ BUILD & SAVE INDEX ───────────────────────────────────────────────────────────
dim   = vectors.shape[1]
index = faiss.IndexFlatIP(dim)      # use Inner Product => cosine if normalized
index.add(vectors)                  # add all vectors

# persist
faiss.write_index(index, INDEX_FILE)
with open(METAS_FILE, 'w', encoding='utf-8') as f:
    json.dump(metas, f, indent=2, ensure_ascii=False)

print(f"✅ Indexed {len(metas)} products → {INDEX_FILE}")
