import sys
import json
import os
from clip_model import get_image_embedding
from utils import cosine_similarity

def predict_style(image_path: str, reference_file: str = "reference_vectors.json") -> str:
    if not os.path.isfile(image_path):
        print(f"❌ Error: File not found → {image_path}")
        return "Invalid image"

    try:
        print(f"\n📸 Embedding input image: {image_path}")
        user_vector = get_image_embedding(image_path)
    except Exception as e:
        print(f"❌ Failed to embed image: {e}")
        return "Embedding failed"

    with open(reference_file, "r") as f:
        reference_vectors = json.load(f)

    print("\n🔎 Comparing to style reference vectors...")
    best_match = None
    highest_score = -1

    for style, ref_vector in reference_vectors.items():
        score = cosine_similarity(user_vector, ref_vector)
        print(f"  • {style:<12} → similarity: {score:.4f}")
        if score > highest_score:
            highest_score = score
            best_match = style

    if best_match:
        print(f"\n✅ Predicted style: **{best_match}** (score: {highest_score:.4f})")
        return best_match
    else:
        print("⚠️ No match found.")
        return "No match"

# Optional CLI support
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_style.py <path_to_image>")
        sys.exit(1)
    predict_style(sys.argv[1])
