import os
import json
import torch
from clip_model import get_image_embedding  # Make sure this works!

STYLE_FOLDER = "style_images"
OUTPUT_FILE = "reference_vectors.json"

def build_vectors():
    print("🚀 Starting reference vector generation...")
    print(f"📁 Looking inside folder: {STYLE_FOLDER}")

    reference_vectors = {}

    if not os.path.isdir(STYLE_FOLDER):
        print(f"❌ ERROR: Folder '{STYLE_FOLDER}' not found!")
        return

    for style_name in os.listdir(STYLE_FOLDER):
        style_path = os.path.join(STYLE_FOLDER, style_name)
        if not os.path.isdir(style_path):
            print(f"⏭️ Skipping non-folder: {style_path}")
            continue

        print(f"\n🔍 Processing style: {style_name}")
        embeddings = []

        for filename in os.listdir(style_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(style_path, filename)
                print(f"🖼️ Found image: {image_path}")

                try:
                    emb = get_image_embedding(image_path)
                    print(f"✅ Embedded image: {filename} → [dim {len(emb)}]")
                    embeddings.append(torch.tensor(emb))
                except Exception as e:
                    print(f"❌ Failed to embed {image_path}: {e}")
            else:
                print(f"⚠️ Skipping non-image file: {filename}")

        if embeddings:
            stacked = torch.stack(embeddings)
            avg_vector = stacked.mean(dim=0)
            avg_vector /= avg_vector.norm()  # normalize
            reference_vectors[style_name] = avg_vector.tolist()
            print(f"📦 Stored averaged vector for: {style_name}")
        else:
            print(f"⚠️ No valid images found for style: {style_name}")

    print("\n💾 Writing vectors to:", OUTPUT_FILE)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(reference_vectors, f)

    print("✅ Done! Reference vectors saved successfully.\n")

if __name__ == "__main__":
    build_vectors()
