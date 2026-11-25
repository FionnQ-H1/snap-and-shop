"""Baseline test: how often does a pretrained MobileNetV3 (from Hugging Face) find the right product
in DeepFashion In-shop?

For each "query" photo (a shopper-style photo), it searches the "gallery" (the shop's product photos)
and checks whether a photo of the same item is in the top 5 results (recall@5).
"""
import random
import time
from pathlib import Path

import timm
import torch
from PIL import Image

DATA = Path(__file__).parent / "data" / "deepfashion"
PARTITION = DATA / "Eval" / "list_eval_partition.txt"
MODEL_NAME = "hf-hub:timm/mobilenetv3_large_100.ra_in1k"  # MobileNetV3 from the Hugging Face Hub

NUM_ITEMS = 300   # how many clothing items to test with; set to None to use the full test set
BATCH = 32        # how many images the model processes at once
random.seed(42)   # picks the same sample every run, so results can be compared

# Load the pretrained model. num_classes=0 removes the last layer, so it outputs fingerprints (embeddings).
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
model.eval()
# The exact resize and colour scaling this model was trained with.
data_config = timm.data.resolve_model_data_config(model)
preprocess = timm.data.create_transform(**data_config, is_training=False)


def load_partition():
    """Read which images are query photos and which are gallery photos, with their item IDs."""
    query, gallery = [], []
    for line in PARTITION.read_text().splitlines()[2:]:  # first two lines are headers
        parts = line.split()
        if len(parts) != 3:
            continue
        image_name, item_id, status = parts
        if status == "query":
            query.append((image_name, item_id))
        elif status == "gallery":
            gallery.append((image_name, item_id))
    return query, gallery


def fingerprints(image_names, label):
    """Turn a list of images into length-1 fingerprints, a batch at a time."""
    results = []
    start = time.time()
    for i in range(0, len(image_names), BATCH):
        batch = torch.stack([
            preprocess(Image.open(DATA / name).convert("RGB")) for name in image_names[i:i + BATCH]
        ])
        with torch.no_grad():
            vectors = model(batch)
        results.append(vectors / vectors.norm(dim=1, keepdim=True))
        print(f"\r  {label}: {min(i + BATCH, len(image_names))}/{len(image_names)} images", end="")
    seconds = time.time() - start
    print(f"  ({seconds:.0f}s, {1000 * seconds / len(image_names):.0f} ms per image)")
    return torch.cat(results)


query, gallery = load_partition()

# Only use items that have both query and gallery photos, then take a fixed random sample.
items = sorted({item for _, item in query} & {item for _, item in gallery})
if NUM_ITEMS is not None:
    items = random.sample(items, min(NUM_ITEMS, len(items)))
chosen = set(items)
query = [q for q in query if q[1] in chosen]
gallery = [g for g in gallery if g[1] in chosen]
print(f"Model: {MODEL_NAME}")
print(f"Testing {len(chosen)} items: {len(query)} query photos, {len(gallery)} gallery photos\n")

query_vectors = fingerprints([name for name, _ in query], "Query")
gallery_vectors = fingerprints([name for name, _ in gallery], "Gallery")

# Cosine similarity between every query and every gallery photo, then each query's top results.
scores = query_vectors @ gallery_vectors.T
gallery_items = [item for _, item in gallery]

for k in (1, 5):
    top = torch.topk(scores, k=k, dim=1).indices
    hits = sum(
        any(gallery_items[j] == query_item for j in top[row].tolist())
        for row, (_, query_item) in enumerate(query)
    )
    print(f"recall@{k}: {100 * hits / len(query):.1f}%  ({hits}/{len(query)} queries)")