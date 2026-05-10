import os
import json
from glob import glob
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

IMAGE_DIR = "images"
OUTPUT_PATH = "mini_dataset_raw.json"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
K = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*")))
print(f"Found {len(image_paths)} images")

results = []

for idx, image_path in enumerate(image_paths, start=1):
    print(f"\n[{idx}/{len(image_paths)}] Processing {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            max_new_tokens=30,
            num_return_sequences=K,
        )

    captions = [
        processor.decode(out, skip_special_tokens=True).strip()
        for out in outputs
    ]

    # remove duplicates while preserving order
    unique_captions = []
    seen = set()
    for c in captions:
        if c not in seen:
            unique_captions.append(c)
            seen.add(c)

    results.append({
        "image_id": idx,
        "image_path": image_path,
        "generated_captions": unique_captions
    })

    for j, c in enumerate(unique_captions, start=1):
        print(f"  {j}. {c}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved dataset to {OUTPUT_PATH}")
