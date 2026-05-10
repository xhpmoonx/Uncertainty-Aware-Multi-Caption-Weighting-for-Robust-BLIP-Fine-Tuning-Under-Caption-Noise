from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import json

IMAGE_PATH = "test.jpg"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
OUTPUT_PATH = "captions_test.json"
K = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading processor and model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

print("Loading image...")
image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Generating {K} captions...")
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

result = {
    "image_path": IMAGE_PATH,
    "generated_captions": unique_captions
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print("\nGenerated captions:")
for i, c in enumerate(unique_captions, start=1):
    print(f"{i}. {c}")

print(f"\nSaved to {OUTPUT_PATH}")
