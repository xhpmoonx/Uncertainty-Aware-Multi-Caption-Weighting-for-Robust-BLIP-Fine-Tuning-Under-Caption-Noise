import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"
IN_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123.json")
OUT_PATH = Path("coco_work/subsets/coco_train_500_generatedK3_seed123.json")

K = 3
MAX_IMAGES = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

with open(IN_PATH, "r") as f:
    data = json.load(f)

data = data[:MAX_IMAGES]
new_data = []

for idx, item in enumerate(data, start=1):
    image = Image.open(item["image_path"]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    captions = []
    for k in range(K):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            )
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        captions.append(text)

    new_data.append({
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "generated_captions": captions
    })

    print(f"[{idx}/{len(data)}] {Path(item['image_path']).name}")

with open(OUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print("Saved:", OUT_PATH)
