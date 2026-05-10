import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

RUN_TAG = "10img_5ep_seed123"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
IMAGE_DIR = "images"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

results_dir = Path("results")

# load the learned log to get the exact image list
with open(results_dir / f"{RUN_TAG}_learned.json") as f:
    learned_log = json.load(f)

image_paths = learned_log["image_paths"]

processor = BlipProcessor.from_pretrained(MODEL_NAME)

def load_blip(path=None):
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    if path is not None:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model

print("Loading models...")
base_model = load_blip(None)
learned_model = load_blip(results_dir / f"{RUN_TAG}_learned_blip.pt")
uniform_model = load_blip(results_dir / f"{RUN_TAG}_uniform_blip.pt")
top1_model = load_blip(results_dir / f"{RUN_TAG}_top1_blip.pt")

models = {
    "base": base_model,
    "learned": learned_model,
    "uniform": uniform_model,
    "top1": top1_model,
}

rows = []

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    row = {"image_path": image_path}

    print(f"\nProcessing {image_path}")

    for name, model in models.items():
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                pixel_values=inputs["pixel_values"],
                max_new_tokens=30,
                num_beams=3,
            )

        caption = processor.decode(out[0], skip_special_tokens=True)
        row[name] = caption
        print(f"  {name}: {caption}")

    rows.append(row)

json_path = results_dir / f"{RUN_TAG}_caption_compare.json"
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

md_path = results_dir / f"{RUN_TAG}_caption_compare.md"
with open(md_path, "w") as f:
    f.write(f"# Caption Comparison: {RUN_TAG}\n\n")
    for row in rows:
        f.write(f"## {row['image_path']}\n\n")
        f.write(f"- **base:** {row['base']}\n")
        f.write(f"- **learned:** {row['learned']}\n")
        f.write(f"- **uniform:** {row['uniform']}\n")
        f.write(f"- **top1:** {row['top1']}\n\n")

print(f"\nSaved {json_path}")
print(f"Saved {md_path}")
