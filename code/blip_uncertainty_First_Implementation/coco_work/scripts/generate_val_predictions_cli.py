import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--run_tag", type=str, required=True)
parser.add_argument("--val_path", type=str, default="coco_work/subsets/coco_val_100_eval_local_seed123.json")
parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
RUN_TAG = args.run_tag
VAL_PATH = args.val_path
MODEL_NAME = args.model_name

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

with open(VAL_PATH, "r") as f:
    data = json.load(f)

predictions = []

for idx, item in enumerate(data, start=1):
    image = Image.open(item["image_path"]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3,
        )

    pred = processor.decode(out[0], skip_special_tokens=True).strip()

    refs = item.get("references", item.get("captions"))
    if refs is None:
        raise KeyError("No 'references' or 'captions' field found in val file.")

    predictions.append({
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "references": refs,
        RUN_TAG: pred,
    })

    print(f"[{idx}/{len(data)}] {Path(item['image_path']).name}")

out_path = Path(f"results/{RUN_TAG}_predictions.json")
with open(out_path, "w") as f:
    json.dump(predictions, f, indent=2)

print("Saved:", out_path)
