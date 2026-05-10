import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

if len(sys.argv) != 2:
    print("Usage: python eval_clipscore_from_predictions.py <predictions.json>")
    sys.exit(1)

PRED_PATH = Path(sys.argv[1])
OUT_JSON = PRED_PATH.with_name(PRED_PATH.stem + "_clipscore.json")
OUT_MD = PRED_PATH.with_name(PRED_PATH.stem + "_clipscore.md")

with open(PRED_PATH, "r") as f:
    data = json.load(f)

meta_keys = {"image_id", "image_path", "references"}
models = [k for k in data[0].keys() if k not in meta_keys]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

def clipscore_for_pair(image_path, text):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    cosine = torch.sum(image_features * text_features, dim=-1).item()

    # Standard CLIPScore-style scaling
    score = max(cosine, 0.0) * 2.5
    return score, cosine

results = {}

for model_key in models:
    scores = []
    cosines = []

    print(f"\nScoring model: {model_key}")
    for i, item in enumerate(data, start=1):
        score, cosine = clipscore_for_pair(item["image_path"], item[model_key])
        scores.append(score)
        cosines.append(cosine)

        if i <= 2:
            print(f"[sample {i}] {model_key}")
            print("  caption:", item[model_key])
            print("  cosine :", f"{cosine:.6f}")
            print("  score  :", f"{score:.6f}")

    results[model_key] = {
        "mean_clipscore": sum(scores) / len(scores),
        "mean_cosine": sum(cosines) / len(cosines),
        "n": len(scores),
    }

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

lines = [f"# CLIPScore Results for {PRED_PATH.name}", ""]
for model_key, vals in results.items():
    lines.append(f"## {model_key}")
    lines.append(f"- mean_clipscore: {vals['mean_clipscore']:.6f}")
    lines.append(f"- mean_cosine: {vals['mean_cosine']:.6f}")
    lines.append(f"- n: {vals['n']}")
    lines.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("\nSaved:", OUT_JSON)
print("Saved:", OUT_MD)
print("\nSummary:")
for model_key, vals in results.items():
    print(model_key)
    print("  mean_clipscore:", f"{vals['mean_clipscore']:.6f}")
    print("  mean_cosine   :", f"{vals['mean_cosine']:.6f}")
