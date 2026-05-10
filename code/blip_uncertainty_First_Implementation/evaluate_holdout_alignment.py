import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

RUN_TAG = "200img_5ep_seed42"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_NAME = "openai/clip-vit-base-patch32"
HOLDOUT_LIST = "holdout_images.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

results_dir = Path("results")


# ---------- load holdout image paths ----------
with open(HOLDOUT_LIST, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

print("Holdout images:")
for p in image_paths:
    print(" ", p)

# ---------- BLIP loader ----------
blip_processor = BlipProcessor.from_pretrained(MODEL_NAME)

def load_blip(checkpoint_path=None):
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model

print("Loading BLIP models...")
models = {
    "base": load_blip(None),
    "learned": load_blip(results_dir / f"{RUN_TAG}_learned_blip.pt"),
    "uniform": load_blip(results_dir / f"{RUN_TAG}_uniform_blip.pt"),
    "top1": load_blip(results_dir / f"{RUN_TAG}_top1_blip.pt"),
}

# ---------- CLIP alignment model ----------
print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
clip_model.eval()

def generate_caption(model, image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=inputs["pixel_values"],
            max_new_tokens=30,
            num_beams=3,
        )
    return blip_processor.decode(out[0], skip_special_tokens=True)

def clip_alignment_score(image, text):
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    score = (image_features * text_features).sum(dim=-1).item()
    return score

rows = []
avg_scores = {k: [] for k in models.keys()}

for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    row = {"image_path": image_path}

    print(f"\nProcessing {image_path}")

    for name, model in models.items():
        caption = generate_caption(model, image)
        score = clip_alignment_score(image, caption)

        row[f"{name}_caption"] = caption
        row[f"{name}_alignment"] = score
        avg_scores[name].append(score)

        print(f"  {name}: {caption}")
        print(f"    alignment={score:.6f}")

    rows.append(row)

summary = {
    "run_tag": RUN_TAG,
    "holdout_images": image_paths,
    "average_alignment": {
        name: sum(vals) / len(vals) if vals else None
        for name, vals in avg_scores.items()
    }
}

json_path = results_dir / f"{RUN_TAG}_holdout_alignment.json"
with open(json_path, "w") as f:
    json.dump({"rows": rows, "summary": summary}, f, indent=2)

md_path = results_dir / f"{RUN_TAG}_holdout_alignment.md"
with open(md_path, "w") as f:
    f.write(f"# Holdout Alignment Evaluation: {RUN_TAG}\n\n")
    f.write("## Average alignment\n\n")
    for name, score in summary["average_alignment"].items():
        f.write(f"- **{name}:** {score:.6f}\n")
    f.write("\n")
    for row in rows:
        f.write(f"## {row['image_path']}\n\n")
        for name in ["base", "learned", "uniform", "top1"]:
            f.write(f"- **{name}:** {row[f'{name}_caption']}  \n")
            f.write(f"  alignment = {row[f'{name}_alignment']:.6f}\n")
        f.write("\n")

print(f"\nSaved {json_path}")
print(f"Saved {md_path}")
print("\nAverage alignment:")
for name, score in summary["average_alignment"].items():
    print(f"  {name}: {score:.6f}")
