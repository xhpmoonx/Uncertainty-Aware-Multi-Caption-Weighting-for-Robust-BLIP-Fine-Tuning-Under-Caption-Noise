import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--stats_path", type=str, required=True)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--max_images", type=int, default=500)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "Salesforce/blip-itm-base-coco"


def sanitize_caption_text(text):
    text = " ".join(str(text).split()).strip()
    return text if text else "object"


def extract_captions(record):
    caps = record["generated_captions"]
    if len(caps) == 0:
        return ["object"], ["unknown"], [False]

    if isinstance(caps[0], str):
        texts = [sanitize_caption_text(c) for c in caps]
        labels = ["unknown"] * len(texts)
        corrupted = [False] * len(texts)
        return texts, labels, corrupted

    texts = [sanitize_caption_text(c.get("text", "")) for c in caps]
    labels = [c.get("label", "unknown") for c in caps]
    corrupted = [bool(c.get("is_corrupted", False)) for c in caps]
    return texts, labels, corrupted


print("Loading ITM filter...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME).to(device)
model.eval()


def compute_itm_scores(image, captions):
    scores = []
    for text in captions:
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, use_itm_head=True)

        if hasattr(outputs, "itm_score") and outputs.itm_score is not None:
            logits = outputs.itm_score
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
            if logits.shape[-1] == 2:
                score = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                score = logits.squeeze().item()
        else:
            score = 0.0

        scores.append(float(score))
    return scores


print("Loading input dataset...")
with open(args.input_path, "r") as f:
    dataset = json.load(f)

dataset = dataset[:args.max_images]
print(f"Loaded {len(dataset)} source images")

expanded = []
fallback_count = 0
kept_clean = 0
kept_corrupted = 0
retained_counts = []

for idx, record in enumerate(dataset, start=1):
    image_path = record["image_path"]
    image = Image.open(image_path).convert("RGB")

    captions, labels, corrupted_flags = extract_captions(record)
    itm_scores = compute_itm_scores(image, captions)

    keep_idx = [i for i, s in enumerate(itm_scores) if s >= args.threshold]
    if len(keep_idx) == 0:
        best_idx = max(range(len(itm_scores)), key=lambda i: itm_scores[i])
        keep_idx = [best_idx]
        fallback_count += 1
        selected_by = "fallback_top1"
    else:
        selected_by = "itm_pass"

    retained_counts.append(len(keep_idx))

    for i in keep_idx:
        row = {
            "image_id": record.get("image_id"),
            "image_path": image_path,
            "generated_captions": [
                {
                    "text": captions[i],
                    "label": labels[i],
                    "is_corrupted": bool(corrupted_flags[i]),
                    "itm_score": float(itm_scores[i]),
                    "selected_by": selected_by,
                }
            ],
        }
        expanded.append(row)

        if labels[i] == "clean":
            kept_clean += 1
        else:
            kept_corrupted += 1

    print(f"[{idx}/{len(dataset)}] kept={len(keep_idx)} | {Path(image_path).name}")

stats = {
    "input_path": args.input_path,
    "output_path": args.output_path,
    "threshold": args.threshold,
    "source_images": len(dataset),
    "expanded_pairs": len(expanded),
    "fallback_count": fallback_count,
    "kept_clean": kept_clean,
    "kept_corrupted": kept_corrupted,
    "mean_retained_per_image": (sum(retained_counts) / len(retained_counts)) if retained_counts else 0.0,
}

with open(args.output_path, "w") as f:
    json.dump(expanded, f, indent=2)

with open(args.stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print("Saved filtered dataset:", args.output_path)
print("Saved stats:", args.stats_path)
print(stats)

