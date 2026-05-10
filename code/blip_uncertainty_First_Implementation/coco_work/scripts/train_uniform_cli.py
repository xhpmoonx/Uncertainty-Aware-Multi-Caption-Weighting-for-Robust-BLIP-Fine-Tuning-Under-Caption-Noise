
import json
import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--run_tag", type=str, required=True)
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--max_images", type=int, default=500)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
parser.add_argument("--caption_lr", type=float, default=1e-5)
args = parser.parse_args()

RUN_TAG = args.run_tag
INPUT_PATH = args.input_path
MAX_IMAGES = args.max_images
EPOCHS = args.epochs
SEED = args.seed
MODEL_NAME = args.model_name
CAPTION_LR = args.caption_lr

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def sanitize_caption_text(text):
    text = " ".join(str(text).split()).strip()
    if not text:
        return "object"
    return text

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

print("Loading dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

dataset = dataset[:MAX_IMAGES]
print(f"Loaded {len(dataset)} items")

print("Loading BLIP caption model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=CAPTION_LR)

epoch_losses = []
label_counts = {}

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    running_loss = 0.0
    used_items = 0

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions, labels, corrupted = extract_captions(record)

        for lab in labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        image = Image.open(image_path).convert("RGB")

        optimizer.zero_grad()
        parts = []

        for caption_text in captions:
            inputs = processor(images=image, text=caption_text, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            labels_tensor = input_ids.clone()
            labels_tensor[labels_tensor == processor.tokenizer.pad_token_id] = -100

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor,
            )

            if torch.isfinite(outputs.loss):
                parts.append(outputs.loss)

        if len(parts) == 0:
            print(f"Skipping {Path(image_path).name}: no finite caption losses")
            continue

        loss = torch.stack(parts).mean()

        if not torch.isfinite(loss):
            print(f"Skipping {Path(image_path).name}: non-finite uniform loss")
            continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        used_items += 1

        print(f"[{idx}/{len(dataset)}] loss={loss.item():.6f} | {Path(image_path).name}")

    avg_loss = float("nan") if used_items == 0 else running_loss / used_items
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1} average loss = {avg_loss:.6f}")
    print(f"Used items this epoch = {used_items}/{len(dataset)}")

log_path = results_dir / f"{RUN_TAG}.json"
ckpt_path = results_dir / f"{RUN_TAG}_blip.pt"

with open(log_path, "w") as f:
    json.dump({
        "run_tag": RUN_TAG,
        "input_path": INPUT_PATH,
        "epochs": EPOCHS,
        "seed": SEED,
        "caption_lr": CAPTION_LR,
        "epoch_losses": epoch_losses,
        "label_counts": label_counts,
    }, f, indent=2)

torch.save(model.state_dict(), ckpt_path)

print(f"Saved {log_path}")
print(f"Saved {ckpt_path}")
