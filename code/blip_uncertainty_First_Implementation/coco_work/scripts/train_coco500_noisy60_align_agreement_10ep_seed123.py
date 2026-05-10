import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================================
# CHANGE THIS FOR EACH RUN
# =========================================
WEIGHT_MODE = "learned"
RUN_TAG = "coco500_noisy60_align_agreement_10ep_seed123"

INPUT_PATH = "coco_work/subsets/coco_train_500_scored_seed123_noisy60_align_agreement.json"
MLP_PATH = "confidence_mlp_warmup.pt"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_IMAGES = 500
EPOCHS = 10
SEED = 123
TEMPERATURE = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Weight mode:", WEIGHT_MODE)
print("Run tag:", RUN_TAG)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class ConfidenceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

print("Loading scored dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

dataset = sorted(dataset, key=lambda r: r["image_path"])[:MAX_IMAGES]
print(f"Training on {len(dataset)} images:")
for r in dataset:
    print(" ", r["image_path"])

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
blip.train()

mlp = None
if WEIGHT_MODE in ["learned", "top1"]:
    print("Loading trained confidence MLP...")
    mlp = ConfidenceMLP().to(device)
    mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
    if WEIGHT_MODE == "learned":
        mlp.train()
    else:
        mlp.eval()

if WEIGHT_MODE == "learned":
    optimizer = torch.optim.Adam([
        {"params": blip.parameters(), "lr": 1e-5},
        {"params": mlp.parameters(), "lr": 1e-4},
    ])
else:
    optimizer = torch.optim.Adam([
        {"params": blip.parameters(), "lr": 1e-5},
    ])

epoch_losses = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions = record["captions"]
        image = Image.open(image_path).convert("RGB")

        optimizer.zero_grad()

        if WEIGHT_MODE == "learned":
            features = [cap["normalized_features"] for cap in captions]
            x = torch.tensor(features, dtype=torch.float32, device=device)
            raw_scores = mlp(x)
            weights = torch.softmax(raw_scores / TEMPERATURE, dim=0)

            weighted_parts = []
            per_caption_losses = []

            for i, cap in enumerate(captions):
                text = cap["text"]
                inputs = processor(images=image, text=text, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                outputs = blip(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss_i = outputs.loss
                per_caption_losses.append(loss_i.item())
                weighted_parts.append(weights[i] * loss_i)

            total_loss = torch.stack(weighted_parts).sum()
            total_loss.backward()
            optimizer.step()

        elif WEIGHT_MODE == "uniform":
            k = len(captions)
            weights = torch.full((k,), 1.0 / k, dtype=torch.float32, device=device)

            weighted_parts = []
            per_caption_losses = []

            for i, cap in enumerate(captions):
                text = cap["text"]
                inputs = processor(images=image, text=text, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                outputs = blip(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss_i = outputs.loss
                per_caption_losses.append(loss_i.item())
                weighted_parts.append(weights[i] * loss_i)

            total_loss = torch.stack(weighted_parts).sum()
            total_loss.backward()
            optimizer.step()

        elif WEIGHT_MODE == "top1":
            features = [cap["normalized_features"] for cap in captions]
            x = torch.tensor(features, dtype=torch.float32, device=device)

            with torch.no_grad():
                raw_scores = mlp(x)
                top_idx = int(torch.argmax(raw_scores).item())

            chosen_caption = captions[top_idx]["text"]

            inputs = processor(images=image, text=chosen_caption, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = blip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss = outputs.loss
            total_loss.backward()
            optimizer.step()

        else:
            raise ValueError(f"Unknown WEIGHT_MODE: {WEIGHT_MODE}")

        running_loss += total_loss.item()
        print(f"[{idx}/{len(dataset)}] {image_path} total_loss={total_loss.item():.6f}")

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1} average loss = {avg_loss:.6f}")

# save log
log_path = results_dir / f"{RUN_TAG}_{WEIGHT_MODE}.json"
with open(log_path, "w") as f:
    json.dump({
        "weight_mode": WEIGHT_MODE,
        "max_images": MAX_IMAGES,
        "epochs": EPOCHS,
        "seed": SEED,
        "image_paths": [r["image_path"] for r in dataset],
        "epoch_losses": epoch_losses
    }, f, indent=2)

# save final model checkpoint
blip_ckpt = results_dir / f"{RUN_TAG}_{WEIGHT_MODE}_blip.pt"
torch.save(blip.state_dict(), blip_ckpt)

if WEIGHT_MODE == "learned":
    mlp_ckpt = results_dir / f"{RUN_TAG}_{WEIGHT_MODE}_mlp.pt"
    torch.save(mlp.state_dict(), mlp_ckpt)
    print(f"Saved {mlp_ckpt}")

print(f"Saved {log_path}")
print(f"Saved {blip_ckpt}")
