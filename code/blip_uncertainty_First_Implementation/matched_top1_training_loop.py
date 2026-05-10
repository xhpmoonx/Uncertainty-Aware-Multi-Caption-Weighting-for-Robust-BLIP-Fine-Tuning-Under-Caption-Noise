import json
import random
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_PATH = "mini_dataset_scored.json"
MLP_PATH = "confidence_mlp_warmup.pt"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_IMAGES = 10
EPOCHS = 5
SEED = 123
TEMPERATURE = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Mode: top1_hard_selection")

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

print("Loading trained confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
mlp.eval()   # fixed selector for this baseline

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

        # choose top-1 caption using learned MLP score
        features = [cap["normalized_features"] for cap in captions]
        x = torch.tensor(features, dtype=torch.float32, device=device)

        with torch.no_grad():
            raw_scores = mlp(x)
            top_idx = int(torch.argmax(raw_scores).item())

        chosen_caption = captions[top_idx]["text"]

        optimizer.zero_grad()

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

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"[{idx}/{len(dataset)}] {image_path}")
        print(f"  chosen_idx = {top_idx}")
        print(f"  chosen_caption = {chosen_caption}")
        print(f"  loss = {loss.item():.6f}")
        for j, cap in enumerate(captions):
            mark = " <-- chosen" if j == top_idx else ""
            print(f"    {j+1}. raw_score={raw_scores[j].item():.4f}{mark}")

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"\nEpoch {epoch+1} average loss = {avg_loss:.6f}")

out_path = "matched_top1_log.json"
with open(out_path, "w") as f:
    json.dump({
        "weight_mode": "top1",
        "max_images": MAX_IMAGES,
        "epochs": EPOCHS,
        "seed": SEED,
        "image_paths": [r["image_path"] for r in dataset],
        "epoch_losses": epoch_losses
    }, f, indent=2)

print(f"\nSaved {out_path}")
