import json
import random
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# CHANGE ONLY THIS LINE
# =========================
WEIGHT_MODE = "uniform"   # "learned" or "uniform"

INPUT_PATH = "mini_dataset_scored.json"
MLP_PATH = "confidence_mlp_warmup.pt"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

MAX_IMAGES = 10
EPOCHS = 5
TEMPERATURE = 1.0
SEED = 123

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Weight mode:", WEIGHT_MODE)

# -------------------------
# Reproducibility
# -------------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Confidence MLP
# -------------------------
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

# -------------------------
# Load dataset
# -------------------------
print("Loading scored dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

dataset = sorted(dataset, key=lambda r: r["image_path"])[:MAX_IMAGES]
print(f"Training on {len(dataset)} images:")
for r in dataset:
    print(" ", r["image_path"])

# -------------------------
# Load BLIP fresh each run
# -------------------------
print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
blip.train()

# -------------------------
# Weighting setup
# -------------------------
mlp = None
if WEIGHT_MODE == "learned":
    print("Loading trained confidence MLP...")
    mlp = ConfidenceMLP().to(device)
    mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
    mlp.train()

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

        # -------------------------
        # Weights
        # -------------------------
        if WEIGHT_MODE == "learned":
            features = [cap["normalized_features"] for cap in captions]
            x = torch.tensor(features, dtype=torch.float32, device=device)
            raw_scores = mlp(x)
            weights = torch.softmax(raw_scores / TEMPERATURE, dim=0)
        else:
            k = len(captions)
            weights = torch.full((k,), 1.0 / k, dtype=torch.float32, device=device)
            raw_scores = None

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

        running_loss += total_loss.item()

        print(f"[{idx}/{len(dataset)}] {image_path}")
        print(f"  total_loss = {total_loss.item():.6f}")
        for j in range(len(captions)):
            line = f"    {j+1}. weight={weights[j].item():.4f} loss={per_caption_losses[j]:.4f}"
            if raw_scores is not None:
                line += f" raw_score={raw_scores[j].item():.4f}"
            print(line)

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"\nEpoch {epoch+1} average loss = {avg_loss:.6f}")

# -------------------------
# Save log
# -------------------------
out_path = f"matched_{WEIGHT_MODE}_log.json"
with open(out_path, "w") as f:
    json.dump({
        "weight_mode": WEIGHT_MODE,
        "max_images": MAX_IMAGES,
        "epochs": EPOCHS,
        "seed": SEED,
        "image_paths": [r["image_path"] for r in dataset],
        "epoch_losses": epoch_losses
    }, f, indent=2)

print(f"\nSaved {out_path}")
