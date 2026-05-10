import json
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_PATH = "mini_dataset_scored.json"
MLP_PATH = "confidence_mlp_warmup.pt"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
TEMPERATURE = 1.0

# start small
MAX_IMAGES = 5
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

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

dataset = dataset[:MAX_IMAGES]
print(f"Training on {len(dataset)} images")

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
blip.train()

print("Loading trained confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
mlp.train()

optimizer = torch.optim.Adam([
    {"params": blip.parameters(), "lr": 1e-5},
    {"params": mlp.parameters(), "lr": 1e-4},
])

epoch_losses = []

for epoch in range(EPOCHS):
    running_loss = 0.0

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions = record["captions"]

        image = Image.open(image_path).convert("RGB")

        # precomputed normalized features -> MLP
        features = [cap["normalized_features"] for cap in captions]
        x = torch.tensor(features, dtype=torch.float32, device=device)

        optimizer.zero_grad()

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

        weighted_loss = torch.stack(weighted_parts).sum()
        weighted_loss.backward()
        optimizer.step()

        running_loss += weighted_loss.item()

        print(f"[{idx}/{len(dataset)}] {image_path}")
        print(f"  weighted_loss = {weighted_loss.item():.6f}")
        for j, cap in enumerate(captions, start=1):
            print(f"    {j}. weight={weights[j-1].item():.4f} loss={per_caption_losses[j-1]:.4f}")

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"\nEpoch {epoch+1} average loss = {avg_loss:.6f}")

    torch.save(mlp.state_dict(), f"mlp_epoch_{epoch+1}.pt")
    torch.save(blip.state_dict(), f"blip_epoch_{epoch+1}.pt")

with open("tiny_training_log.json", "w") as f:
    json.dump({"epoch_losses": epoch_losses}, f, indent=2)

print("\nTraining finished.")
print("Saved:")
print("  tiny_training_log.json")
for epoch in range(EPOCHS):
    print(f"  mlp_epoch_{epoch+1}.pt")
    print(f"  blip_epoch_{epoch+1}.pt")
