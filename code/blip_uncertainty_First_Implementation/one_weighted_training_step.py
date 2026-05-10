import json
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_PATH = "mini_dataset_scored.json"
MLP_PATH = "confidence_mlp_warmup.pt"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
TEMPERATURE = 1.0

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

record = dataset[0]
image_path = record["image_path"]
captions = record["captions"]

print(f"Using image: {image_path}")

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
blip.train()

print("Loading trained confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
mlp.train()

# Two learning rates: smaller for BLIP, bigger for MLP
optimizer = torch.optim.Adam([
    {"params": blip.parameters(), "lr": 1e-5},
    {"params": mlp.parameters(), "lr": 1e-4},
])

image = Image.open(image_path).convert("RGB")

# Build feature tensor from precomputed normalized features
features = [cap["normalized_features"] for cap in captions]
x = torch.tensor(features, dtype=torch.float32, device=device)

# Keep copies of one parameter from each model to verify they change
mlp_param_before = mlp.net[0].weight.detach().clone()
blip_param_before = next(blip.parameters()).detach().clone()

optimizer.zero_grad()

# Forward through MLP -> scores -> weights
raw_scores = mlp(x)
weights = torch.softmax(raw_scores / TEMPERATURE, dim=0)

per_caption_losses = []
weighted_parts = []

for i, cap in enumerate(captions, start=1):
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
    weighted_i = weights[i - 1] * loss_i

    per_caption_losses.append(loss_i)
    weighted_parts.append(weighted_i)

weighted_loss = torch.stack(weighted_parts).sum()

print("\nBefore backward:")
for i, cap in enumerate(captions):
    print(f"{i+1}. {cap['text']}")
    print(f"   raw_score      = {raw_scores[i].item():.6f}")
    print(f"   softmax_weight = {weights[i].item():.6f}")
    print(f"   caption_loss   = {per_caption_losses[i].item():.6f}")
    print(f"   weighted_part  = {weighted_parts[i].item():.6f}")

print(f"\nFinal weighted loss = {weighted_loss.item():.6f}")

weighted_loss.backward()

# Report gradient norms
mlp_grad_norm = mlp.net[0].weight.grad.norm().item()
first_blip_grad = next(p for p in blip.parameters() if p.grad is not None)
blip_grad_norm = first_blip_grad.grad.norm().item() if hasattr(first_blip_grad, "grad") else first_blip_grad.norm().item()

print("\nGradient norms:")
print(f"MLP first layer grad norm  = {mlp_grad_norm:.6f}")
print(f"BLIP first grad norm       = {blip_grad_norm:.6f}")

optimizer.step()

mlp_param_after = mlp.net[0].weight.detach().clone()
blip_param_after = next(blip.parameters()).detach().clone()

mlp_change = (mlp_param_after - mlp_param_before).abs().mean().item()
blip_change = (blip_param_after - blip_param_before).abs().mean().item()

print("\nParameter update check:")
print(f"Mean abs change in MLP param  = {mlp_change:.10f}")
print(f"Mean abs change in BLIP param = {blip_change:.10f}")

print("\nOne weighted optimization step completed successfully.")
