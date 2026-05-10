import json
import torch
import torch.nn as nn

INPUT_PATH = "captions_with_features.json"
OUTPUT_PATH = "captions_with_scores.json"
TEMPERATURE = 1.0

torch.manual_seed(42)

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
        return self.net(x)

print("Loading data...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

items = data["captions_with_agreement"]

features = [item["normalized_features"] for item in items]
x = torch.tensor(features, dtype=torch.float32)   # shape: [K, 3]

print("Input feature tensor shape:", x.shape)

model = ConfidenceMLP()
model.eval()

with torch.no_grad():
    raw_scores = model(x).squeeze(-1)   # shape: [K]
    weights = torch.softmax(raw_scores / TEMPERATURE, dim=0)

print("\nRaw scores and softmax weights:")
for i, item in enumerate(items):
    item["raw_score"] = float(raw_scores[i].item())
    item["softmax_weight"] = float(weights[i].item())

    print(f"{i+1}. {item['text']}")
    print(f"   normalized_features = {item['normalized_features']}")
    print(f"   raw_score = {item['raw_score']:.6f}")
    print(f"   softmax_weight = {item['softmax_weight']:.6f}")

print(f"\nSum of weights = {weights.sum().item():.6f}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
