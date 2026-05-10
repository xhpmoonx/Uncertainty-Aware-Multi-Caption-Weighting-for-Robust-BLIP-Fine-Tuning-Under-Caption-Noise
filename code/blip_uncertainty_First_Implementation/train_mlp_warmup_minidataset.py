import json
import torch
import torch.nn as nn
import torch.optim as optim

INPUT_PATH = "warmup_pairs_with_features.json"
MODEL_OUT = "confidence_mlp_warmup.pt"
RESULTS_OUT = "warmup_training_results.json"

EPOCHS = 400
LR = 1e-3
MARGIN = 1.0

torch.manual_seed(42)

# ----------------------------
# Load dataset
# ----------------------------
print("Loading warm-up dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} training records")

# ----------------------------
# Model
# ----------------------------
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

model = ConfidenceMLP()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Helper: build one record tensor
# ----------------------------
def record_to_tensor(record):
    feats = [c["normalized_features"] for c in record["captions"]]
    return torch.tensor(feats, dtype=torch.float32)   # shape [4, 3]

# ----------------------------
# Evaluate before training
# ----------------------------
model.eval()
before_results = []

with torch.no_grad():
    for record in dataset:
        x = record_to_tensor(record)
        scores = model(x)
        score_list = scores.tolist()

        best_idx = int(torch.argmax(scores).item())
        best_type = record["captions"][best_idx]["type"]

        before_results.append({
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "best_type_before": best_type,
            "scores_before": {
                c["type"]: float(score_list[i])
                for i, c in enumerate(record["captions"])
            }
        })

num_correct_before = sum(1 for r in before_results if r["best_type_before"] == "original")
print(f"Before training: original is top-scoring for {num_correct_before}/{len(dataset)} images")

# ----------------------------
# Training loop
# ----------------------------
print("\nTraining confidence MLP...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for record in dataset:
        x = record_to_tensor(record)

        optimizer.zero_grad()
        scores = model(x)

        s_pos = scores[0]  # original caption
        losses = []

        for i in range(1, scores.shape[0]):
            s_neg = scores[i]
            loss_i = torch.clamp(MARGIN - s_pos + s_neg, min=0.0)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataset)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - avg ranking loss: {avg_loss:.6f}")

# ----------------------------
# Evaluate after training
# ----------------------------
model.eval()
after_results = []

with torch.no_grad():
    for record in dataset:
        x = record_to_tensor(record)
        scores = model(x)
        score_list = scores.tolist()

        best_idx = int(torch.argmax(scores).item())
        best_type = record["captions"][best_idx]["type"]

        after_results.append({
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "best_type_after": best_type,
            "scores_after": {
                c["type"]: float(score_list[i])
                for i, c in enumerate(record["captions"])
            }
        })

num_correct_after = sum(1 for r in after_results if r["best_type_after"] == "original")
print(f"\nAfter training: original is top-scoring for {num_correct_after}/{len(dataset)} images")

# ----------------------------
# Merge detailed results
# ----------------------------
results = []

for record, before_r, after_r in zip(dataset, before_results, after_results):
    merged = {
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "captions": [],
        "best_type_before": before_r["best_type_before"],
        "best_type_after": after_r["best_type_after"],
    }

    for c in record["captions"]:
        ctype = c["type"]
        merged["captions"].append({
            "type": ctype,
            "text": c["text"],
            "normalized_features": c["normalized_features"],
            "score_before": before_r["scores_before"][ctype],
            "score_after": after_r["scores_after"][ctype],
        })

    results.append(merged)

# ----------------------------
# Save outputs
# ----------------------------
torch.save(model.state_dict(), MODEL_OUT)

with open(RESULTS_OUT, "w") as f:
    json.dump({
        "num_records": len(dataset),
        "num_correct_before": num_correct_before,
        "num_correct_after": num_correct_after,
        "results": results
    }, f, indent=2)

print(f"\nSaved trained model to {MODEL_OUT}")
print(f"Saved detailed results to {RESULTS_OUT}")
