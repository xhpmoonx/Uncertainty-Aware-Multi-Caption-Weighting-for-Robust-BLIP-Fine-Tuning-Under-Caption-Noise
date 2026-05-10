import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForImageTextRetrieval,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config
# ----------------------------
IMAGE_PATH = "test.jpg"
OUTPUT_PATH = "warmup_demo_results.json"

ORIGINAL_CAPTION = "a ripe straw with green straw and strawberry"

DEGRADED_CAPTIONS = [
    "ripe straw green straw strawberry",        # word dropping
    "strawberry green with straw ripe and",     # word shuffling
    "a red car parked on the street"            # mismatched
]

ALIGN_MODEL = "Salesforce/blip-itm-base-coco"
FLUENCY_MODEL = "distilgpt2"
AGREE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MARGIN = 1.0
EPOCHS = 300
LR = 1e-3

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load models
# ----------------------------
print("Loading BLIP ITM model...")
align_processor = BlipProcessor.from_pretrained(ALIGN_MODEL)
align_model = BlipForImageTextRetrieval.from_pretrained(ALIGN_MODEL).to(device)
align_model.eval()

print("Loading fluency model...")
fluency_tokenizer = AutoTokenizer.from_pretrained(FLUENCY_MODEL)
fluency_model = AutoModelForCausalLM.from_pretrained(FLUENCY_MODEL).to(device)
fluency_model.eval()

if fluency_tokenizer.pad_token is None:
    fluency_tokenizer.pad_token = fluency_tokenizer.eos_token

print("Loading agreement model...")
agree_model = SentenceTransformer(AGREE_MODEL)

# ----------------------------
# Prepare captions
# ----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
captions = [ORIGINAL_CAPTION] + DEGRADED_CAPTIONS

print("\nCaptions used in warm-up:")
for i, c in enumerate(captions, start=1):
    label = "ORIGINAL" if i == 1 else "DEGRADED"
    print(f"{i}. [{label}] {c}")

# ----------------------------
# Agreement scores
# ----------------------------
embeddings = agree_model.encode(captions)
sim_matrix = cosine_similarity(embeddings)

agreement_scores = []
n = len(captions)
for i in range(n):
    others = [sim_matrix[i][j] for j in range(n) if j != i]
    score = sum(others) / len(others) if others else 0.0
    agreement_scores.append(float(score))

# ----------------------------
# Fluency scores
# ----------------------------
fluency_scores = []
for text in captions:
    inputs = fluency_tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = fluency_model(**inputs, labels=inputs["input_ids"])
    avg_nll = outputs.loss.item()
    fluency_scores.append(float(-avg_nll))  # higher is better

# ----------------------------
# Alignment scores
# ----------------------------
alignment_scores = []
for text in captions:
    inputs = align_processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = align_model(**inputs, use_itm_head=True)

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
        raise ValueError("Could not find ITM logits.")

    alignment_scores.append(float(score))

# ----------------------------
# Build raw feature vectors
# order = [alignment, fluency, agreement]
# ----------------------------
raw_features = []
for a, f, g in zip(alignment_scores, fluency_scores, agreement_scores):
    raw_features.append([a, f, g])

# z-score normalization per dimension
means = []
stds = []
for j in range(3):
    vals = [vec[j] for vec in raw_features]
    mean_j = sum(vals) / len(vals)
    var_j = sum((v - mean_j) ** 2 for v in vals) / len(vals)
    std_j = math.sqrt(var_j)
    if std_j == 0:
        std_j = 1.0
    means.append(mean_j)
    stds.append(std_j)

norm_features = []
for vec in raw_features:
    norm_vec = [(vec[j] - means[j]) / stds[j] for j in range(3)]
    norm_features.append(norm_vec)

x = torch.tensor(norm_features, dtype=torch.float32, device=device)

# ----------------------------
# Confidence MLP
# ----------------------------
class ConfidenceMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)

model = ConfidenceMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Before training
# ----------------------------
model.eval()
with torch.no_grad():
    before_scores = model(x).detach().cpu().tolist()

print("\nScores BEFORE training:")
for i, s in enumerate(before_scores, start=1):
    label = "ORIGINAL" if i == 1 else "DEGRADED"
    print(f"{i}. [{label}] score = {s:.6f}")

# ----------------------------
# Ranking warm-up
# original should score higher than each degraded caption
# ----------------------------
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    scores = model(x)

    s_pos = scores[0]
    losses = []
    for i in range(1, len(captions)):
        s_neg = scores[i]
        loss_i = torch.clamp(MARGIN - s_pos + s_neg, min=0.0)
        losses.append(loss_i)

    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - ranking loss: {loss.item():.6f}")

# ----------------------------
# After training
# ----------------------------
model.eval()
with torch.no_grad():
    after_scores = model(x).detach().cpu().tolist()

print("\nScores AFTER training:")
for i, s in enumerate(after_scores, start=1):
    label = "ORIGINAL" if i == 1 else "DEGRADED"
    print(f"{i}. [{label}] score = {s:.6f}")

results = {
    "image_path": IMAGE_PATH,
    "captions": [],
    "scores_before_training": before_scores,
    "scores_after_training": after_scores,
}

for i, text in enumerate(captions):
    results["captions"].append({
        "text": text,
        "type": "original" if i == 0 else "degraded",
        "alignment_score": alignment_scores[i],
        "fluency_score": fluency_scores[i],
        "agreement_score": agreement_scores[i],
        "raw_features": raw_features[i],
        "normalized_features": norm_features[i],
        "score_before_training": before_scores[i],
        "score_after_training": after_scores[i],
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
