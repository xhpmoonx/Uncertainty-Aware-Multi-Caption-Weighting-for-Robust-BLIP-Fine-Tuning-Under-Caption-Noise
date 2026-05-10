import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
RUN_TAG = "dynamic_joint_noisy60_seed123"
INPUT_PATH = "coco_work/subsets/coco_train_500_raw_local_seed123_noisy60.json"

MODEL_NAME = "Salesforce/blip-image-captioning-base"
ALIGN_MODEL = "Salesforce/blip-itm-base-coco"
FLUENCY_MODEL = "distilgpt2"
AGREE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MLP_INIT_PATH = "confidence_mlp_warmup.pt"

MAX_IMAGES = 500
EPOCHS = 10
SEED = 123

TAU_START = 2.0
TAU_END = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# REPRO
# =========================
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# =========================
# MODELS
# =========================
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

    def forward(self, x):
        return self.net(x).squeeze(-1)

print("Loading dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

dataset = dataset[:MAX_IMAGES]
print(f"Loaded {len(dataset)} items")

print("Loading captioning BLIP...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
caption_model.train()

print("Loading alignment model...")
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

print("Loading confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(MLP_INIT_PATH, map_location=device))
mlp.train()

optimizer = torch.optim.Adam([
    {"params": caption_model.parameters(), "lr": 1e-5},
    {"params": mlp.parameters(), "lr": 1e-4},
])

# =========================
# HELPERS
# =========================
def zscore_features(raw_features):
    # raw_features: list of [a, f, g]
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

    norm = []
    for vec in raw_features:
        norm.append([(vec[j] - means[j]) / stds[j] for j in range(3)])
    return norm

def get_tau(epoch_idx, total_epochs):
    if total_epochs == 1:
        return TAU_END
    alpha = epoch_idx / (total_epochs - 1)
    return TAU_START + alpha * (TAU_END - TAU_START)

def compute_alignment_scores(image, captions):
    scores = []
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
            raise ValueError("Could not find ITM logits in model output.")

        scores.append(float(score))
    return scores

def compute_fluency_scores(captions):
    scores = []
    for text in captions:
        inputs = fluency_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = fluency_model(**inputs, labels=inputs["input_ids"])
        avg_nll = outputs.loss.item()
        scores.append(float(-avg_nll))
    return scores

def compute_agreement_scores(captions):
    embeddings = agree_model.encode(captions)
    sim_matrix = cosine_similarity(embeddings)
    out = []
    n = len(captions)
    for i in range(n):
        others = [sim_matrix[i][j] for j in range(n) if j != i]
        score = sum(others) / len(others) if others else 0.0
        out.append(float(score))
    return out

# =========================
# TRAIN
# =========================
epoch_losses = []

for epoch in range(EPOCHS):
    tau = get_tau(epoch, EPOCHS)
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} | tau={tau:.4f} =====")

    running_loss = 0.0

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions = record["generated_captions"]

        image = Image.open(image_path).convert("RGB")

        # dynamic / current signals
        alignment_scores = compute_alignment_scores(image, captions)
        fluency_scores = compute_fluency_scores(captions)
        agreement_scores = compute_agreement_scores(captions)

        raw_features = []
        for a, f, g in zip(alignment_scores, fluency_scores, agreement_scores):
            raw_features.append([a, f, g])

        norm_features = zscore_features(raw_features)
        x = torch.tensor(norm_features, dtype=torch.float32, device=device)

        raw_scores = mlp(x)
        weights = torch.softmax(raw_scores / tau, dim=0)

        optimizer.zero_grad()
        weighted_parts = []

        for i, text in enumerate(captions):
            inputs = processor(images=image, text=text, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = caption_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            weighted_parts.append(weights[i] * outputs.loss)

        total_loss = torch.stack(weighted_parts).sum()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        print(f"[{idx}/{len(dataset)}] total_loss={total_loss.item():.6f} | {Path(image_path).name}")

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1} average loss = {avg_loss:.6f}")

# =========================
# SAVE
# =========================
log_path = results_dir / f"{RUN_TAG}.json"
with open(log_path, "w") as f:
    json.dump({
        "run_tag": RUN_TAG,
        "input_path": INPUT_PATH,
        "epochs": EPOCHS,
        "seed": SEED,
        "tau_start": TAU_START,
        "tau_end": TAU_END,
        "epoch_losses": epoch_losses,
    }, f, indent=2)

blip_ckpt = results_dir / f"{RUN_TAG}_blip.pt"
mlp_ckpt = results_dir / f"{RUN_TAG}_mlp.pt"

torch.save(caption_model.state_dict(), blip_ckpt)
torch.save(mlp.state_dict(), mlp_ckpt)

print(f"Saved {log_path}")
print(f"Saved {blip_ckpt}")
print(f"Saved {mlp_ckpt}")
