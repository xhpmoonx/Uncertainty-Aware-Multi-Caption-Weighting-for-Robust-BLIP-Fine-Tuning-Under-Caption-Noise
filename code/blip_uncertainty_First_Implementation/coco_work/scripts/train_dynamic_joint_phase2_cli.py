import json
import math
import argparse
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


parser = argparse.ArgumentParser()
parser.add_argument("--run_tag", type=str, required=True)
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--max_images", type=int, default=500)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=123)

parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
parser.add_argument("--align_model_name", type=str, default="Salesforce/blip-itm-base-coco")
parser.add_argument("--fluency_model_name", type=str, default="distilgpt2")
parser.add_argument("--agreement_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

parser.add_argument("--caption_lr", type=float, default=1e-5)
parser.add_argument("--mlp_lr", type=float, default=1e-3)
parser.add_argument("--tau_start", type=float, default=2.0)
parser.add_argument("--tau_end", type=float, default=0.5)

parser.add_argument("--use_alignment", type=int, default=1)
parser.add_argument("--use_fluency", type=int, default=1)
parser.add_argument("--use_agreement", type=int, default=1)

args = parser.parse_args()

RUN_TAG = args.run_tag
INPUT_PATH = args.input_path
MAX_IMAGES = args.max_images
EPOCHS = args.epochs
SEED = args.seed

MODEL_NAME = args.model_name
ALIGN_MODEL_NAME = args.align_model_name
FLUENCY_MODEL_NAME = args.fluency_model_name
AGREEMENT_MODEL_NAME = args.agreement_model_name

CAPTION_LR = args.caption_lr
MLP_LR = args.mlp_lr
TAU_START = args.tau_start
TAU_END = args.tau_end

USE_ALIGNMENT = bool(args.use_alignment)
USE_FLUENCY = bool(args.use_fluency)
USE_AGREEMENT = bool(args.use_agreement)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


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


def safe_float(x, fallback=0.0):
    try:
        x = float(x)
    except Exception:
        return fallback
    if math.isnan(x) or math.isinf(x):
        return fallback
    return x


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


def zscore_features(raw_features):
    means = []
    stds = []

    for j in range(3):
        vals = [safe_float(vec[j], 0.0) for vec in raw_features]
        mean_j = sum(vals) / len(vals)
        var_j = sum((v - mean_j) ** 2 for v in vals) / len(vals)
        std_j = math.sqrt(var_j)
        if std_j == 0 or math.isnan(std_j) or math.isinf(std_j):
            std_j = 1.0
        means.append(mean_j)
        stds.append(std_j)

    norm = []
    for vec in raw_features:
        out = []
        for j in range(3):
            v = safe_float(vec[j], 0.0)
            z = (v - means[j]) / stds[j]
            out.append(safe_float(z, 0.0))
        norm.append(out)
    return norm


def compute_tau(epoch_idx, total_epochs, tau_start, tau_end):
    if total_epochs <= 1:
        return tau_end
    alpha = epoch_idx / float(total_epochs - 1)
    return tau_start + alpha * (tau_end - tau_start)


print("Loading dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)
dataset = dataset[:MAX_IMAGES]
print(f"Loaded {len(dataset)} items")

print("Loading captioning BLIP...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.train()

print("Loading alignment model...")
align_processor = BlipProcessor.from_pretrained(ALIGN_MODEL_NAME)
align_model = BlipForImageTextRetrieval.from_pretrained(ALIGN_MODEL_NAME).to(device)
align_model.eval()

print("Loading fluency model...")
fluency_tokenizer = AutoTokenizer.from_pretrained(FLUENCY_MODEL_NAME)
fluency_model = AutoModelForCausalLM.from_pretrained(FLUENCY_MODEL_NAME).to(device)
fluency_model.eval()
if fluency_tokenizer.pad_token is None:
    fluency_tokenizer.pad_token = fluency_tokenizer.eos_token

print("Loading agreement model...")
agree_model = SentenceTransformer(AGREEMENT_MODEL_NAME, device=device)

print("Loading confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.train()

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(mlp.parameters()),
    lr=CAPTION_LR,
)
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=MLP_LR)


def compute_alignment_scores(image, captions):
    scores = []
    for text in captions:
        text = sanitize_caption_text(text)
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
            score = 0.0

        scores.append(safe_float(score, 0.0))
    return scores


def compute_fluency_scores(captions):
    scores = []
    for text in captions:
        text = sanitize_caption_text(text)
        inputs = fluency_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = fluency_model(**inputs, labels=inputs["input_ids"])

        score = -outputs.loss.item()
        scores.append(safe_float(score, 0.0))
    return scores


def compute_agreement_scores(captions):
    captions = [sanitize_caption_text(c) for c in captions]
    embeddings = agree_model.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    sim_matrix = cosine_similarity(embeddings)

    out = []
    n = len(captions)
    for i in range(n):
        others = [sim_matrix[i][j] for j in range(n) if j != i]
        score = sum(others) / len(others) if others else 0.0
        out.append(safe_float(score, 0.0))
    return out


def compute_caption_loss(image, caption_text):
    inputs = processor(images=image, text=caption_text, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    labels_tensor = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels_tensor[labels_tensor == pad_id] = -100

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_tensor,
    )
    return outputs.loss


epoch_losses = []
label_counts = {}

skipped_nonfinite_raw_scores = 0
skipped_nonfinite_weights = 0
skipped_no_finite_caption_losses = 0
skipped_nonfinite_total_loss = 0

for epoch in range(EPOCHS):
    tau = compute_tau(epoch, EPOCHS, TAU_START, TAU_END)
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} | tau={tau:.4f} =====")

    running_loss = 0.0
    used_items = 0

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions, labels, corrupted = extract_captions(record)

        for lab in labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        image = Image.open(image_path).convert("RGB")

        alignment_scores = compute_alignment_scores(image, captions)
        fluency_scores = compute_fluency_scores(captions)
        agreement_scores = compute_agreement_scores(captions)

        raw_features = []
        for a, f, g in zip(alignment_scores, fluency_scores, agreement_scores):
            raw_features.append([a, f, g])

        norm_features = zscore_features(raw_features)

        for vec in norm_features:
            if not USE_ALIGNMENT:
                vec[0] = 0.0
            if not USE_FLUENCY:
                vec[1] = 0.0
            if not USE_AGREEMENT:
                vec[2] = 0.0

        x = torch.tensor(norm_features, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        mlp_optimizer.zero_grad()

        raw_scores = mlp(x)
        if not torch.all(torch.isfinite(raw_scores)):
            skipped_nonfinite_raw_scores += 1
            print(f"Skipping {Path(image_path).name}: non-finite raw_scores")
            image.close()
            continue

        weights = torch.softmax(raw_scores / tau, dim=0)
        threshold = (1.0 / len(captions)) * 0.5
        weights = weights * (weights > threshold).float()
        if weights.sum().item() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)

        if not torch.all(torch.isfinite(weights)):
            skipped_nonfinite_weights += 1
            print(f"Skipping {Path(image_path).name}: non-finite weights")
            image.close()
            continue

        finite_losses = []
        finite_weights = []

        for i, caption_text in enumerate(captions):
            loss_i = compute_caption_loss(image, caption_text)
            if torch.isfinite(loss_i):
                finite_losses.append(loss_i)
                finite_weights.append(weights[i])
            else:
                print(f"  caption[{i}] non-finite loss -> dropped")

        if len(finite_losses) == 0:
            skipped_no_finite_caption_losses += 1
            print(f"Skipping {Path(image_path).name}: no finite caption losses")
            image.close()
            continue

        finite_weights = torch.stack(finite_weights)
        if finite_weights.sum().item() > 0:
            finite_weights = finite_weights / finite_weights.sum()
        else:
            finite_weights = torch.ones_like(finite_weights) / len(finite_weights)

        total_loss = torch.zeros((), device=device)
        for w, loss_i in zip(finite_weights, finite_losses):
            total_loss = total_loss + w * loss_i

        if not torch.isfinite(total_loss):
            skipped_nonfinite_total_loss += 1
            print(f"Skipping {Path(image_path).name}: non-finite total loss")
            image.close()
            continue

        total_loss.backward()
        optimizer.step()
        mlp_optimizer.step()

        running_loss += total_loss.item()
        used_items += 1

        print(f"[{idx}/{len(dataset)}] total_loss={total_loss.item():.6f} | {Path(image_path).name}")
        image.close()

    avg_loss = float("nan") if used_items == 0 else running_loss / used_items
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1} average loss = {avg_loss:.6f}")
    print(f"Used items this epoch = {used_items}/{len(dataset)}")

log_path = results_dir / f"{RUN_TAG}.json"
blip_ckpt_path = results_dir / f"{RUN_TAG}_blip.pt"
mlp_ckpt_path = results_dir / f"{RUN_TAG}_mlp.pt"

with open(log_path, "w") as f:
    json.dump(
        {
            "run_tag": RUN_TAG,
            "input_path": INPUT_PATH,
            "epochs": EPOCHS,
            "seed": SEED,
            "tau_start": TAU_START,
            "tau_end": TAU_END,
            "caption_lr": CAPTION_LR,
            "mlp_lr": MLP_LR,
            "use_alignment": USE_ALIGNMENT,
            "use_fluency": USE_FLUENCY,
            "use_agreement": USE_AGREEMENT,
            "epoch_losses": epoch_losses,
            "label_counts": label_counts,
            "skipped_nonfinite_raw_scores": skipped_nonfinite_raw_scores,
            "skipped_nonfinite_weights": skipped_nonfinite_weights,
            "skipped_no_finite_caption_losses": skipped_no_finite_caption_losses,
            "skipped_nonfinite_total_loss": skipped_nonfinite_total_loss,
        },
        f,
        indent=2,
    )

torch.save(model.state_dict(), blip_ckpt_path)
torch.save(mlp.state_dict(), mlp_ckpt_path)

print(f"Saved {log_path}")
print(f"Saved {blip_ckpt_path}")
print(f"Saved {mlp_ckpt_path}")
