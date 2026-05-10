import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForImageTextRetrieval,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="coco_work/subsets/coco_train_500_generatedK3_shuffle60_seed123.json")
parser.add_argument("--mlp_path", type=str, default="results/dynamic_joint_generatedK3_shuffle60_seed123_mlp.pt")
parser.add_argument("--max_examples", type=int, default=15)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--mixed_only", action="store_true")
parser.add_argument("--output_path", type=str, default="results/shuffle60_weight_audit_seed123.json")
args = parser.parse_args()

ALIGN_MODEL = "Salesforce/blip-itm-base-coco"
FLUENCY_MODEL = "distilgpt2"
AGREE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
agree_model = SentenceTransformer(AGREE_MODEL, device=device)

print("Loading confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(args.mlp_path, map_location=device))
mlp.eval()


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


print("Loading dataset...")
with open(args.input_path, "r") as f:
    dataset = json.load(f)

selected = []
for record in dataset:
    captions, labels, corrupted = extract_captions(record)
    if args.mixed_only and len(set(labels)) <= 1:
        continue
    selected.append(record)
    if len(selected) >= args.max_examples:
        break

print(f"Auditing {len(selected)} examples")

audit = []
picked_clean = 0
picked_corrupted = 0

for ex_idx, record in enumerate(selected, start=1):
    image_path = record["image_path"]
    image = Image.open(image_path).convert("RGB")
    captions, labels, corrupted_flags = extract_captions(record)

    alignment_scores = compute_alignment_scores(image, captions)
    fluency_scores = compute_fluency_scores(captions)
    agreement_scores = compute_agreement_scores(captions)

    raw_features = []
    for a, f, g in zip(alignment_scores, fluency_scores, agreement_scores):
        raw_features.append([a, f, g])

    norm_features = zscore_features(raw_features)
    x = torch.tensor(norm_features, dtype=torch.float32, device=device)

    with torch.no_grad():
        raw_scores = mlp(x)
        weights = torch.softmax(raw_scores / args.temperature, dim=0)

    threshold = (1.0 / len(captions)) * 0.5
    clipped_weights = weights * (weights > threshold).float()
    if clipped_weights.sum().item() > 0:
        clipped_weights = clipped_weights / clipped_weights.sum()
    else:
        clipped_weights = weights.clone()

    picked_idx = int(torch.argmax(weights).item())
    picked_label = labels[picked_idx]

    if picked_label == "clean":
        picked_clean += 1
    else:
        picked_corrupted += 1

    print("\n" + "=" * 100)
    print(f"[{ex_idx}/{len(selected)}] {Path(image_path).name} | picked_idx={picked_idx} | picked_label={picked_label}")
    print(f"clip_threshold = {threshold:.6f}")

    row_list = []
    for i, (text, label) in enumerate(zip(captions, labels)):
        row = {
            "idx": i,
            "label": label,
            "is_corrupted": bool(corrupted_flags[i]),
            "text": text,
            "alignment": safe_float(alignment_scores[i]),
            "fluency": safe_float(fluency_scores[i]),
            "agreement": safe_float(agreement_scores[i]),
            "z_alignment": safe_float(norm_features[i][0]),
            "z_fluency": safe_float(norm_features[i][1]),
            "z_agreement": safe_float(norm_features[i][2]),
            "raw_score": safe_float(raw_scores[i].item()),
            "weight": safe_float(weights[i].item()),
            "clipped_weight": safe_float(clipped_weights[i].item()),
        }
        row_list.append(row)

        short_text = text[:110].replace("\n", " ")
        print(
            f"[{i}] label={label:<8} "
            f"w={row['weight']:.4f} "
            f"wclip={row['clipped_weight']:.4f} "
            f"raw={row['raw_score']:.4f} "
            f"a={row['alignment']:.4f} "
            f"f={row['fluency']:.4f} "
            f"g={row['agreement']:.4f} "
            f"| {short_text}"
        )

    audit.append({
        "image_id": record.get("image_id"),
        "image_path": image_path,
        "picked_idx": picked_idx,
        "picked_label": picked_label,
        "rows": row_list,
    })

summary = {
    "input_path": args.input_path,
    "mlp_path": args.mlp_path,
    "audited_examples": len(audit),
    "mixed_only": args.mixed_only,
    "temperature": args.temperature,
    "picked_clean": picked_clean,
    "picked_corrupted": picked_corrupted,
}

out = {
    "summary": summary,
    "examples": audit,
}

with open(args.output_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "#" * 100)
print("SUMMARY")
print(summary)
print("Saved:", args.output_path)

