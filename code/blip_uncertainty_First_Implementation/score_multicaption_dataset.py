import json
import math
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

INPUT_PATH = "mini_dataset_raw.json"
MODEL_PATH = "confidence_mlp_warmup.pt"
OUTPUT_PATH = "mini_dataset_scored.json"

ALIGN_MODEL = "Salesforce/blip-itm-base-coco"
FLUENCY_MODEL = "distilgpt2"
AGREE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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

print("Loading multi-caption dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

print("Loading trained confidence MLP...")
mlp = ConfidenceMLP().to(device)
mlp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
mlp.eval()

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

results = []

for idx, item in enumerate(dataset, start=1):
    image_path = item["image_path"]
    captions = item["generated_captions"]

    print(f"\n[{idx}/{len(dataset)}] Processing {image_path}")
    image = Image.open(image_path).convert("RGB")

    # ----------------------
    # Agreement
    # ----------------------
    embeddings = agree_model.encode(captions)
    sim_matrix = cosine_similarity(embeddings)

    agreement_scores = []
    n = len(captions)
    for i in range(n):
        others = [sim_matrix[i][j] for j in range(n) if j != i]
        score = sum(others) / len(others) if others else 0.0
        agreement_scores.append(float(score))

    # ----------------------
    # Fluency
    # ----------------------
    fluency_scores = []
    for text in captions:
        inputs = fluency_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = fluency_model(**inputs, labels=inputs["input_ids"])
        avg_nll = outputs.loss.item()
        fluency_scores.append(float(-avg_nll))

    # ----------------------
    # Alignment
    # ----------------------
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
            raise ValueError("Could not find ITM logits in model output.")

        alignment_scores.append(float(score))

    # ----------------------
    # Features
    # order = [alignment, fluency, agreement]
    # ----------------------
    raw_features = []
    for a, f, g in zip(alignment_scores, fluency_scores, agreement_scores):
        raw_features.append([a, f, g])

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

    with torch.no_grad():
        raw_scores = mlp(x)
        weights = torch.softmax(raw_scores / TEMPERATURE, dim=0)

    record = {
        "image_id": item["image_id"],
        "image_path": image_path,
        "captions": []
    }

    for i, text in enumerate(captions):
        rec = {
            "text": text,
            "alignment_score": alignment_scores[i],
            "fluency_score": fluency_scores[i],
            "agreement_score": agreement_scores[i],
            "raw_features": {
                "alignment": raw_features[i][0],
                "fluency": raw_features[i][1],
                "agreement": raw_features[i][2],
            },
            "normalized_features": norm_features[i],
            "raw_score": float(raw_scores[i].item()),
            "softmax_weight": float(weights[i].item()),
        }
        record["captions"].append(rec)

        print(f"  {i+1}. {text}")
        print(f"     raw_score={rec['raw_score']:.4f}, weight={rec['softmax_weight']:.4f}")

    print(f"  Sum of weights = {weights.sum().item():.4f}")
    results.append(record)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved scored dataset to {OUTPUT_PATH}")
