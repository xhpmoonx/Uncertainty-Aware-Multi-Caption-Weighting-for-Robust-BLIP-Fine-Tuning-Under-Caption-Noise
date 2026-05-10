import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_PATH = "captions_test.json"
OUTPUT_PATH = "captions_with_agreement.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading captions...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

captions = data["generated_captions"]
print("Captions:")
for i, c in enumerate(captions, start=1):
    print(f"{i}. {c}")

print("\nLoading sentence embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Encoding captions...")
embeddings = model.encode(captions)

print("Computing cosine similarity matrix...")
sim_matrix = cosine_similarity(embeddings)

agreement_scores = []
n = len(captions)

for i in range(n):
    others = [sim_matrix[i][j] for j in range(n) if j != i]
    avg_agreement = sum(others) / len(others) if others else 0.0
    agreement_scores.append(float(avg_agreement))

data["captions_with_agreement"] = []
for caption, score in zip(captions, agreement_scores):
    data["captions_with_agreement"].append({
        "text": caption,
        "agreement_score": score
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print("\nAgreement scores:")
for i, (caption, score) in enumerate(zip(captions, agreement_scores), start=1):
    print(f"{i}. {caption}")
    print(f"   agreement_score = {score:.4f}")

print(f"\nSaved to {OUTPUT_PATH}")
