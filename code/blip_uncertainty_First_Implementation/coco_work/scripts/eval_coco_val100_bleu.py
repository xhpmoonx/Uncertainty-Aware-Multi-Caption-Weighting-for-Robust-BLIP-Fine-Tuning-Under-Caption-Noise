import json
from pathlib import Path

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

PRED_PATH = Path("results/coco500_val100_seed123_predictions.json")
OUT_JSON = Path("results/coco500_val100_seed123_bleu.json")
OUT_MD = Path("results/coco500_val100_seed123_bleu.md")

with open(PRED_PATH, "r") as f:
    data = json.load(f)

models = ["base", "learned", "uniform", "top1"]
smooth = SmoothingFunction().method1

results = {}

for model in models:
    refs = []
    hyps = []

    for item in data:
        references = [r.lower().split() for r in item["references"]]
        hypothesis = item[model].lower().split()

        refs.append(references)
        hyps.append(hypothesis)

    bleu1 = corpus_bleu(refs, hyps, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    results[model] = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
    }

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

lines = ["# COCO Val-100 BLEU Results", ""]
for model, scores in results.items():
    lines.append(f"## {model}")
    for k, v in scores.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("Saved:", OUT_JSON)
print("Saved:", OUT_MD)
print()
for model, scores in results.items():
    print(model)
    for k, v in scores.items():
        print(f"  {k}: {v:.6f}")
    print()
