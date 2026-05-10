import json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

PRED_PATH = Path("results/coco500_val100_seed123_predictions_noisy60_ablation.json")
OUT_JSON = Path("results/coco500_val100_seed123_bleu_noisy60_ablation.json")
OUT_MD = Path("results/coco500_val100_seed123_bleu_noisy60_ablation.md")

with open(PRED_PATH) as f:
    data = json.load(f)

models = [
    "base",
    "learned_noisy60",
    "alignonly_noisy60",
    "fluencyonly_noisy60",
    "agreementonly_noisy60",
    "uniform_noisy60",
    "top1_noisy60",
]

smooth = SmoothingFunction().method1
results = {}

for model in models:
    refs, hyps = [], []
    for item in data:
        refs.append([r.lower().split() for r in item["references"]])
        hyps.append(item[model].lower().split())

    results[model] = {
        "BLEU-1": corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

lines = ["# Noisy60 Ablation BLEU Results", ""]
for model, scores in results.items():
    lines.append(f"## {model}")
    for k, v in scores.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

for model, scores in results.items():
    print(model)
    for k, v in scores.items():
        print(f"  {k}: {v:.6f}")
    print()

print("Saved:", OUT_JSON)
print("Saved:", OUT_MD)
