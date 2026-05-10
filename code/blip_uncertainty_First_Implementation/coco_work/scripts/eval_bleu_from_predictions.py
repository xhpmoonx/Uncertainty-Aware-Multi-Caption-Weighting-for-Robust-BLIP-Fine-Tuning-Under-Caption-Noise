import json
import sys
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

if len(sys.argv) != 2:
    print("Usage: python eval_bleu_from_predictions.py <predictions.json>")
    sys.exit(1)

PRED_PATH = Path(sys.argv[1])
OUT_JSON = PRED_PATH.with_name(PRED_PATH.stem + "_bleu.json")

with open(PRED_PATH) as f:
    data = json.load(f)

meta_keys = {"image_id", "image_path", "references"}
models = [k for k in data[0].keys() if k not in meta_keys]

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

print("Saved:", OUT_JSON)
for model, scores in results.items():
    print(model, scores)
