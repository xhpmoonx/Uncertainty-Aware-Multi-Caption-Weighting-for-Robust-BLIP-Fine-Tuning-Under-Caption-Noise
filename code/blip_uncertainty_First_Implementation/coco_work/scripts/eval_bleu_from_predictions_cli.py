import json
import argparse
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, required=True)
args = parser.parse_args()

PRED_PATH = Path(args.pred_path)
smooth = SmoothingFunction().method1

with open(PRED_PATH, "r") as f:
    data = json.load(f)

meta_keys = {"image_id", "image_path", "references"}
model_keys = [k for k in data[0].keys() if k not in meta_keys]

results = {}

for model_key in model_keys:
    refs, hyps = [], []
    for item in data:
        refs.append([r.lower().split() for r in item["references"]])
        hyps.append(item[model_key].lower().split())

    results[model_key] = {
        "BLEU-1": corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs, hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

out_json = PRED_PATH.with_name(PRED_PATH.stem + "_bleu.json")
out_md = PRED_PATH.with_name(PRED_PATH.stem + "_bleu.md")

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

lines = [f"# BLEU Results for {PRED_PATH.name}", ""]
for model_key, vals in results.items():
    lines.append(f"## {model_key}")
    for k, v in vals.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")

with open(out_md, "w") as f:
    f.write("\n".join(lines))

print("Saved:", out_json)
print("Saved:", out_md)
for model_key, vals in results.items():
    print(model_key)
    for k, v in vals.items():
        print(f"  {k}: {v:.6f}")
    print()
