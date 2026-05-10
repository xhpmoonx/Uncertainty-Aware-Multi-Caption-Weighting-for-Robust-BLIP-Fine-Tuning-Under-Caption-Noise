import json
import sys
from pathlib import Path

from pycocoevalcap.cider.cider import Cider

if len(sys.argv) != 2:
    print("Usage: python eval_cider_from_predictions.py <predictions.json>")
    sys.exit(1)

PRED_PATH = Path(sys.argv[1])
OUT_JSON = PRED_PATH.with_name(PRED_PATH.stem + "_cider.json")
OUT_MD = PRED_PATH.with_name(PRED_PATH.stem + "_cider.md")

with open(PRED_PATH, "r") as f:
    data = json.load(f)

meta_keys = {"image_id", "image_path", "references"}
models = [k for k in data[0].keys() if k not in meta_keys]

results = {}

for model_key in models:
    gts = {}
    res = {}

    for i, item in enumerate(data):
        idx = str(i)
        gts[idx] = item["references"]
        res[idx] = [item[model_key]]

    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)

    results[model_key] = {
        "CIDEr": float(score),
        "n": len(data),
    }

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

lines = [f"# CIDEr Results for {PRED_PATH.name}", ""]
for model_key, vals in results.items():
    lines.append(f"## {model_key}")
    lines.append(f"- CIDEr: {vals['CIDEr']:.6f}")
    lines.append(f"- n: {vals['n']}")
    lines.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("Saved:", OUT_JSON)
print("Saved:", OUT_MD)
print()
for model_key, vals in results.items():
    print(model_key)
    print("  CIDEr:", f"{vals['CIDEr']:.6f}")
