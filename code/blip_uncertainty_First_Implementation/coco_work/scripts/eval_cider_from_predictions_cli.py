import json
import argparse
from pathlib import Path
from pycocoevalcap.cider.cider import Cider

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, required=True)
args = parser.parse_args()

PRED_PATH = Path(args.pred_path)

with open(PRED_PATH, "r") as f:
    data = json.load(f)

meta_keys = {"image_id", "image_path", "references"}
model_keys = [k for k in data[0].keys() if k not in meta_keys]

results = {}

for model_key in model_keys:
    gts = {}
    res = {}
    for i, item in enumerate(data):
        gts[i] = item["references"]
        res[i] = [item[model_key]]

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    results[model_key] = {
        "CIDEr": float(score),
        "n": len(data),
    }

out_json = PRED_PATH.with_name(PRED_PATH.stem + "_cider.json")
out_md = PRED_PATH.with_name(PRED_PATH.stem + "_cider.md")

with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

lines = [f"# CIDEr Results for {PRED_PATH.name}", ""]
for model_key, vals in results.items():
    lines.append(f"## {model_key}")
    lines.append(f"- CIDEr: {vals['CIDEr']:.6f}")
    lines.append(f"- n: {vals['n']}")
    lines.append("")

with open(out_md, "w") as f:
    f.write("\n".join(lines))

print("Saved:", out_json)
print("Saved:", out_md)
for model_key, vals in results.items():
    print(model_key)
    print(f"  CIDEr: {vals['CIDEr']:.6f}")
    print()
