import json
from pathlib import Path

IN_PATH = Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60.json")
OUT_PATH = Path("coco_work/subsets/coco_train_500_hardfilter020_seed123_noisy60.json")
THRESH = 0.20

with open(IN_PATH, "r") as f:
    data = json.load(f)

new_data = []
kept_counts = []

for item in data:
    caps = item["captions"]
    caps = sorted(caps, key=lambda c: c.get("softmax_weight", 0.0), reverse=True)

    kept = [c for c in caps if c.get("softmax_weight", 0.0) >= THRESH]
    if not kept:
        kept = [caps[0]]

    new_item = dict(item)
    new_item["captions"] = kept
    new_data.append(new_item)
    kept_counts.append(len(kept))

with open(OUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print("Saved:", OUT_PATH)
print("Items:", len(new_data))
print("Avg kept per image:", sum(kept_counts) / len(kept_counts))
print("Min kept:", min(kept_counts))
print("Max kept:", max(kept_counts))
