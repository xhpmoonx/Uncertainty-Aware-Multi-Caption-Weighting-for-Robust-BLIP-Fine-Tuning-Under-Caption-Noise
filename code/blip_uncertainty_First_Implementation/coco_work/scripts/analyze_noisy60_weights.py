import json
from pathlib import Path
from collections import defaultdict

LABELED_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123_noisy60_labeled.json")
SCORED_PATH = Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60.json")

with open(LABELED_PATH) as f:
    labeled = json.load(f)

with open(SCORED_PATH) as f:
    scored = json.load(f)

groups = defaultdict(list)

for item_lab, item_sc in zip(labeled, scored):
    for c_lab, c_sc in zip(item_lab["captions"], item_sc["captions"]):
        groups[c_lab["label"]].append(c_sc["softmax_weight"])
        groups["corrupted" if c_lab["is_corrupted"] else "clean_binary"].append(c_sc["softmax_weight"])

def avg(xs):
    return sum(xs) / len(xs)

for key in ["clean_binary", "corrupted", "clean", "drop", "shuffle", "mismatch"]:
    if key in groups:
        print(f"{key}: n={len(groups[key])}, mean_weight={avg(groups[key]):.6f}")
