import json
import math
from pathlib import Path

IN_PATH = Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60.json")

OUT_PATHS = {
    "alignonly": Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60_alignonly.json"),
    "fluencyonly": Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60_fluencyonly.json"),
    "agreementonly": Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60_agreementonly.json"),
}

INDEX = {
    "alignonly": 0,
    "fluencyonly": 1,
    "agreementonly": 2,
}

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

with open(IN_PATH) as f:
    data = json.load(f)

for mode, out_path in OUT_PATHS.items():
    idx = INDEX[mode]
    new_data = []

    for item in data:
        new_item = {
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "captions": []
        }

        raw_scores = [cap["normalized_features"][idx] for cap in item["captions"]]
        weights = softmax(raw_scores)

        for cap, raw_score, weight in zip(item["captions"], raw_scores, weights):
            new_cap = dict(cap)
            new_cap["raw_score"] = raw_score
            new_cap["softmax_weight"] = weight
            new_item["captions"].append(new_cap)

        new_data.append(new_item)

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print("Saved:", out_path)
