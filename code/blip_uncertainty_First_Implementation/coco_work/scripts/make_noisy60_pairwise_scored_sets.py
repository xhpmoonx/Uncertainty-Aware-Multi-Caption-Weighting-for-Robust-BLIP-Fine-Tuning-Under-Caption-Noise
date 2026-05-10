import json
from pathlib import Path

IN_PATH = Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60.json")

OUTS = {
    "align_fluency": Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60_align_fluency.json"),
    "align_agreement": Path("coco_work/subsets/coco_train_500_scored_seed123_noisy60_align_agreement.json"),
}

with open(IN_PATH) as f:
    data = json.load(f)

for mode, out_path in OUTS.items():
    new_data = []

    for item in data:
        new_item = {
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "captions": []
        }

        for cap in item["captions"]:
            new_cap = dict(cap)
            a, f, g = cap["normalized_features"]

            if mode == "align_fluency":
                new_cap["normalized_features"] = [a, f, 0.0]
            elif mode == "align_agreement":
                new_cap["normalized_features"] = [a, 0.0, g]

            new_item["captions"].append(new_cap)

        new_data.append(new_item)

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print("Saved:", out_path)
