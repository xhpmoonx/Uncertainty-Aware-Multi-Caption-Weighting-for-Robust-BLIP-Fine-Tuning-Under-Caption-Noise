import json
from pathlib import Path

IN_TRAIN = Path.home() / "blip_uncertainty" / "coco_work" / "subsets" / "coco_train_500_seed123.json"
IN_VAL   = Path.home() / "blip_uncertainty" / "coco_work" / "subsets" / "coco_val_100_seed123.json"

OUT_TRAIN = Path.home() / "blip_uncertainty" / "coco_work" / "subsets" / "coco_train_500_raw_seed123.json"
OUT_VAL   = Path.home() / "blip_uncertainty" / "coco_work" / "subsets" / "coco_val_100_raw_seed123.json"

def convert(infile, outfile):
    with open(infile, "r") as f:
        data = json.load(f)

    converted = []
    for item in data:
        converted.append({
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "generated_captions": item["captions"]
        })

    with open(outfile, "w") as f:
        json.dump(converted, f, indent=2)

    print("Saved:", outfile)
    print("Count:", len(converted))
    print("Example:")
    print(json.dumps(converted[0], indent=2)[:1200])

convert(IN_TRAIN, OUT_TRAIN)
print()
convert(IN_VAL, OUT_VAL)
