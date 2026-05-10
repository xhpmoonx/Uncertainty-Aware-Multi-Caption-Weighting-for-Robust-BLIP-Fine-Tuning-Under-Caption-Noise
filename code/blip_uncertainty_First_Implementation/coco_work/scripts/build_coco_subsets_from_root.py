import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python3 build_coco_subsets_from_root.py /path/to/COCO_ROOT")
    sys.exit(1)

COCO_ROOT = Path(sys.argv[1])

TRAIN_JSON = COCO_ROOT / "annotations" / "captions_train2017.json"
VAL_JSON   = COCO_ROOT / "annotations" / "captions_val2017.json"
TRAIN_DIR  = COCO_ROOT / "train2017"
VAL_DIR    = COCO_ROOT / "val2017"

OUT_DIR = Path.home() / "blip_uncertainty" / "coco_work" / "subsets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_N = 500
VAL_N = 100
SEED = 123

def build_subset(annotation_file, image_dir, split, n, seed):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    caps_by_image = defaultdict(list)
    for ann in data["annotations"]:
        caps_by_image[ann["image_id"]].append(ann["caption"])

    valid_ids = [
        iid for iid in id_to_file
        if iid in caps_by_image and (image_dir / id_to_file[iid]).exists()
    ]

    if len(valid_ids) < n:
        raise ValueError(f"Requested {n} images, but only found {len(valid_ids)} valid ones in {split}")

    rng = random.Random(seed)
    chosen_ids = rng.sample(valid_ids, n)

    subset = []
    for iid in chosen_ids:
        fname = id_to_file[iid]
        subset.append({
            "image_id": iid,
            "file_name": fname,
            "image_path": str(image_dir / fname),
            "captions": caps_by_image[iid],
            "split": split
        })

    return subset

train_subset = build_subset(TRAIN_JSON, TRAIN_DIR, "train", TRAIN_N, SEED)
val_subset   = build_subset(VAL_JSON, VAL_DIR, "val", VAL_N, SEED)

train_out = OUT_DIR / "coco_train_500_seed123.json"
val_out   = OUT_DIR / "coco_val_100_seed123.json"

with open(train_out, "w") as f:
    json.dump(train_subset, f, indent=2)

with open(val_out, "w") as f:
    json.dump(val_subset, f, indent=2)

print("Saved:", train_out)
print("Saved:", val_out)
print("Train examples:", len(train_subset))
print("Val examples:", len(val_subset))
print("\nExample:")
print(json.dumps(train_subset[0], indent=2)[:1200])
