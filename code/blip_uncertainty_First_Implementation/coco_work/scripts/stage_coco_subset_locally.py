import json
import shutil
from pathlib import Path

BASE = Path.home() / "blip_uncertainty" / "coco_work"
SUBSETS = BASE / "subsets"
LOCAL = BASE / "local_coco"

TRAIN_IN = SUBSETS / "coco_train_500_raw_seed123.json"
VAL_IN   = SUBSETS / "coco_val_100_raw_seed123.json"

TRAIN_IMG_DIR = LOCAL / "train2017"
VAL_IMG_DIR   = LOCAL / "val2017"

TRAIN_OUT = SUBSETS / "coco_train_500_raw_local_seed123.json"
VAL_OUT   = SUBSETS / "coco_val_100_raw_local_seed123.json"

TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)

def stage(infile, outjson, outdir):
    with open(infile, "r") as f:
        data = json.load(f)

    new_data = []
    for i, item in enumerate(data, start=1):
        src = Path(item["image_path"])
        dst = outdir / src.name

        if not dst.exists():
            shutil.copy2(src, dst)

        new_item = dict(item)
        new_item["image_path"] = str(dst)
        new_data.append(new_item)

        if i <= 3:
            print(f"[sample {i}] {src} -> {dst}")

    with open(outjson, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Saved {outjson} ({len(new_data)} items)")

stage(TRAIN_IN, TRAIN_OUT, TRAIN_IMG_DIR)
print()
stage(VAL_IN, VAL_OUT, VAL_IMG_DIR)
