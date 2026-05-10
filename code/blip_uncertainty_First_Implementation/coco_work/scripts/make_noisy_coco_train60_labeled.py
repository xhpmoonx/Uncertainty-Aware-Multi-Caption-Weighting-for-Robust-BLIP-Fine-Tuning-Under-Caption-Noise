import json
import random
from pathlib import Path

IN_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123.json")
OUT_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123_noisy60_labeled.json")

SEED = 123
NOISE_PROB = 0.6
rng = random.Random(SEED)

with open(IN_PATH, "r") as f:
    data = json.load(f)

all_captions = []
for item in data:
    all_captions.extend(item["generated_captions"])

def drop_words(text, drop_prob=0.3):
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if rng.random() > drop_prob]
    if len(kept) < 2:
        kept = words[:2]
    return " ".join(kept)

def shuffle_words(text):
    words = text.split()
    if len(words) <= 3:
        return text
    words = words[:]
    rng.shuffle(words)
    return " ".join(words)

def mismatched_caption():
    return rng.choice(all_captions)

new_data = []

for item in data:
    new_caps = []
    for cap in item["generated_captions"]:
        entry = {
            "original_text": cap,
            "text": cap,
            "label": "clean",
            "is_corrupted": False,
        }

        if rng.random() < NOISE_PROB:
            mode = rng.choice(["drop", "shuffle", "mismatch"])
            if mode == "drop":
                noisy = drop_words(cap)
            elif mode == "shuffle":
                noisy = shuffle_words(cap)
            else:
                noisy = mismatched_caption()

            entry["text"] = noisy
            entry["label"] = mode
            entry["is_corrupted"] = True

        new_caps.append(entry)

    new_item = {
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "captions": new_caps,
    }
    new_data.append(new_item)

with open(OUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print("Saved:", OUT_PATH)
print("Items:", len(new_data))
