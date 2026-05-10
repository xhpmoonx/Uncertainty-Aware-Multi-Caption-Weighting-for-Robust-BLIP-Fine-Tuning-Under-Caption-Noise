import json
import random
from pathlib import Path

IN_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123.json")
OUT_PATH = Path("coco_work/subsets/coco_train_500_raw_local_seed123_noisy40.json")

SEED = 123
NOISE_PROB = 0.4

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
    middle = words[:]
    rng.shuffle(middle)
    return " ".join(middle)

def mismatched_caption():
    return rng.choice(all_captions)

def corrupt(text):
    mode = rng.choice(["drop", "shuffle", "mismatch"])
    if mode == "drop":
        return drop_words(text)
    elif mode == "shuffle":
        return shuffle_words(text)
    else:
        return mismatched_caption()

new_data = []
num_changed = 0
total_caps = 0

for item in data:
    caps = []
    for cap in item["generated_captions"]:
        total_caps += 1
        if rng.random() < NOISE_PROB:
            caps.append(corrupt(cap))
            num_changed += 1
        else:
            caps.append(cap)

    new_item = dict(item)
    new_item["generated_captions"] = caps
    new_data.append(new_item)

with open(OUT_PATH, "w") as f:
    json.dump(new_data, f, indent=2)

print("Saved:", OUT_PATH)
print("Items:", len(new_data))
print("Captions changed:", num_changed, "/", total_caps, f"= {num_changed/total_caps:.3f}")

print("\nExample before/after:")
for i in range(2):
    print(f"\nImage {i+1}")
    print("OLD:")
    for c in data[i]["generated_captions"]:
        print(" -", c)
    print("NEW:")
    for c in new_data[i]["generated_captions"]:
        print(" -", c)
