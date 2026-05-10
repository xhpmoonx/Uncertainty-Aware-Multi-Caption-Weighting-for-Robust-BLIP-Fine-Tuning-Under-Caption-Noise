import json
import random
from copy import deepcopy
from pathlib import Path

SEED = 123
random.seed(SEED)

IN_PATH = Path("coco_work/subsets/coco_train_500_generatedK3_seed123.json")
OUT_DIR = Path("coco_work/subsets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(IN_PATH, "r") as f:
    data = json.load(f)

all_caption_pool = []
for item in data:
    for cap in item["generated_captions"]:
        all_caption_pool.append(cap)

def drop_words(text, drop_prob, min_words=3):
    words = [w for w in text.split() if w.strip()]
    if len(words) <= min_words:
        return " ".join(words)

    kept = [w for w in words if random.random() > drop_prob]

    if len(kept) < min_words:
        if len(words) >= min_words:
            kept = random.sample(words, min_words)
        else:
            kept = words[:]

    return " ".join(kept)

def shuffle_words(text):
    words = text.split()
    if len(words) <= 1:
        return text
    words = words[:]
    random.shuffle(words)
    return " ".join(words)

def mismatch_caption(original_text):
    pool = [c for c in all_caption_pool if c != original_text]
    return random.choice(pool) if pool else original_text

def corrupt_item(item, corruption_type, corruption_rate):
    new_item = deepcopy(item)
    new_caps = []

    for cap in item["generated_captions"]:
        is_corrupted = random.random() < corruption_rate
        label = "clean"

        if is_corrupted:
            if corruption_type == "drop":
                cap_new = drop_words(cap, drop_prob=0.4)
                label = "drop"
            elif corruption_type == "shuffle":
                cap_new = shuffle_words(cap)
                label = "shuffle"
            elif corruption_type == "mismatch":
                cap_new = mismatch_caption(cap)
                label = "mismatch"
            else:
                raise ValueError(f"Unknown corruption_type: {corruption_type}")
        else:
            cap_new = cap

        new_caps.append({
            "text": cap_new,
            "label": label,
            "is_corrupted": label != "clean"
        })

    new_item["generated_captions"] = new_caps
    return new_item

# 1) clean version
clean_data = []
for item in data:
    new_item = deepcopy(item)
    new_item["generated_captions"] = [
        {
            "text": cap,
            "label": "clean",
            "is_corrupted": False
        }
        for cap in item["generated_captions"]
    ]
    clean_data.append(new_item)

clean_out = OUT_DIR / "coco_train_500_generatedK3_clean_seed123.json"
with open(clean_out, "w") as f:
    json.dump(clean_data, f, indent=2)
print("Saved:", clean_out)

# 2) corrupted versions
configs = [
    ("drop", 0.4),
    ("drop", 0.6),
    ("shuffle", 0.4),
    ("shuffle", 0.6),
    ("mismatch", 0.4),
    ("mismatch", 0.6),
]

for corruption_type, corruption_rate in configs:
    out_data = [corrupt_item(item, corruption_type, corruption_rate) for item in data]
    rate_str = str(int(corruption_rate * 100))
    out_path = OUT_DIR / f"coco_train_500_generatedK3_{corruption_type}{rate_str}_seed123.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print("Saved:", out_path)
