import json
import random

INPUT_PATH = "mini_dataset_raw.json"
OUTPUT_PATH = "warmup_pairs_text_only.json"

random.seed(42)

def word_drop(text, drop_ratio=0.3):
    words = text.split()
    if len(words) <= 2:
        return text
    keep = []
    for w in words:
        if random.random() > drop_ratio:
            keep.append(w)
    if len(keep) < 2:
        keep = words[:2]
    return " ".join(keep)

def word_shuffle(text):
    words = text.split()
    if len(words) <= 2:
        return text
    words_copy = words[:]
    random.shuffle(words_copy)
    return " ".join(words_copy)

print("Loading mini dataset...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# collect the first caption from every image for mismatch sampling
first_captions = [item["generated_captions"][0] for item in data if item["generated_captions"]]

results = []

for i, item in enumerate(data):
    captions = item["generated_captions"]
    if not captions:
        continue

    original = captions[0]

    # mismatch = use another image's first caption
    mismatch_candidates = [c for c in first_captions if c != original]
    mismatch_caption = random.choice(mismatch_candidates) if mismatch_candidates else original

    record = {
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "original_caption": original,
        "degraded_captions": {
            "word_drop": word_drop(original),
            "word_shuffle": word_shuffle(original),
            "mismatch": mismatch_caption
        }
    }

    results.append(record)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} warm-up examples to {OUTPUT_PATH}\n")

for rec in results:
    print(f"Image {rec['image_id']} -> {rec['image_path']}")
    print(f"  original:      {rec['original_caption']}")
    print(f"  word_drop:     {rec['degraded_captions']['word_drop']}")
    print(f"  word_shuffle:  {rec['degraded_captions']['word_shuffle']}")
    print(f"  mismatch:      {rec['degraded_captions']['mismatch']}")
    print()
