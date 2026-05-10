###################################################################
#  This script processes the captions with agreement data to compute raw and normalized feature vectors for each caption. The features are based on alignment, fluency, and agreement scores. The normalized features are computed by standardizing the raw features using their mean and standard deviation across the dataset. Finally, the processed data with added features is saved to a new JSON file.
#  The output JSON file will contain the original caption data along with two new fields for each caption: "raw_features" which includes the original alignment, fluency, and agreement scores, and "normalized_features" which contains the z-score normalized versions of these features. This processed data can then be used for training a confidence MLP to predict weights for each caption based on these features.
###################################################################

import json
import math

INPUT_PATH = "captions_with_all_signals.json"
OUTPUT_PATH = "captions_with_features.json"

print("Loading data...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

items = data["captions_with_agreement"]

# Build raw feature vectors in the order:
# [alignment, fluency, agreement]
raw_features = []
for item in items:
    vec = [
        float(item["alignment_score"]),
        float(item["fluency_score"]),
        float(item["agreement_score"]),
    ]
    raw_features.append(vec)

# Compute mean and std for each feature dimension
num_features = 3
means = []
stds = []

for j in range(num_features):
    values = [vec[j] for vec in raw_features]
    mean_j = sum(values) / len(values)
    var_j = sum((v - mean_j) ** 2 for v in values) / len(values)
    std_j = math.sqrt(var_j)

    # avoid division by zero
    if std_j == 0:
        std_j = 1.0

    means.append(mean_j)
    stds.append(std_j)

# Add raw and normalized features to each caption
for item, vec in zip(items, raw_features):
    norm_vec = [
        (vec[j] - means[j]) / stds[j]
        for j in range(num_features)
    ]

    item["raw_features"] = {
        "alignment": vec[0],
        "fluency": vec[1],
        "agreement": vec[2]
    }

    item["normalized_features"] = norm_vec

print("\nFeature statistics:")
print(f"alignment mean={means[0]:.6f}, std={stds[0]:.6f}")
print(f"fluency   mean={means[1]:.6f}, std={stds[1]:.6f}")
print(f"agreement mean={means[2]:.6f}, std={stds[2]:.6f}")

print("\nNormalized feature vectors:")
for i, item in enumerate(items, start=1):
    print(f"{i}. {item['text']}")
    print(f"   raw_features = {item['raw_features']}")
    print(f"   normalized_features = {item['normalized_features']}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
