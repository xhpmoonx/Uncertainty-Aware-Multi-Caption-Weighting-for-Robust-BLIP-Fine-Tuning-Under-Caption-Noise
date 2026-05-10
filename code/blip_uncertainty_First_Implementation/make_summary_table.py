import json
import os
from pathlib import Path

RESULTS_DIR = Path("results")

rows = []

for learned_file in sorted(RESULTS_DIR.glob("*_learned.json")):
    tag = learned_file.name.replace("_learned.json", "")
    uniform_file = RESULTS_DIR / f"{tag}_uniform.json"

    if not uniform_file.exists():
        print(f"Skipping {tag}: missing uniform file")
        continue

    with open(learned_file) as f:
        learned = json.load(f)
    with open(uniform_file) as f:
        uniform = json.load(f)

    learned_losses = learned["epoch_losses"]
    uniform_losses = uniform["epoch_losses"]

    n_epochs = min(len(learned_losses), len(uniform_losses))

    row = {
        "tag": tag,
        "max_images": learned.get("max_images"),
        "epochs": learned.get("epochs"),
        "seed": learned.get("seed"),
        "learned_final": learned_losses[n_epochs - 1],
        "uniform_final": uniform_losses[n_epochs - 1],
        "final_gap": uniform_losses[n_epochs - 1] - learned_losses[n_epochs - 1],
        "learned_losses": learned_losses[:n_epochs],
        "uniform_losses": uniform_losses[:n_epochs],
    }
    rows.append(row)

rows = sorted(rows, key=lambda r: (r["max_images"], r["epochs"], r["seed"]))

print("\n===== SUMMARY TABLE =====\n")
for r in rows:
    print(f"Tag        : {r['tag']}")
    print(f"Images     : {r['max_images']}")
    print(f"Epochs     : {r['epochs']}")
    print(f"Seed       : {r['seed']}")
    print(f"Learned    : {r['learned_final']:.6f}")
    print(f"Uniform    : {r['uniform_final']:.6f}")
    print(f"Gap(U-L)   : {r['final_gap']:.6f}")
    print(f"L losses   : {r['learned_losses']}")
    print(f"U losses   : {r['uniform_losses']}")
    print()

with open("results/summary_table.json", "w") as f:
    json.dump(rows, f, indent=2)

print("Saved results/summary_table.json")
