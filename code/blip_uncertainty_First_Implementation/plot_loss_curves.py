import json
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")

for learned_file in sorted(RESULTS_DIR.glob("*_learned.json")):
    tag = learned_file.name.replace("_learned.json", "")
    uniform_file = RESULTS_DIR / f"{tag}_uniform.json"
    if not uniform_file.exists():
        continue

    with open(learned_file) as f:
        learned = json.load(f)
    with open(uniform_file) as f:
        uniform = json.load(f)

    learned_losses = learned["epoch_losses"]
    uniform_losses = uniform["epoch_losses"]
    n_epochs = min(len(learned_losses), len(uniform_losses))
    x = list(range(1, n_epochs + 1))

    plt.figure()
    plt.plot(x, learned_losses[:n_epochs], marker="o", label="Learned weighting")
    plt.plot(x, uniform_losses[:n_epochs], marker="o", label="Uniform weighting")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"{tag}_loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
