import json
from pathlib import Path

FILES = {
    "clean_train_loss": {
        "learned": "results/coco500_10ep_seed123_learned_learned.json",
        "uniform": "results/coco500_10ep_seed123_uniform_uniform.json",
        "top1":    "results/coco500_10ep_seed123_top1_top1.json",
    },
    "noisy40_train_loss": {
        "learned": "results/coco500_noisy40_10ep_seed123_learned_learned.json",
        "uniform": "results/coco500_noisy40_10ep_seed123_uniform_uniform.json",
        "top1":    "results/coco500_noisy40_10ep_seed123_top1_top1.json",
    },
    "noisy60_train_loss": {
        "learned": "results/coco500_noisy60_10ep_seed123_learned_learned.json",
        "uniform": "results/coco500_noisy60_10ep_seed123_uniform_uniform.json",
        "top1":    "results/coco500_noisy60_10ep_seed123_top1_top1.json",
    },
    "clean_bleu": "results/coco500_val100_seed123_bleu.json",
    "noisy40_bleu": "results/coco500_val100_seed123_bleu_noisy40.json",
    "noisy60_bleu": "results/coco500_val100_seed123_bleu_noisy60.json",
}

summary = {}

for group in ["clean_train_loss", "noisy40_train_loss", "noisy60_train_loss"]:
    summary[group] = {}
    for name, path in FILES[group].items():
        with open(path) as f:
            d = json.load(f)
        summary[group][name] = {
            "epoch_losses": d["epoch_losses"],
            "final_loss": d["epoch_losses"][-1],
        }

for group in ["clean_bleu", "noisy40_bleu", "noisy60_bleu"]:
    with open(FILES[group]) as f:
        summary[group] = json.load(f)

out = Path("results/final_experiment_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved:", out)
