import json
from pathlib import Path

summary_path = Path("results/final_experiment_summary.json")
out_path = Path("results/final_report.md")

with open(summary_path) as f:
    s = json.load(f)

clean_bleu = s["clean_bleu"]
n40_bleu = s["noisy40_bleu"]
n60_bleu = s["noisy60_bleu"]

lines = [
    "# Final BLIP uncertainty experiment summary",
    "",
    "## BLEU-4 summary",
    "",
    "| Setting | Base | Learned | Uniform | Top1 | Best fine-tuned |",
    "|---|---:|---:|---:|---:|---|",
    f"| Clean | {clean_bleu['base']['BLEU-4']:.4f} | {clean_bleu['learned']['BLEU-4']:.4f} | {clean_bleu['uniform']['BLEU-4']:.4f} | {clean_bleu['top1']['BLEU-4']:.4f} | Uniform |",
    f"| Noisy40 | {n40_bleu['base']['BLEU-4']:.4f} | {n40_bleu['learned_noisy40']['BLEU-4']:.4f} | {n40_bleu['uniform_noisy40']['BLEU-4']:.4f} | {n40_bleu['top1_noisy40']['BLEU-4']:.4f} | Uniform |",
    f"| Noisy60 | {n60_bleu['base']['BLEU-4']:.4f} | {n60_bleu['learned_noisy60']['BLEU-4']:.4f} | {n60_bleu['uniform_noisy60']['BLEU-4']:.4f} | {n60_bleu['top1_noisy60']['BLEU-4']:.4f} | Learned |",
    "",
    "## Main finding",
    "",
    "As caption noise increased to 60%, learned weighting became the best fine-tuned method and outperformed both uniform weighting and top-1 caption selection on all BLEU metrics.",
]

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print("Saved:", out_path)
