import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PRED_PATH = Path("results/coco500_val100_seed123_predictions_noisy60.json")
OUT_MD = Path("results/coco500_val100_seed123_qualitative_noisy60.md")

smooth = SmoothingFunction().method1

with open(PRED_PATH) as f:
    data = json.load(f)

def score(pred, refs):
    refs_tok = [r.lower().split() for r in refs]
    pred_tok = pred.lower().split()
    return sentence_bleu(
        refs_tok,
        pred_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

rows = []
for item in data:
    refs = item["references"]
    s_learned = score(item["learned_noisy60"], refs)
    s_uniform = score(item["uniform_noisy60"], refs)
    s_top1 = score(item["top1_noisy60"], refs)
    s_base = score(item["base"], refs)

    rows.append({
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "references": refs,
        "base": item["base"],
        "learned_noisy60": item["learned_noisy60"],
        "uniform_noisy60": item["uniform_noisy60"],
        "top1_noisy60": item["top1_noisy60"],
        "score_base": s_base,
        "score_learned": s_learned,
        "score_uniform": s_uniform,
        "score_top1": s_top1,
        "margin_vs_uniform": s_learned - s_uniform,
        "margin_vs_top1": s_learned - s_top1,
    })

wins = [
    r for r in rows
    if r["score_learned"] > r["score_uniform"] and r["score_learned"] > r["score_top1"]
]

wins.sort(key=lambda r: (r["margin_vs_uniform"] + r["margin_vs_top1"]), reverse=True)

lines = ["# Qualitative examples where learned_noisy60 beats uniform_noisy60 and top1_noisy60", ""]
for i, r in enumerate(wins[:10], start=1):
    lines.append(f"## Example {i}: image_id={r['image_id']}")
    lines.append(f"- image_path: `{r['image_path']}`")
    lines.append(f"- sentence_BLEU4 base: {r['score_base']:.4f}")
    lines.append(f"- sentence_BLEU4 learned_noisy60: {r['score_learned']:.4f}")
    lines.append(f"- sentence_BLEU4 uniform_noisy60: {r['score_uniform']:.4f}")
    lines.append(f"- sentence_BLEU4 top1_noisy60: {r['score_top1']:.4f}")
    lines.append("- references:")
    for ref in r["references"]:
        lines.append(f"  - {ref}")
    lines.append(f"- base: {r['base']}")
    lines.append(f"- learned_noisy60: {r['learned_noisy60']}")
    lines.append(f"- uniform_noisy60: {r['uniform_noisy60']}")
    lines.append(f"- top1_noisy60: {r['top1_noisy60']}")
    lines.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("Total learned wins:", len(wins))
print("Saved:", OUT_MD)
