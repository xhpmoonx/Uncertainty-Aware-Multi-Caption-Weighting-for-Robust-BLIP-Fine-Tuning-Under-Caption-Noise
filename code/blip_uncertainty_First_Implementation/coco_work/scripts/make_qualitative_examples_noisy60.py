import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PRED_PATH = Path("results/coco500_val100_seed123_predictions_noisy60_hardfilter.json")
OUT_MD = Path("results/qualitative_examples_noisy60.md")

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

    s_base = score(item["base"], refs)
    s_learned = score(item["learned_noisy60"], refs)
    s_uniform = score(item["uniform_noisy60"], refs)
    s_top1 = score(item["top1_noisy60"], refs)
    s_hard = score(item["hardfilter_noisy60"], refs)

    rows.append({
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "references": refs,
        "base": item["base"],
        "learned": item["learned_noisy60"],
        "uniform": item["uniform_noisy60"],
        "top1": item["top1_noisy60"],
        "hardfilter": item["hardfilter_noisy60"],
        "score_base": s_base,
        "score_learned": s_learned,
        "score_uniform": s_uniform,
        "score_top1": s_top1,
        "score_hard": s_hard,
    })

learned_wins = [
    r for r in rows
    if r["score_learned"] > r["score_uniform"] and r["score_learned"] > r["score_top1"]
]
learned_wins.sort(key=lambda r: r["score_learned"] - max(r["score_uniform"], r["score_top1"]), reverse=True)

hardfilter_wins = [
    r for r in rows
    if r["score_hard"] > r["score_learned"]
]
hardfilter_wins.sort(key=lambda r: r["score_hard"] - r["score_learned"], reverse=True)

failure_cases = [
    r for r in rows
    if r["score_base"] > r["score_learned"]
]
failure_cases.sort(key=lambda r: r["score_base"] - r["score_learned"], reverse=True)

lines = ["# Qualitative Examples (Noisy60)", ""]

def add_example(title, r):
    lines.append(f"## {title} — image_id={r['image_id']}")
    lines.append(f"- image_path: `{r['image_path']}`")
    lines.append(f"- sentence_BLEU4 base: {r['score_base']:.4f}")
    lines.append(f"- sentence_BLEU4 learned: {r['score_learned']:.4f}")
    lines.append(f"- sentence_BLEU4 uniform: {r['score_uniform']:.4f}")
    lines.append(f"- sentence_BLEU4 top1: {r['score_top1']:.4f}")
    lines.append(f"- sentence_BLEU4 hardfilter: {r['score_hard']:.4f}")
    lines.append("- references:")
    for ref in r["references"][:5]:
        lines.append(f"  - {ref}")
    lines.append(f"- base: {r['base']}")
    lines.append(f"- learned: {r['learned']}")
    lines.append(f"- uniform: {r['uniform']}")
    lines.append(f"- top1: {r['top1']}")
    lines.append(f"- hardfilter: {r['hardfilter']}")
    lines.append("")

if learned_wins:
    add_example("Learned win #1", learned_wins[0])
if len(learned_wins) > 1:
    add_example("Learned win #2", learned_wins[1])
if hardfilter_wins:
    add_example("Hard-filter win", hardfilter_wins[0])
if failure_cases:
    add_example("Failure case", failure_cases[0])

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("Saved:", OUT_MD)
print("learned_wins:", len(learned_wins))
print("hardfilter_wins:", len(hardfilter_wins))
print("failure_cases:", len(failure_cases))
