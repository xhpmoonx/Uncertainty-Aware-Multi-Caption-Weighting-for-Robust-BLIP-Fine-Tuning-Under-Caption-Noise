import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PRED_PATH = Path("results/coco500_val100_seed123_predictions_noisy60_fixed_ablation.json")
OUT_MD = Path("results/qualitative_examples_agreement.md")

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
    s_ag = score(item["agreementonly_fixed_noisy60"], refs)
    s_fl = score(item["fluencyonly_fixed_noisy60"], refs)
    s_al = score(item["alignonly_fixed_noisy60"], refs)

    rows.append({
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "references": refs,
        "agreement": item["agreementonly_fixed_noisy60"],
        "fluency": item["fluencyonly_fixed_noisy60"],
        "align": item["alignonly_fixed_noisy60"],
        "score_agreement": s_ag,
        "score_fluency": s_fl,
        "score_align": s_al,
    })

wins = [r for r in rows if r["score_agreement"] > r["score_fluency"]]
wins.sort(key=lambda r: r["score_agreement"] - r["score_fluency"], reverse=True)

lines = ["# Agreement vs Fluency Qualitative Example", ""]

if wins:
    r = wins[0]
    lines.append(f"## Agreement intuition — image_id={r['image_id']}")
    lines.append(f"- image_path: `{r['image_path']}`")
    lines.append(f"- sentence_BLEU4 agreement: {r['score_agreement']:.4f}")
    lines.append(f"- sentence_BLEU4 fluency: {r['score_fluency']:.4f}")
    lines.append(f"- sentence_BLEU4 align: {r['score_align']:.4f}")
    lines.append("- references:")
    for ref in r["references"][:5]:
        lines.append(f"  - {ref}")
    lines.append(f"- agreementonly: {r['agreement']}")
    lines.append(f"- fluencyonly: {r['fluency']}")
    lines.append(f"- alignonly: {r['align']}")

with open(OUT_MD, "w") as f:
    f.write("\n".join(lines))

print("Saved:", OUT_MD)
print("wins:", len(wins))
