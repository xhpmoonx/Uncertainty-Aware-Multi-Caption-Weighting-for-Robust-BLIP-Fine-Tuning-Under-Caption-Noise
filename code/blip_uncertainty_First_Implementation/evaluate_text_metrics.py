import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

METHODS = ["base", "learned", "uniform", "top1"]


def simple_words(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def distinct_n(captions, n=1):
    all_ngrams = []
    for text in captions:
        toks = simple_words(text)
        if n == 1:
            all_ngrams.extend(toks)
        else:
            all_ngrams.extend(list(zip(*[toks[i:] for i in range(n)])))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repeated_bigram_rate(captions):
    flagged = 0
    total = 0
    for text in captions:
        toks = simple_words(text)
        bgs = bigrams(toks)
        total += 1
        if len(bgs) != len(set(bgs)):
            flagged += 1
    return flagged / total if total else 0.0


def avg_len(captions):
    if not captions:
        return 0.0
    return sum(len(simple_words(x)) for x in captions) / len(captions)


def find_records(data):
    preferred_keys = ["results", "records", "items", "examples", "per_image"]
    for k in preferred_keys:
        if k in data and isinstance(data[k], list):
            return data[k]

    # fallback: find first list of dicts that looks caption-like
    for _, v in data.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            sample = v[0]
            if any((m in sample) or (f"{m}_caption" in sample) for m in METHODS):
                return v

    raise ValueError("Could not find per-image records in JSON.")


def extract_caption(rec, method):
    # case 1: nested dict like rec["learned"]["caption"]
    if method in rec:
        v = rec[method]
        if isinstance(v, dict):
            for key in ["caption", "text", "generated_caption", "caption_text"]:
                if key in v and isinstance(v[key], str):
                    return v[key]
        elif isinstance(v, str):
            return v

    # case 2: flat key like rec["learned_caption"]
    for key in [f"{method}_caption", f"{method}_text", f"{method}_generated_caption"]:
        if key in rec and isinstance(rec[key], str):
            return rec[key]

    return None


@torch.no_grad()
def compute_ppl_scores(captions, model_name="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses = []
    ppls = []

    for text in captions:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.item())
        ppl = math.exp(min(loss, 20.0))
        losses.append(loss)
        ppls.append(ppl)

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_ppl = sum(ppls) / len(ppls) if ppls else 0.0
    return avg_loss, avg_ppl


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_text_metrics.py <holdout_alignment.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with open(input_path, "r") as f:
        data = json.load(f)

    records = find_records(data)

    captions_by_method = defaultdict(list)
    for rec in records:
        for method in METHODS:
            c = extract_caption(rec, method)
            if c is not None:
                captions_by_method[method].append(c)

    summary = {}

    print("Loading GPT-2 for fluency scoring...")
    for method in METHODS:
        caps = captions_by_method[method]
        lm_loss, ppl = compute_ppl_scores(caps)

        summary[method] = {
            "num_captions": len(caps),
            "avg_words": avg_len(caps),
            "distinct_1": distinct_n(caps, 1),
            "distinct_2": distinct_n(caps, 2),
            "repeated_bigram_rate": repeated_bigram_rate(caps),
            "avg_gpt2_nll": lm_loss,
            "avg_gpt2_ppl": ppl,
        }

    out_json = input_path.with_name(input_path.stem + "_text_metrics.json")
    out_md = input_path.with_name(input_path.stem + "_text_metrics.md")

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append(f"# Text Metrics for {input_path.name}\n")
    lines.append("| Method | Avg words | Dist-1 | Dist-2 | Repeat bigram rate | GPT2 NLL | GPT2 PPL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        s = summary[method]
        lines.append(
            f"| {method} | "
            f"{s['avg_words']:.3f} | "
            f"{s['distinct_1']:.6f} | "
            f"{s['distinct_2']:.6f} | "
            f"{s['repeated_bigram_rate']:.6f} | "
            f"{s['avg_gpt2_nll']:.6f} | "
            f"{s['avg_gpt2_ppl']:.6f} |"
        )

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== TEXT METRICS SUMMARY ===")
    for method in METHODS:
        s = summary[method]
        print(f"\n[{method}]")
        print(f"  num_captions          : {s['num_captions']}")
        print(f"  avg_words             : {s['avg_words']:.6f}")
        print(f"  distinct_1            : {s['distinct_1']:.6f}")
        print(f"  distinct_2            : {s['distinct_2']:.6f}")
        print(f"  repeated_bigram_rate  : {s['repeated_bigram_rate']:.6f}")
        print(f"  avg_gpt2_nll          : {s['avg_gpt2_nll']:.6f}")
        print(f"  avg_gpt2_ppl          : {s['avg_gpt2_ppl']:.6f}")

    print(f"\nSaved {out_json}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
