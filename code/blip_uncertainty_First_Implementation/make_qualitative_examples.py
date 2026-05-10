import json
from pathlib import Path

METHODS = ["base", "learned", "uniform", "top1"]


def find_records(data):
    for k in ["results", "records", "items", "examples", "per_image"]:
        if k in data and isinstance(data[k], list):
            return data[k]
    for _, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise ValueError("Could not find per-image records in JSON.")


def get_image_path(rec):
    for k in ["image_path", "image", "path", "file", "image_file"]:
        if k in rec and isinstance(rec[k], str):
            return rec[k]
    return "unknown_image"


def extract_caption(rec, method):
    if method in rec:
        v = rec[method]
        if isinstance(v, dict):
            for key in ["caption", "text", "generated_caption", "caption_text"]:
                if key in v and isinstance(v[key], str):
                    return v[key]
        elif isinstance(v, str):
            return v

    for key in [f"{method}_caption", f"{method}_text", f"{method}_generated_caption"]:
        if key in rec and isinstance(rec[key], str):
            return rec[key]

    return None


def extract_alignment(rec, method):
    if method in rec:
        v = rec[method]
        if isinstance(v, dict):
            for key in ["alignment", "score", "clipscore", "alignment_score"]:
                if key in v:
                    try:
                        return float(v[key])
                    except Exception:
                        pass

    for key in [
        f"{method}_alignment",
        f"{method}_score",
        f"{method}_clipscore",
        f"{method}_alignment_score",
    ]:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                pass

    return None


def clean_records(records):
    out = []
    for rec in records:
        item = {"image_path": get_image_path(rec)}
        ok = True
        for m in METHODS:
            cap = extract_caption(rec, m)
            ali = extract_alignment(rec, m)
            if cap is None or ali is None:
                ok = False
                break
            item[m] = {"caption": cap, "alignment": ali}
        if ok:
            out.append(item)
    return out


def pick_unique(items, seen, n):
    picked = []
    for x in items:
        if x["image_path"] not in seen:
            picked.append(x)
            seen.add(x["image_path"])
        if len(picked) >= n:
            break
    return picked


def main():
    input_path = Path(
        "results/scaled_run_200img_10ep_seed123/200img_10ep_seed123_holdout_alignment.json"
    )
    output_path = Path(
        "results/scaled_run_200img_10ep_seed123/200img_10ep_seed123_qualitative_examples.md"
    )

    with open(input_path, "r") as f:
        data = json.load(f)

    records = clean_records(find_records(data))

    learned_best = sorted(
        [r for r in records if r["learned"]["alignment"] > max(r["base"]["alignment"], r["uniform"]["alignment"], r["top1"]["alignment"])],
        key=lambda r: r["learned"]["alignment"] - max(r["base"]["alignment"], r["uniform"]["alignment"], r["top1"]["alignment"]),
        reverse=True,
    )

    top1_best = sorted(
        [r for r in records if r["top1"]["alignment"] > max(r["base"]["alignment"], r["uniform"]["alignment"], r["learned"]["alignment"])],
        key=lambda r: r["top1"]["alignment"] - max(r["base"]["alignment"], r["uniform"]["alignment"], r["learned"]["alignment"]),
        reverse=True,
    )

    close_cases = sorted(
        records,
        key=lambda r: max(r[m]["alignment"] for m in METHODS) - min(r[m]["alignment"] for m in METHODS),
    )

    seen = set()
    chosen = []
    chosen += pick_unique(learned_best, seen, 4)
    chosen += pick_unique(top1_best, seen, 3)
    chosen += pick_unique(close_cases, seen, 3)

    lines = []
    lines.append("# Qualitative Caption Examples\n")
    lines.append("These examples were selected from the holdout set to show representative differences across methods.\n")

    for i, r in enumerate(chosen, start=1):
        lines.append(f"## Example {i}: `{r['image_path']}`\n")
        for m in METHODS:
            lines.append(f"- **{m}**: {r[m]['caption']}")
            lines.append(f"  - alignment: {r[m]['alignment']:.6f}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
