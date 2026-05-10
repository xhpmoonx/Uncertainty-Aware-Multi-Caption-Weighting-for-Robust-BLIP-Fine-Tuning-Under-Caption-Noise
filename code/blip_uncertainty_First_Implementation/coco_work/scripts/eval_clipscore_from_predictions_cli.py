import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

LOCAL_CLIP_PATH = "./.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return " ".join(x.split()).strip()
    if isinstance(x, list):
        return " ".join(str(v) for v in x).strip()
    return str(x).strip()


def infer_tag(pred_path: Path) -> str:
    name = pred_path.name
    suffix = "_predictions.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return pred_path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    tag = infer_tag(pred_path)

    with open(pred_path) as f:
        data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_PATH)
    model = CLIPModel.from_pretrained(LOCAL_CLIP_PATH).to(device)
    model.eval()

    scores = []
    skipped = 0

    for i, row in enumerate(data):
        image_path = row.get("image_path")
        if not image_path or not Path(image_path).exists():
            skipped += 1
            continue

        # Prefer prediction; fallback only if needed
        text = row.get("prediction", None)
        if text is None:
            text = row.get("caption", None)
        if text is None:
            text = row.get("references", None)

        text = normalize_text(text)
        if text == "":
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
            score = (image_embeds * text_embeds).sum(dim=-1).item()

        scores.append(score)

        if (i + 1) % 20 == 0 or i == len(data) - 1:
            print(f"[{i+1}/{len(data)}] score={score:.6f}")

    mean_cosine = float(sum(scores) / len(scores)) if scores else float("nan")

    out_json = pred_path.with_name(f"{tag}_predictions_clipscore.json")
    out_md = pred_path.with_name(f"{tag}_predictions_clipscore.md")

    payload = {
        tag: {
            "mean_cosine": mean_cosine,
            "n": len(scores),
            "skipped": skipped,
        }
    }

    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_md, "w") as f:
        f.write(f"{tag}\n")
        f.write(f"mean_cosine: {mean_cosine}\n")
        f.write(f"n: {len(scores)}\n")
        f.write(f"skipped: {skipped}\n")

    print("Saved:", out_json)
    print("Saved:", out_md)
    print(tag)
    print("mean_cosine:", mean_cosine)


if __name__ == "__main__":
    main()
