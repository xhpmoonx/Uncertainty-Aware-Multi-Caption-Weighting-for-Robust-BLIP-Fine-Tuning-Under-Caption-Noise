import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

INPUT_PATH = "captions_with_agreement_fluency.json"
OUTPUT_PATH = "captions_with_all_signals.json"

# BLIP ITM / retrieval model for alignment scoring
MODEL_NAME = "Salesforce/blip-itm-base-coco"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading JSON...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

image_path = data["image_path"]
captions = data["captions_with_agreement"]

print("Loading image...")
image = Image.open(image_path).convert("RGB")

print("Loading BLIP ITM model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("\nComputing alignment scores...")
for item in captions:
    text = item["text"]

    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_itm_head=True)

    # Most common BLIP ITM output path
    if hasattr(outputs, "itm_score") and outputs.itm_score is not None:
        logits = outputs.itm_score
        alignment_score = torch.softmax(logits, dim=-1)[0, 1].item()

    # Fallback in case model version exposes logits differently
    elif hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        if logits.shape[-1] == 2:
            alignment_score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            alignment_score = logits.squeeze().item()

    else:
        raise ValueError("Could not find ITM logits in model output.")

    item["alignment_score"] = float(alignment_score)

print("\nAlignment scores:")
for i, item in enumerate(captions, start=1):
    print(f"{i}. {item['text']}")
    print(f"   alignment_score = {item['alignment_score']:.4f}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
