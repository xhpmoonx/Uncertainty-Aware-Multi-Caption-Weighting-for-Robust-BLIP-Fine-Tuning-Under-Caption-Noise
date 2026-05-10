import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"
VAL_PATH = Path("coco_work/subsets/coco_val_100_raw_local_seed123.json")
OUT_PATH = Path("results/coco500_val100_seed123_predictions_noisy60_pairwise.json")

CHECKPOINTS = {
    "base": None,
    "learned_noisy60": Path("results/coco500_noisy60_10ep_seed123_learned_learned_blip.pt"),
    "uniform_noisy60": Path("results/coco500_noisy60_10ep_seed123_uniform_uniform_blip.pt"),
    "top1_noisy60": Path("results/coco500_noisy60_10ep_seed123_top1_top1_blip.pt"),
    "hardfilter_noisy60": Path("results/coco500_noisy60_hardfilter020_10ep_seed123_uniform_blip.pt"),
    "align_fluency_noisy60": Path("results/coco500_noisy60_align_fluency_10ep_seed123_learned_blip.pt"),
    "align_agreement_noisy60": Path("results/coco500_noisy60_align_agreement_10ep_seed123_learned_blip.pt"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = BlipProcessor.from_pretrained(MODEL_NAME)

with open(VAL_PATH, "r") as f:
    val_data = json.load(f)

predictions = [
    {
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "references": item["generated_captions"],
    }
    for item in val_data
]

def load_model(checkpoint_path):
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model

for model_name, checkpoint_path in CHECKPOINTS.items():
    print(f"\nLoading: {model_name}")
    model = load_model(checkpoint_path)

    for item in predictions:
        image = Image.open(item["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=30, num_beams=3)

        item[model_name] = processor.decode(out_ids[0], skip_special_tokens=True).strip()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT_PATH, "w") as f:
    json.dump(predictions, f, indent=2)

print("Saved:", OUT_PATH)
