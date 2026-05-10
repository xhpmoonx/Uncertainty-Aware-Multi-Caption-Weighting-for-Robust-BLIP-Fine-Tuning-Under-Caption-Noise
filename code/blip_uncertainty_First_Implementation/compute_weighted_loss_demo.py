import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_PATH = "mini_dataset_scored.json"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading scored dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

# Take the first image as a demo
record = dataset[0]
image_path = record["image_path"]
captions = record["captions"]

print(f"Using image: {image_path}")

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

image = Image.open(image_path).convert("RGB")

per_caption_losses = []
weights = []

print("\nPer-caption losses:")
for i, cap in enumerate(captions, start=1):
    text = cap["text"]
    weight = float(cap["softmax_weight"])

    inputs = processor(images=image, text=text, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss_i = outputs.loss.item()

    per_caption_losses.append(loss_i)
    weights.append(weight)

    print(f"{i}. {text}")
    print(f"   softmax_weight = {weight:.6f}")
    print(f"   caption_loss   = {loss_i:.6f}")
    print(f"   weighted_part  = {weight * loss_i:.6f}")

weighted_loss = sum(w * l for w, l in zip(weights, per_caption_losses))
uniform_loss = sum(per_caption_losses) / len(per_caption_losses)

print("\nFinal losses:")
print(f"Weighted caption loss = {weighted_loss:.6f}")
print(f"Uniform average loss  = {uniform_loss:.6f}")
