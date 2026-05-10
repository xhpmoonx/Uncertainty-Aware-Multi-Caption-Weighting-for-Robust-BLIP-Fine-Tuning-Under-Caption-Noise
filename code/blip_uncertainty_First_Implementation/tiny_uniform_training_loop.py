import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

INPUT_PATH = "mini_dataset_scored.json"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

# keep these the same as your learned-weight run
MAX_IMAGES = 3
EPOCHS = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading scored dataset...")
with open(INPUT_PATH, "r") as f:
    dataset = json.load(f)

dataset = dataset[:MAX_IMAGES]
print(f"Training on {len(dataset)} images")

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
blip = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
blip.train()

optimizer = torch.optim.Adam([
    {"params": blip.parameters(), "lr": 1e-5},
])

epoch_losses = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    for idx, record in enumerate(dataset, start=1):
        image_path = record["image_path"]
        captions = record["captions"]
        image = Image.open(image_path).convert("RGB")

        k = len(captions)
        weights = torch.full((k,), 1.0 / k, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        weighted_parts = []
        per_caption_losses = []

        for i, cap in enumerate(captions):
            text = cap["text"]

            inputs = processor(images=image, text=text, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = blip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss_i = outputs.loss
            per_caption_losses.append(loss_i.item())
            weighted_parts.append(weights[i] * loss_i)

        uniform_loss = torch.stack(weighted_parts).sum()
        uniform_loss.backward()
        optimizer.step()

        running_loss += uniform_loss.item()

        print(f"[{idx}/{len(dataset)}] {image_path}")
        print(f"  uniform_loss = {uniform_loss.item():.6f}")
        for j in range(k):
            print(f"    {j+1}. weight={weights[j].item():.4f} loss={per_caption_losses[j]:.4f}")

    avg_loss = running_loss / len(dataset)
    epoch_losses.append(avg_loss)
    print(f"\nEpoch {epoch+1} average loss = {avg_loss:.6f}")

    torch.save(blip.state_dict(), f"blip_uniform_epoch_{epoch+1}.pt")

with open("tiny_uniform_training_log.json", "w") as f:
    json.dump({"epoch_losses": epoch_losses}, f, indent=2)

print("\nUniform baseline training finished.")
print("Saved:")
print("  tiny_uniform_training_log.json")
for epoch in range(EPOCHS):
    print(f"  blip_uniform_epoch_{epoch+1}.pt")
