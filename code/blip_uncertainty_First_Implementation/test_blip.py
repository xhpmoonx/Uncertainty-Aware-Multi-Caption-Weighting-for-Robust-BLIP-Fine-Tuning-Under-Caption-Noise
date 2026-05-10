from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

IMAGE_PATH = "test.jpg"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading processor and model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

print("Loading image...")
image = Image.open(IMAGE_PATH).convert("RGB")

print("Running caption generation...")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30)

caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated caption:")
print(caption)

