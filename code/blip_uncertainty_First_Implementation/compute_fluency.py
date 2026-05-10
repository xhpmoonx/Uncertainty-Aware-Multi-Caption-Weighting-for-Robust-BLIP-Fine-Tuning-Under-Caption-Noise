import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

INPUT_PATH = "captions_with_agreement.json"
OUTPUT_PATH = "captions_with_agreement_fluency.json"
LM_NAME = "distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading captions...")
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
model = AutoModelForCausalLM.from_pretrained(LM_NAME).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nComputing fluency scores...")
for item in data["captions_with_agreement"]:
    text = item["text"]

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    avg_nll = outputs.loss.item()
    fluency_score = -avg_nll   # higher is better

    item["fluency_score"] = float(fluency_score)

print("\nFluency scores:")
for i, item in enumerate(data["captions_with_agreement"], start=1):
    print(f"{i}. {item['text']}")
    print(f"   fluency_score = {item['fluency_score']:.4f}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")
