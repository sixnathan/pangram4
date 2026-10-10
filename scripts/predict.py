"""Minimal inference: score a text with the trained stage-1 model.
Run from scripts/:  uv run python predict.py "some text here"
"""
import sys

import torch
from transformers import AutoTokenizer

from model import load_model

model = load_model()
model.load_state_dict(torch.load("../checkpoints/stage1.pt"))
model.eval()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

text = sys.argv[1] if len(sys.argv) > 1 else input("text to check: ")
enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")

with torch.no_grad():
    seg, hum = model(enc["input_ids"], enc["attention_mask"])

bucket = int(seg.argmax(dim=1))
label = ["human", "ai", "humanized_ai", "mixed"][int(hum.argmax(dim=1))]
print(f"f_ai bucket: {bucket}/14  (~{bucket / 15:.0%} AI-written)")
print(f"humanizer : {label}")
