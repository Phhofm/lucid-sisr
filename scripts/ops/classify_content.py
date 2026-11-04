"""Classify candidate tiles into broad content categories with CLIP
(zero-shot), to measure and fix content skew (e.g. grass/leaves dominance)."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

CATEGORIES = [
    "a close-up of grass or leaves",
    "a tree or forest",
    "flowers or plants",
    "a landscape with mountains or fields",
    "the sky, clouds or a sunset",
    "a building, architecture or a city street",
    "an indoor room or interior",
    "a person or a portrait",
    "an animal",
    "food or drink",
    "a vehicle, car or boat",
    "an object or product on a plain background",
    "text, a document or a screenshot",
    "a texture or an abstract pattern",
    "water, the sea or a river",
]


def classify(paths: list[Path], batch: int = 16) -> list[str]:
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [f"a photo of {c}" for c in CATEGORIES]
    text = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        tfeat = model.get_text_features(**text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            inp = proc(images=imgs, return_tensors="pt").to(device)
            f = model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = f @ tfeat.T
            for j in range(len(imgs)):
                out.append(CATEGORIES[int(sims[j].argmax())])
    return out
