"""Inference for a single image."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .model import expects_unnormalized_input
from .transforms import build_eval_transforms


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device,
    image_size: tuple[int, int] = (224, 224),
    probability_threshold: float = 0.5,
) -> dict[str, float | str]:
    """Classify one image and return readable FAKE/REAL probabilities."""
    model.eval()
    with Image.open(image_path) as image:
        tensor = build_eval_transforms(
            image_size,
            normalize=not expects_unnormalized_input(model),
        )(image.convert("RGB"))
    probability_real = torch.sigmoid(model(tensor.unsqueeze(0).to(device))).item()
    label = "REAL" if probability_real >= probability_threshold else "FAKE"
    confidence = probability_real if label == "REAL" else 1.0 - probability_real
    return {
        "label": label,
        "confidence": confidence,
        "probability_real": probability_real,
        "probability_fake": 1.0 - probability_real,
    }
