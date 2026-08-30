"""EfficientNet model creation and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


def get_device() -> torch.device:
    """Use an NVIDIA CUDA GPU when available, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 and replace its output with one binary logit."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(device)


def load_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load saved model weights, move them to the device, and enable inference mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(device, pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
