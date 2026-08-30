"""EfficientNet model creation and checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from .hybrid_model import HYBRID_MODEL_TYPE, HybridAIGCDetector


def get_device() -> torch.device:
    """Use an NVIDIA CUDA GPU when available, otherwise fall back to the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 and replace its output with one binary logit."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model.to(device)


def expects_unnormalized_input(model: nn.Module) -> bool:
    """Return whether a model performs its own spatial input normalization."""
    return bool(getattr(model, "expects_unnormalized_input", False))


def load_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load saved model weights, move them to the device, and enable inference mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = checkpoint.get("model_type") if isinstance(checkpoint, Mapping) else None
    if model_type == HYBRID_MODEL_TYPE:
        model = HybridAIGCDetector(pretrained_spatial=False).to(device)
    elif model_type in (None, "efficientnet"):
        model = build_model(device, pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
