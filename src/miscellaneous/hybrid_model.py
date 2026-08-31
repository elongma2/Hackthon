"""Lightweight EfficientNet and FFT hybrid detector."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from ..transforms import IMAGENET_MEAN, IMAGENET_STD


HYBRID_MODEL_TYPE = "hybrid"
SPATIAL_FEATURE_DIM = 1280
FREQUENCY_FEATURE_DIM = 256
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.3


class FFTPreprocessor(nn.Module):
    """Convert an RGB image batch into normalized log-magnitude FFT spectra."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "luminance_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute FFT preprocessing in float32 even inside an autocast region."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "FFT preprocessing expects RGB tensors shaped [batch, 3, height, width]."
            )
        with torch.autocast(device_type=images.device.type, enabled=False):
            frequency_input = images.float()
            grayscale = (frequency_input * self.luminance_weights.float()).sum(
                dim=1,
                keepdim=True,
            )
            spectrum = torch.fft.fft2(grayscale, dim=(-2, -1), norm="ortho")
            shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))
            log_magnitude = torch.log1p(torch.abs(shifted))
            mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
            std = log_magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
            normalized = (log_magnitude - mean) / std.clamp_min(self.epsilon)
        return normalized


class FrequencyBranch(nn.Module):
    """Extract a compact representation from one normalized FFT spectrum."""

    def __init__(self, output_dim: int = FREQUENCY_FEATURE_DIM) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.preprocessor = FFTPreprocessor()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spectrum = self.preprocessor(images)
        return torch.flatten(self.cnn(spectrum), 1)


class HybridClassifier(nn.Module):
    """Own the complete trainable FFT branch and spatial-frequency fusion head."""

    def __init__(
        self,
        spatial_dim: int = SPATIAL_FEATURE_DIM,
        frequency_dim: int = FREQUENCY_FEATURE_DIM,
        hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = FUSION_DROPOUT,
    ) -> None:
        super().__init__()
        self.frequency_branch = FrequencyBranch(frequency_dim)
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + frequency_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> torch.Tensor:
        frequency_features = self.frequency_branch(raw_images)
        combined = torch.cat((spatial_features, frequency_features), dim=1)
        return self.fusion(combined)


class HybridAIGCDetector(nn.Module):
    """Fuse EfficientNet spatial features with lightweight FFT features."""

    model_type = HYBRID_MODEL_TYPE
    expects_unnormalized_input = True
    spatial_feature_dim = SPATIAL_FEATURE_DIM
    frequency_feature_dim = FREQUENCY_FEATURE_DIM
    fusion_hidden_dim = FUSION_HIDDEN_DIM
    fusion_dropout = FUSION_DROPOUT

    def __init__(self, pretrained_spatial: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_spatial else None
        efficientnet = models.efficientnet_b0(weights=weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.classifier = HybridClassifier()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def extract_spatial_features(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize raw RGB tensors and return pooled EfficientNet features."""
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        feature_map = self.features(normalized)
        return torch.flatten(self.avgpool(feature_map), 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spatial_features = self.extract_spatial_features(images)
        return self.classifier(spatial_features, images)


def _checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state dictionary from a checkpoint envelope or plain mapping."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Spatial checkpoint must contain a state-dict mapping.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("Spatial checkpoint state_dict must map string keys to tensors.")
    return state


def load_spatial_checkpoint(
    model: HybridAIGCDetector,
    checkpoint_path: str | Path,
    device: torch.device,
) -> int:
    """Strictly initialize every spatial state entry from an EfficientNet checkpoint."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = _checkpoint_state_dict(checkpoint)
    unsupported = sorted(
        key
        for key in state
        if not key.startswith("features.") and not key.startswith("classifier.")
    )
    if unsupported:
        raise ValueError(
            "Spatial checkpoint contains unsupported state prefixes: "
            + ", ".join(unsupported[:10])
        )

    extracted = {
        key.removeprefix("features."): value
        for key, value in state.items()
        if key.startswith("features.")
    }
    expected = model.features.state_dict()
    missing = sorted(set(expected) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing[:10]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:10]))
        raise ValueError("Spatial checkpoint keys do not match exactly (" + "; ".join(details) + ").")

    mismatched = [
        f"{key}: expected {tuple(expected[key].shape)}, found {tuple(extracted[key].shape)}"
        for key in expected
        if expected[key].shape != extracted[key].shape
    ]
    if mismatched:
        raise ValueError(
            "Spatial checkpoint tensor shapes do not match: " + "; ".join(mismatched[:10])
        )

    model.features.load_state_dict(extracted, strict=True)
    ignored = sorted(key for key in state if key.startswith("classifier."))
    print(f"Loaded {len(expected):,} spatial state entries from {path.resolve()}")
    if ignored:
        print("Ignored source classifier entries: " + ", ".join(ignored))
    return len(expected)


def build_hybrid_model(
    device: torch.device,
    spatial_checkpoint: str | Path | None = None,
) -> HybridAIGCDetector:
    """Build the hybrid, using a strict detector checkpoint or ImageNet spatial weights."""
    if spatial_checkpoint is None:
        print("Hybrid spatial initialization: ImageNet EfficientNet-B0 weights")
        model = HybridAIGCDetector(pretrained_spatial=True)
    else:
        model = HybridAIGCDetector(pretrained_spatial=False)
        load_spatial_checkpoint(model, spatial_checkpoint, device)
    return model.to(device)
