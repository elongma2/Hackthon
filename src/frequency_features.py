"""Shared final-model FFT branches and strict spatial checkpoint validation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


DEFAULT_FREQUENCY_SCALE = 0.25
DEFAULT_FREQUENCY_BRANCH_DROPOUT = 0.20
DEFAULT_FREQUENCY_MASK_PROB = 0.0
SPATIAL_FEATURE_DIM = 1280
MAGNITUDE_FEATURE_DIM = 128
PHASE_FEATURE_DIM = 128
FREQUENCY_HIDDEN_DIM = 64
FREQUENCY_DROPOUT = 0.5
FREQUENCY_LEARNING_RATE = 5e-5
SPATIAL_LEARNING_RATE = 1e-5


class SharedFFTPreprocessor(nn.Module):
    """Derive normalized magnitude and wrap-safe sin/cos phase from one FFT."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        """Configure the numerical floor used for per-image standardization."""
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "luminance_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 magnitude [B,1,H,W] and sin/cos phase [B,2,H,W]."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("FFT preprocessing expects [batch,3,height,width] RGB tensors.")
        with torch.autocast(device_type=images.device.type, enabled=False):
            luminance = (images.float() * self.luminance_weights.float()).sum(dim=1, keepdim=True)
            fft = torch.fft.fft2(luminance, dim=(-2, -1), norm="ortho")
            shifted = torch.fft.fftshift(fft, dim=(-2, -1))
            log_magnitude = torch.log1p(torch.abs(shifted))
            mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
            std = log_magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
            magnitude = (log_magnitude - mean) / std.clamp_min(self.epsilon)
            phase = torch.angle(shifted)
            phase_input = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        return magnitude, phase_input


def _build_spectrum_cnn(input_channels: int, output_dim: int) -> nn.Sequential:
    """Build the lightweight three-convolution forensic feature extractor."""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(64, output_dim, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
    )


class MagnitudeBranch(nn.Module):
    """Extract magnitude features with optional training-only spectrum masking."""

    def __init__(self, output_dim: int = MAGNITUDE_FEATURE_DIM, mask_probability: float = 0.0) -> None:
        """Configure output width and per-sample mild mask probability."""
        super().__init__()
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("frequency mask probability must be between 0 and 1.")
        self.output_dim = output_dim
        self.mask_probability = float(mask_probability)
        self.cnn = _build_spectrum_cnn(1, output_dim)

    def apply_frequency_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Zero one small random region for selected training samples only."""
        if not self.training or self.mask_probability == 0.0:
            return spectrum
        height, width = spectrum.shape[-2:]
        if height < 2 or width < 2:
            return spectrum
        selected = torch.rand(spectrum.shape[0], device=spectrum.device) < self.mask_probability
        if not bool(selected.any()):
            return spectrum
        mask_height = min(max(1, round(height * 0.10)), height - 1)
        mask_width = min(max(1, round(width * 0.10)), width - 1)
        masked = spectrum.clone()
        for index in selected.nonzero(as_tuple=False).flatten().tolist():
            top = int(torch.randint(0, height - mask_height + 1, (1,), device=spectrum.device).item())
            left = int(torch.randint(0, width - mask_width + 1, (1,), device=spectrum.device).item())
            masked[index, :, top : top + mask_height, left : left + mask_width] = 0.0
        return masked

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Return one flattened magnitude feature vector per image."""
        return torch.flatten(self.cnn(self.apply_frequency_mask(magnitude)), 1)


class PhaseBranch(nn.Module):
    """Extract phase features from the bounded sine/cosine representation."""

    def __init__(self, output_dim: int = PHASE_FEATURE_DIM) -> None:
        """Configure the phase feature width."""
        super().__init__()
        self.output_dim = output_dim
        self.cnn = _build_spectrum_cnn(2, output_dim)

    def forward(self, phase_input: torch.Tensor) -> torch.Tensor:
        """Return one flattened phase feature vector per image."""
        return torch.flatten(self.cnn(phase_input), 1)


def build_frequency_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Build the small forensic feature-to-logit prediction head."""
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))


def checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state mapping from a supported checkpoint envelope."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Spatial checkpoint must contain a state-dict mapping.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("Spatial checkpoint state_dict must map string keys to tensors.")
    return state


def validated_feature_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Require exact EfficientNet feature keys and shapes before changing the model."""
    unsupported = sorted(key for key in state if not key.startswith("features.") and not key.startswith("classifier."))
    if unsupported:
        raise ValueError("Spatial checkpoint contains unsupported state prefixes: " + ", ".join(unsupported[:10]))
    extracted = {key.removeprefix("features."): value for key, value in state.items() if key.startswith("features.")}
    expected = model.features.state_dict()
    missing = sorted(set(expected) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected))
    if missing or unexpected:
        details = (["missing: " + ", ".join(missing[:10])] if missing else []) + (["unexpected: " + ", ".join(unexpected[:10])] if unexpected else [])
        raise ValueError("Spatial checkpoint keys do not match exactly (" + "; ".join(details) + ").")
    mismatched = [f"{key}: expected {tuple(expected[key].shape)}, found {tuple(extracted[key].shape)}" for key in expected if expected[key].shape != extracted[key].shape]
    if mismatched:
        raise ValueError("Spatial checkpoint tensor shapes do not match: " + "; ".join(mismatched[:10]))
    return extracted


def validated_spatial_classifier_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor] | None, str]:
    """Return one complete compatible binary classifier pair or a random fallback."""
    candidates = {
        "EfficientNet classifier.1": ("classifier.1.weight", "classifier.1.bias"),
        "hybrid residual spatial classifier": ("classifier.spatial_classifier.weight", "classifier.spatial_classifier.bias"),
    }
    compatible: list[tuple[str, dict[str, torch.Tensor]]] = []
    incompatible: list[str] = []
    for source_name, (weight_key, bias_key) in candidates.items():
        has_weight, has_bias = weight_key in state, bias_key in state
        if has_weight != has_bias:
            raise ValueError(f"Spatial checkpoint contains a partial {source_name}: expected both {weight_key} and {bias_key}.")
        if not has_weight:
            continue
        weight, bias = state[weight_key], state[bias_key]
        if weight.shape == torch.Size([1, model.spatial_feature_dim]) and bias.shape == torch.Size([1]):
            compatible.append((source_name, {"weight": weight, "bias": bias}))
        else:
            incompatible.append(f"{source_name} has weight {tuple(weight.shape)} and bias {tuple(bias.shape)}")
    if len(compatible) > 1:
        raise ValueError("Spatial checkpoint contains multiple compatible binary classifiers.")
    if compatible:
        return compatible[0][1], compatible[0][0]
    if incompatible:
        print("No compatible binary spatial classifier was loaded: " + "; ".join(incompatible))
    else:
        ignored = sorted(key for key in state if key.startswith("classifier."))
        print("No compatible spatial classifier was present; ignored classifier entries: " + ", ".join(ignored) if ignored else "Spatial checkpoint contains features only; spatial classifier is random.")
    return None, "random"
