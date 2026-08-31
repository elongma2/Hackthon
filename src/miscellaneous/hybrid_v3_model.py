"""EfficientNet detector with controlled magnitude and phase FFT residuals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from .hybrid_v2_model import (
    DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    DEFAULT_FREQUENCY_MASK_PROB,
    DEFAULT_FREQUENCY_SCALE,
    HYBRID_V2_MODEL_TYPE,
    V2_FREQUENCY_DROPOUT,
    V2_FREQUENCY_FEATURE_DIM,
    V2_FREQUENCY_HIDDEN_DIM,
    V2_FREQUENCY_LEARNING_RATE,
    V2_SPATIAL_FEATURE_DIM,
    V2_SPATIAL_LEARNING_RATE,
    _checkpoint_state_dict,
    _validated_feature_state,
)
from ..transforms import IMAGENET_MEAN, IMAGENET_STD


HYBRID_V3_MODEL_TYPE = "hybrid_v3"
V3_SPATIAL_FEATURE_DIM = V2_SPATIAL_FEATURE_DIM
V3_MAGNITUDE_FEATURE_DIM = V2_FREQUENCY_FEATURE_DIM
V3_PHASE_FEATURE_DIM = 128
V3_FREQUENCY_HIDDEN_DIM = V2_FREQUENCY_HIDDEN_DIM
V3_FREQUENCY_DROPOUT = V2_FREQUENCY_DROPOUT
V3_FREQUENCY_LEARNING_RATE = V2_FREQUENCY_LEARNING_RATE
V3_SPATIAL_LEARNING_RATE = V2_SPATIAL_LEARNING_RATE
DEFAULT_MAGNITUDE_WEIGHT = 0.5
DEFAULT_PHASE_WEIGHT = 0.5


@dataclass(frozen=True)
class V3InitializationResult:
    """Describe the strict weights initialized before V3 training."""

    feature_entry_count: int
    classifier_loaded: bool
    classifier_source: str
    magnitude_loaded: bool = False


class SharedFFTPreprocessor(nn.Module):
    """Derive normalized magnitude and wrapped-safe phase from one FFT."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "luminance_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 magnitude `[B,1,H,W]` and phase `[B,2,H,W]`."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "FFT preprocessing expects RGB tensors shaped [batch, 3, height, width]."
            )
        with torch.autocast(device_type=images.device.type, enabled=False):
            frequency_input = images.float()
            luminance = (frequency_input * self.luminance_weights.float()).sum(
                dim=1,
                keepdim=True,
            )
            fft = torch.fft.fft2(luminance, dim=(-2, -1), norm="ortho")
            shifted = torch.fft.fftshift(fft, dim=(-2, -1))

            log_magnitude = torch.log1p(torch.abs(shifted))
            mean = log_magnitude.mean(dim=(-2, -1), keepdim=True)
            std = log_magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
            magnitude = (log_magnitude - mean) / std.clamp_min(self.epsilon)

            phase = torch.angle(shifted)
            phase_input = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
        return magnitude, phase_input


class MagnitudeBranchV3(nn.Module):
    """Extract magnitude features with optional V2-style training masking."""

    def __init__(
        self,
        output_dim: int = V3_MAGNITUDE_FEATURE_DIM,
        mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("frequency mask probability must be between 0 and 1.")
        self.output_dim = output_dim
        self.mask_probability = float(mask_probability)
        self.cnn = _build_spectrum_cnn(1, output_dim)

    def apply_frequency_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Zero one mild region per selected magnitude sample during training."""
        if not self.training or self.mask_probability == 0.0:
            return spectrum
        height, width = spectrum.shape[-2:]
        if height < 2 or width < 2:
            return spectrum
        selected = torch.rand(spectrum.shape[0], device=spectrum.device)
        selected = selected < self.mask_probability
        if not bool(selected.any()):
            return spectrum

        mask_height = min(max(1, round(height * 0.10)), height - 1)
        mask_width = min(max(1, round(width * 0.10)), width - 1)
        masked = spectrum.clone()
        for batch_index in selected.nonzero(as_tuple=False).flatten().tolist():
            top = int(
                torch.randint(
                    0,
                    height - mask_height + 1,
                    (1,),
                    device=spectrum.device,
                ).item()
            )
            left = int(
                torch.randint(
                    0,
                    width - mask_width + 1,
                    (1,),
                    device=spectrum.device,
                ).item()
            )
            masked[
                batch_index,
                :,
                top : top + mask_height,
                left : left + mask_width,
            ] = 0.0
        return masked

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.cnn(self.apply_frequency_mask(magnitude)), 1)


class PhaseBranchV3(nn.Module):
    """Extract phase features from sine/cosine phase channels."""

    def __init__(self, output_dim: int = V3_PHASE_FEATURE_DIM) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.cnn = _build_spectrum_cnn(2, output_dim)

    def forward(self, phase_input: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.cnn(phase_input), 1)


def _build_spectrum_cnn(input_channels: int, output_dim: int) -> nn.Sequential:
    """Build the shared lightweight three-convolution branch shape."""
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
    )


def _build_frequency_head(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class MagnitudePhaseResidualClassifier(nn.Module):
    """Add one controlled magnitude/phase residual to a spatial prediction."""

    def __init__(
        self,
        spatial_dim: int = V3_SPATIAL_FEATURE_DIM,
        magnitude_dim: int = V3_MAGNITUDE_FEATURE_DIM,
        phase_dim: int = V3_PHASE_FEATURE_DIM,
        frequency_hidden_dim: int = V3_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V3_FREQUENCY_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        magnitude_weight: float = DEFAULT_MAGNITUDE_WEIGHT,
        phase_weight: float = DEFAULT_PHASE_WEIGHT,
        branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        supplied_magnitude, supplied_phase, normalized_magnitude, normalized_phase = (
            _validate_and_normalize_weights(magnitude_weight, phase_weight)
        )
        if not math.isfinite(frequency_scale) or frequency_scale < 0.0:
            raise ValueError("frequency_scale must be greater than or equal to 0.")
        if not 0.0 <= branch_dropout <= 1.0:
            raise ValueError("frequency branch dropout must be between 0 and 1.")
        if not 0.0 <= frequency_dropout <= 1.0:
            raise ValueError("frequency head dropout must be between 0 and 1.")

        self.frequency_scale = float(frequency_scale)
        self.supplied_magnitude_weight = supplied_magnitude
        self.supplied_phase_weight = supplied_phase
        self.magnitude_weight = normalized_magnitude
        self.phase_weight = normalized_phase
        self.branch_dropout = float(branch_dropout)
        self.fft_preprocessor = SharedFFTPreprocessor()
        self.spatial_classifier = nn.Linear(spatial_dim, 1)
        self.magnitude_branch = MagnitudeBranchV3(magnitude_dim, mask_probability)
        self.phase_branch = PhaseBranchV3(phase_dim)
        self.magnitude_head = _build_frequency_head(
            magnitude_dim,
            frequency_hidden_dim,
            frequency_dropout,
        )
        self.phase_head = _build_frequency_head(
            phase_dim,
            frequency_hidden_dim,
            frequency_dropout,
        )

    def extract_frequency_features(
        self,
        raw_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude, phase = self.fft_preprocessor(raw_images)
        return self.magnitude_branch(magnitude), self.phase_branch(phase)

    def forward_components(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_logit = self.spatial_classifier(spatial_features)
        magnitude_features, phase_features = self.extract_frequency_features(raw_images)
        magnitude_logit = self.magnitude_head(magnitude_features)
        phase_logit = self.phase_head(phase_features)
        return spatial_logit, magnitude_logit, phase_logit

    def combine_logits(
        self,
        spatial_logit: torch.Tensor,
        magnitude_logit: torch.Tensor,
        phase_logit: torch.Tensor,
    ) -> torch.Tensor:
        frequency_logit = (
            self.magnitude_weight * magnitude_logit
            + self.phase_weight * phase_logit
        )
        if self.training and self.branch_dropout > 0.0:
            retained = torch.rand(
                frequency_logit.shape[0],
                1,
                device=frequency_logit.device,
            ) >= self.branch_dropout
            frequency_logit = frequency_logit * retained.to(
                dtype=frequency_logit.dtype
            )
        return spatial_logit + self.frequency_scale * frequency_logit

    def forward(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> torch.Tensor:
        return self.combine_logits(*self.forward_components(spatial_features, raw_images))


class HybridV3AIGCDetector(nn.Module):
    """Use EfficientNet as primary detector with magnitude and phase residuals."""

    model_type = HYBRID_V3_MODEL_TYPE
    expects_unnormalized_input = True

    def __init__(
        self,
        pretrained_spatial: bool = True,
        spatial_feature_dim: int = V3_SPATIAL_FEATURE_DIM,
        magnitude_feature_dim: int = V3_MAGNITUDE_FEATURE_DIM,
        phase_feature_dim: int = V3_PHASE_FEATURE_DIM,
        frequency_hidden_dim: int = V3_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V3_FREQUENCY_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        magnitude_weight: float = DEFAULT_MAGNITUDE_WEIGHT,
        phase_weight: float = DEFAULT_PHASE_WEIGHT,
        frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_spatial else None
        efficientnet = models.efficientnet_b0(weights=weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.classifier = MagnitudePhaseResidualClassifier(
            spatial_dim=spatial_feature_dim,
            magnitude_dim=magnitude_feature_dim,
            phase_dim=phase_feature_dim,
            frequency_hidden_dim=frequency_hidden_dim,
            frequency_dropout=frequency_dropout,
            frequency_scale=frequency_scale,
            magnitude_weight=magnitude_weight,
            phase_weight=phase_weight,
            branch_dropout=frequency_branch_dropout,
            mask_probability=frequency_mask_probability,
        )
        self.spatial_feature_dim = spatial_feature_dim
        self.magnitude_feature_dim = magnitude_feature_dim
        self.phase_feature_dim = phase_feature_dim
        self.frequency_hidden_dim = frequency_hidden_dim
        self.frequency_dropout = float(frequency_dropout)
        self.frequency_scale = float(frequency_scale)
        self.supplied_magnitude_weight = self.classifier.supplied_magnitude_weight
        self.supplied_phase_weight = self.classifier.supplied_phase_weight
        self.magnitude_weight = self.classifier.magnitude_weight
        self.phase_weight = self.classifier.phase_weight
        self.frequency_branch_dropout = float(frequency_branch_dropout)
        self.frequency_mask_probability = float(frequency_mask_probability)
        self.spatial_classifier_loaded = False
        self.spatial_classifier_source = "random"
        self.magnitude_initialized_from_v2 = False
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
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        return torch.flatten(self.avgpool(self.features(normalized)), 1)

    def extract_frequency_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier.extract_frequency_features(images)

    def forward_components(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_features = self.extract_spatial_features(images)
        return self.classifier.forward_components(spatial_features, images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier.combine_logits(*self.forward_components(images))


def _validate_and_normalize_weights(
    magnitude_weight: float,
    phase_weight: float,
) -> tuple[float, float, float, float]:
    values = (float(magnitude_weight), float(phase_weight))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("magnitude_weight and phase_weight must be finite and nonnegative.")
    total = sum(values)
    if not math.isfinite(total):
        raise ValueError("magnitude_weight and phase_weight sum must be finite.")
    if total == 0.0:
        raise ValueError("magnitude_weight and phase_weight cannot both be zero.")
    return values[0], values[1], values[0] / total, values[1] / total


def _validated_spatial_classifier_state(
    model: HybridV3AIGCDetector,
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor] | None, str]:
    candidates = {
        "EfficientNet classifier.1": (
            "classifier.1.weight",
            "classifier.1.bias",
        ),
        "hybrid residual spatial classifier": (
            "classifier.spatial_classifier.weight",
            "classifier.spatial_classifier.bias",
        ),
    }
    compatible: list[tuple[str, dict[str, torch.Tensor]]] = []
    incompatible: list[str] = []
    for source_name, (weight_key, bias_key) in candidates.items():
        has_weight = weight_key in state
        has_bias = bias_key in state
        if has_weight != has_bias:
            raise ValueError(
                f"Spatial checkpoint contains a partial {source_name}: expected both "
                f"{weight_key} and {bias_key}."
            )
        if not has_weight:
            continue
        weight = state[weight_key]
        bias = state[bias_key]
        if weight.shape == torch.Size([1, model.spatial_feature_dim]) and bias.shape == torch.Size([1]):
            compatible.append((source_name, {"weight": weight, "bias": bias}))
        else:
            incompatible.append(
                f"{source_name} has weight {tuple(weight.shape)} and bias {tuple(bias.shape)}"
            )
    if len(compatible) > 1:
        raise ValueError("Spatial checkpoint contains multiple compatible binary classifiers.")
    if compatible:
        return compatible[0][1], compatible[0][0]
    if incompatible:
        print("No compatible binary spatial classifier was loaded: " + "; ".join(incompatible))
    else:
        ignored = sorted(key for key in state if key.startswith("classifier."))
        if ignored:
            print(
                "No compatible spatial classifier was present; ignored classifier entries: "
                + ", ".join(ignored)
            )
        else:
            print("Spatial checkpoint contains features only; spatial classifier is random.")
    return None, "random"


def load_v3_spatial_checkpoint(
    model: HybridV3AIGCDetector,
    checkpoint_path: str | Path,
    device: torch.device,
) -> V3InitializationResult:
    """Strictly load all EfficientNet features and one compatible binary head."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = _checkpoint_state_dict(checkpoint)
    features = _validated_feature_state(model, state)
    classifier_state, classifier_source = _validated_spatial_classifier_state(model, state)

    model.features.load_state_dict(features, strict=True)
    if classifier_state is not None:
        model.classifier.spatial_classifier.load_state_dict(classifier_state, strict=True)
    model.spatial_classifier_loaded = classifier_state is not None
    model.spatial_classifier_source = classifier_source
    print(f"Loaded {len(features):,} spatial state entries from {path.resolve()}")
    if classifier_state is not None:
        print(f"Loaded spatial classifier from {classifier_source}")
        used_keys = {
            "classifier.1.weight",
            "classifier.1.bias",
        } if classifier_source == "EfficientNet classifier.1" else {
            "classifier.spatial_classifier.weight",
            "classifier.spatial_classifier.bias",
        }
        ignored = sorted(
            key
            for key in state
            if key.startswith("classifier.") and key not in used_keys
        )
        if ignored:
            print("Ignored non-spatial classifier entries: " + ", ".join(ignored))
    else:
        print("Initialized the Hybrid V3 spatial classifier randomly")
    return V3InitializationResult(
        feature_entry_count=len(features),
        classifier_loaded=classifier_state is not None,
        classifier_source=classifier_source,
    )


def _validated_mapped_state(
    state: Mapping[str, torch.Tensor],
    source_prefix: str,
    expected: Mapping[str, torch.Tensor],
    description: str,
) -> dict[str, torch.Tensor]:
    extracted = {
        key.removeprefix(source_prefix): value
        for key, value in state.items()
        if key.startswith(source_prefix)
    }
    missing = sorted(set(expected) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing[:10]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:10]))
        raise ValueError(f"V2 {description} keys do not match exactly (" + "; ".join(details) + ").")
    mismatched = [
        f"{key}: expected {tuple(expected[key].shape)}, found {tuple(extracted[key].shape)}"
        for key in expected
        if expected[key].shape != extracted[key].shape
    ]
    if mismatched:
        raise ValueError(f"V2 {description} tensor shapes do not match: " + "; ".join(mismatched[:10]))
    return extracted


def load_v2_warm_start(
    model: HybridV3AIGCDetector,
    checkpoint_path: str | Path,
    device: torch.device,
) -> V3InitializationResult:
    """Strictly initialize V3 spatial and magnitude paths from a V2 checkpoint."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("model_type") != HYBRID_V2_MODEL_TYPE:
        raise ValueError("--v2-checkpoint must contain model_type='hybrid_v2'.")
    state = _checkpoint_state_dict(checkpoint)
    features = _validated_feature_state(model, state)
    spatial = _validated_mapped_state(
        state,
        "classifier.spatial_classifier.",
        model.classifier.spatial_classifier.state_dict(),
        "spatial classifier",
    )
    magnitude_cnn = _validated_mapped_state(
        state,
        "classifier.frequency_branch.cnn.",
        model.classifier.magnitude_branch.cnn.state_dict(),
        "magnitude CNN",
    )
    magnitude_head = _validated_mapped_state(
        state,
        "classifier.frequency_head.",
        model.classifier.magnitude_head.state_dict(),
        "magnitude head",
    )

    model.features.load_state_dict(features, strict=True)
    model.classifier.spatial_classifier.load_state_dict(spatial, strict=True)
    model.classifier.magnitude_branch.cnn.load_state_dict(magnitude_cnn, strict=True)
    model.classifier.magnitude_head.load_state_dict(magnitude_head, strict=True)
    model.spatial_classifier_loaded = True
    model.spatial_classifier_source = "Hybrid V2 spatial classifier"
    model.magnitude_initialized_from_v2 = True
    print(f"Loaded {len(features):,} spatial state entries from {path.resolve()}")
    print("Loaded the V3 spatial classifier, magnitude CNN, and magnitude head from Hybrid V2")
    print("Initialized the new Hybrid V3 phase CNN and phase head randomly")
    return V3InitializationResult(
        feature_entry_count=len(features),
        classifier_loaded=True,
        classifier_source=model.spatial_classifier_source,
        magnitude_loaded=True,
    )


def build_hybrid_v3_model(
    device: torch.device,
    spatial_checkpoint: str | Path | None = None,
    v2_checkpoint: str | Path | None = None,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    magnitude_weight: float = DEFAULT_MAGNITUDE_WEIGHT,
    phase_weight: float = DEFAULT_PHASE_WEIGHT,
    frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
) -> HybridV3AIGCDetector:
    """Build V3 from ImageNet, a spatial detector, or an exact V2 warm start."""
    if spatial_checkpoint is not None and v2_checkpoint is not None:
        raise ValueError("--spatial-checkpoint and --v2-checkpoint are mutually exclusive.")
    model = HybridV3AIGCDetector(
        pretrained_spatial=spatial_checkpoint is None and v2_checkpoint is None,
        frequency_scale=frequency_scale,
        magnitude_weight=magnitude_weight,
        phase_weight=phase_weight,
        frequency_branch_dropout=frequency_branch_dropout,
        frequency_mask_probability=frequency_mask_probability,
    )
    if v2_checkpoint is not None:
        load_v2_warm_start(model, v2_checkpoint, device)
    elif spatial_checkpoint is not None:
        load_v3_spatial_checkpoint(model, spatial_checkpoint, device)
    else:
        print("Hybrid V3 spatial initialization: ImageNet EfficientNet-B0 features")
        print("Hybrid V3 spatial classifier initialization: random")
    return model.to(device)


def configure_hybrid_v3_stage(
    model: nn.Module,
    stage: str,
) -> tuple[tuple[nn.Module, ...], list[dict[str, object]]]:
    """Configure V3 trainable parameters and optimizer groups per shared stage."""
    if not isinstance(model, HybridV3AIGCDetector):
        raise TypeError("Hybrid V3 stage configuration requires HybridV3AIGCDetector.")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported Hybrid V3 training stage: {stage!r}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    feature_blocks = tuple(model.features.children())
    magnitude_parameters = [
        *model.classifier.magnitude_branch.parameters(),
        *model.classifier.magnitude_head.parameters(),
    ]
    phase_parameters = [
        *model.classifier.phase_branch.parameters(),
        *model.classifier.phase_head.parameters(),
    ]
    for parameter in (*magnitude_parameters, *phase_parameters):
        parameter.requires_grad = True
    groups: list[dict[str, object]] = [
        {
            "params": magnitude_parameters,
            "lr": V3_FREQUENCY_LEARNING_RATE,
            "name": "magnitude",
        },
        {
            "params": phase_parameters,
            "lr": V3_FREQUENCY_LEARNING_RATE,
            "name": "phase",
        },
    ]

    if stage == "stage1":
        frozen_modules: tuple[nn.Module, ...] = feature_blocks
        if model.spatial_classifier_loaded:
            frozen_modules += (model.classifier.spatial_classifier,)
        else:
            spatial_parameters = list(model.classifier.spatial_classifier.parameters())
            for parameter in spatial_parameters:
                parameter.requires_grad = True
            groups.append(
                {
                    "params": spatial_parameters,
                    "lr": V3_SPATIAL_LEARNING_RATE,
                    "name": "spatial_classifier",
                }
            )
        return frozen_modules, groups

    frozen_blocks = feature_blocks[:-3]
    trainable_blocks = feature_blocks[-3:]
    backbone_parameters = [
        parameter
        for block in trainable_blocks
        for parameter in block.parameters()
    ]
    spatial_parameters = list(model.classifier.spatial_classifier.parameters())
    for parameter in (*backbone_parameters, *spatial_parameters):
        parameter.requires_grad = True
    groups.extend(
        [
            {
                "params": spatial_parameters,
                "lr": V3_SPATIAL_LEARNING_RATE,
                "name": "spatial_classifier",
            },
            {
                "params": backbone_parameters,
                "lr": V3_SPATIAL_LEARNING_RATE,
                "name": "backbone",
            },
        ]
    )
    return frozen_blocks, groups
