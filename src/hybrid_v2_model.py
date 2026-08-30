"""Residual EfficientNet and FFT hybrid detector with a controlled frequency path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from .hybrid_model import FFTPreprocessor
from .transforms import IMAGENET_MEAN, IMAGENET_STD


HYBRID_V2_MODEL_TYPE = "hybrid_v2"
V2_SPATIAL_FEATURE_DIM = 1280
V2_FREQUENCY_FEATURE_DIM = 128
V2_FREQUENCY_HIDDEN_DIM = 64
V2_FREQUENCY_DROPOUT = 0.5
DEFAULT_FREQUENCY_SCALE = 0.25
DEFAULT_FREQUENCY_BRANCH_DROPOUT = 0.20
DEFAULT_FREQUENCY_MASK_PROB = 0.0
V2_FREQUENCY_LEARNING_RATE = 5e-5
V2_SPATIAL_LEARNING_RATE = 1e-5


@dataclass(frozen=True)
class SpatialInitializationResult:
    """Describe which strict spatial weights were initialized from a checkpoint."""

    feature_entry_count: int
    classifier_loaded: bool
    classifier_source: str


class FrequencyBranchV2(nn.Module):
    """Extract a small FFT representation with optional training-only masking."""

    def __init__(
        self,
        output_dim: int = V2_FREQUENCY_FEATURE_DIM,
        mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("frequency mask probability must be between 0 and 1.")
        self.output_dim = output_dim
        self.mask_probability = float(mask_probability)
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

    def apply_frequency_mask(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Zero one mild random region per selected sample during training only."""
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spectrum = self.apply_frequency_mask(self.preprocessor(images))
        return torch.flatten(self.cnn(spectrum), 1)


class ResidualHybridClassifier(nn.Module):
    """Predict a spatial logit and add a controlled FFT residual correction."""

    def __init__(
        self,
        spatial_dim: int = V2_SPATIAL_FEATURE_DIM,
        frequency_dim: int = V2_FREQUENCY_FEATURE_DIM,
        frequency_hidden_dim: int = V2_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V2_FREQUENCY_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        if not math.isfinite(frequency_scale) or frequency_scale < 0.0:
            raise ValueError("frequency_scale must be greater than or equal to 0.")
        if not 0.0 <= branch_dropout <= 1.0:
            raise ValueError("frequency branch dropout must be between 0 and 1.")
        if not 0.0 <= frequency_dropout <= 1.0:
            raise ValueError("frequency head dropout must be between 0 and 1.")
        self.frequency_scale = float(frequency_scale)
        self.branch_dropout = float(branch_dropout)
        self.spatial_classifier = nn.Linear(spatial_dim, 1)
        self.frequency_branch = FrequencyBranchV2(
            output_dim=frequency_dim,
            mask_probability=mask_probability,
        )
        self.frequency_head = nn.Sequential(
            nn.Linear(frequency_dim, frequency_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(frequency_dropout),
            nn.Linear(frequency_hidden_dim, 1),
        )

    def forward_components(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the independent spatial and frequency predictions."""
        spatial_logit = self.spatial_classifier(spatial_features)
        frequency_features = self.frequency_branch(raw_images)
        frequency_logit = self.frequency_head(frequency_features)
        return spatial_logit, frequency_logit

    def combine_logits(
        self,
        spatial_logit: torch.Tensor,
        frequency_logit: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fixed residual scale and training-only branch dropout."""
        correction = frequency_logit
        if self.training and self.branch_dropout > 0.0:
            retained = torch.rand(
                frequency_logit.shape[0],
                1,
                device=frequency_logit.device,
            ) >= self.branch_dropout
            correction = correction * retained.to(dtype=correction.dtype)
        return spatial_logit + self.frequency_scale * correction

    def forward(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> torch.Tensor:
        return self.combine_logits(*self.forward_components(spatial_features, raw_images))


class HybridV2AIGCDetector(nn.Module):
    """Use EfficientNet as the primary detector and FFT as a residual correction."""

    model_type = HYBRID_V2_MODEL_TYPE
    expects_unnormalized_input = True

    def __init__(
        self,
        pretrained_spatial: bool = True,
        spatial_feature_dim: int = V2_SPATIAL_FEATURE_DIM,
        frequency_feature_dim: int = V2_FREQUENCY_FEATURE_DIM,
        frequency_hidden_dim: int = V2_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V2_FREQUENCY_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_spatial else None
        efficientnet = models.efficientnet_b0(weights=weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.classifier = ResidualHybridClassifier(
            spatial_dim=spatial_feature_dim,
            frequency_dim=frequency_feature_dim,
            frequency_hidden_dim=frequency_hidden_dim,
            frequency_dropout=frequency_dropout,
            frequency_scale=frequency_scale,
            branch_dropout=frequency_branch_dropout,
            mask_probability=frequency_mask_probability,
        )
        self.spatial_feature_dim = spatial_feature_dim
        self.frequency_feature_dim = frequency_feature_dim
        self.frequency_hidden_dim = frequency_hidden_dim
        self.frequency_dropout = float(frequency_dropout)
        self.frequency_scale = float(frequency_scale)
        self.frequency_branch_dropout = float(frequency_branch_dropout)
        self.frequency_mask_probability = float(frequency_mask_probability)
        self.spatial_classifier_loaded = False
        self.spatial_classifier_source = "random"
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

    def extract_frequency_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier.frequency_branch(images)

    def forward_components(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_features = self.extract_spatial_features(images)
        return self.classifier.forward_components(spatial_features, images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier.combine_logits(*self.forward_components(images))


def _checkpoint_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Spatial checkpoint must contain a state-dict mapping.")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("Spatial checkpoint state_dict must map string keys to tensors.")
    return state


def _validated_feature_state(
    model: HybridV2AIGCDetector,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
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
        raise ValueError(
            "Spatial checkpoint keys do not match exactly ("
            + "; ".join(details)
            + ")."
        )
    mismatched = [
        f"{key}: expected {tuple(expected[key].shape)}, found {tuple(extracted[key].shape)}"
        for key in expected
        if expected[key].shape != extracted[key].shape
    ]
    if mismatched:
        raise ValueError(
            "Spatial checkpoint tensor shapes do not match: "
            + "; ".join(mismatched[:10])
        )
    return extracted


def _load_compatible_spatial_classifier(
    model: HybridV2AIGCDetector,
    state: Mapping[str, torch.Tensor],
) -> tuple[bool, str]:
    candidates = {
        "EfficientNet classifier.1": (
            "classifier.1.weight",
            "classifier.1.bias",
        ),
        "Hybrid V2 spatial classifier": (
            "classifier.spatial_classifier.weight",
            "classifier.spatial_classifier.bias",
        ),
    }
    compatible: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    incompatible: list[str] = []
    for source_name, (weight_key, bias_key) in candidates.items():
        has_weight = weight_key in state
        has_bias = bias_key in state
        if has_weight != has_bias:
            raise ValueError(
                f"Spatial checkpoint contains a partial {source_name}: "
                f"expected both {weight_key} and {bias_key}."
            )
        if not has_weight:
            continue
        weight = state[weight_key]
        bias = state[bias_key]
        if weight.shape == torch.Size([1, model.spatial_feature_dim]) and bias.shape == torch.Size([1]):
            compatible.append((source_name, weight, bias))
        else:
            incompatible.append(
                f"{source_name} has weight {tuple(weight.shape)} and bias {tuple(bias.shape)}"
            )

    if len(compatible) > 1:
        raise ValueError("Spatial checkpoint contains multiple compatible binary classifiers.")
    if compatible:
        source_name, weight, bias = compatible[0]
        model.classifier.spatial_classifier.load_state_dict(
            {"weight": weight, "bias": bias},
            strict=True,
        )
        used_keys = set(candidates[source_name])
        ignored = sorted(
            key
            for key in state
            if key.startswith("classifier.") and key not in used_keys
        )
        if ignored:
            print(
                "Ignored non-spatial classifier entries: "
                + ", ".join(ignored)
            )
        return True, source_name
    if incompatible:
        print(
            "No compatible binary spatial classifier was loaded: "
            + "; ".join(incompatible)
        )
    else:
        ignored = sorted(key for key in state if key.startswith("classifier."))
        if ignored:
            print(
                "No compatible spatial classifier was present; ignored classifier "
                "entries: "
                + ", ".join(ignored)
            )
        else:
            print("Spatial checkpoint contains features only; spatial classifier is random.")
    return False, "random"


def load_v2_spatial_checkpoint(
    model: HybridV2AIGCDetector,
    checkpoint_path: str | Path,
    device: torch.device,
) -> SpatialInitializationResult:
    """Strictly load all features and reuse one compatible binary classifier."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = _checkpoint_state_dict(checkpoint)
    features = _validated_feature_state(model, state)
    classifier_loaded, classifier_source = _load_compatible_spatial_classifier(
        model,
        state,
    )
    model.features.load_state_dict(features, strict=True)
    model.spatial_classifier_loaded = classifier_loaded
    model.spatial_classifier_source = classifier_source
    print(f"Loaded {len(features):,} spatial state entries from {path.resolve()}")
    if classifier_loaded:
        print(f"Loaded spatial classifier from {classifier_source}")
    else:
        print("Initialized the Hybrid V2 spatial classifier randomly")
    return SpatialInitializationResult(
        feature_entry_count=len(features),
        classifier_loaded=classifier_loaded,
        classifier_source=classifier_source,
    )


def build_hybrid_v2_model(
    device: torch.device,
    spatial_checkpoint: str | Path | None = None,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
) -> HybridV2AIGCDetector:
    """Build V2 from strict detector weights or ImageNet spatial features."""
    model = HybridV2AIGCDetector(
        pretrained_spatial=spatial_checkpoint is None,
        frequency_scale=frequency_scale,
        frequency_branch_dropout=frequency_branch_dropout,
        frequency_mask_probability=frequency_mask_probability,
    )
    if spatial_checkpoint is None:
        print("Hybrid V2 spatial initialization: ImageNet EfficientNet-B0 features")
        print("Hybrid V2 spatial classifier initialization: random")
    else:
        load_v2_spatial_checkpoint(model, spatial_checkpoint, device)
    return model.to(device)


def configure_hybrid_v2_stage(
    model: nn.Module,
    stage: str,
) -> tuple[tuple[nn.Module, ...], list[dict[str, object]]]:
    """Configure V2 trainable parameters and named optimizer groups per stage."""
    if not isinstance(model, HybridV2AIGCDetector):
        raise TypeError("Hybrid V2 stage configuration requires HybridV2AIGCDetector.")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported Hybrid V2 training stage: {stage!r}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    feature_blocks = tuple(model.features.children())
    frequency_parameters = [
        *model.classifier.frequency_branch.parameters(),
        *model.classifier.frequency_head.parameters(),
    ]
    for parameter in frequency_parameters:
        parameter.requires_grad = True

    groups: list[dict[str, object]] = [
        {
            "params": frequency_parameters,
            "lr": V2_FREQUENCY_LEARNING_RATE,
            "name": "frequency",
        }
    ]
    if stage == "stage1":
        frozen_modules: tuple[nn.Module, ...] = feature_blocks
        if model.spatial_classifier_loaded:
            frozen_modules += (model.classifier.spatial_classifier,)
        else:
            for parameter in model.classifier.spatial_classifier.parameters():
                parameter.requires_grad = True
            groups.append(
                {
                    "params": list(model.classifier.spatial_classifier.parameters()),
                    "lr": V2_SPATIAL_LEARNING_RATE,
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
    for parameter in backbone_parameters:
        parameter.requires_grad = True
    for parameter in model.classifier.spatial_classifier.parameters():
        parameter.requires_grad = True
    groups.extend(
        [
            {
                "params": list(model.classifier.spatial_classifier.parameters()),
                "lr": V2_SPATIAL_LEARNING_RATE,
                "name": "spatial_classifier",
            },
            {
                "params": backbone_parameters,
                "lr": V2_SPATIAL_LEARNING_RATE,
                "name": "backbone",
            },
        ]
    )
    return frozen_blocks, groups
