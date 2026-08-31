"""Hybrid V3.1 detector with radial frequency features and learned fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from .frequency_features import (
    DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    DEFAULT_FREQUENCY_MASK_PROB,
    DEFAULT_FREQUENCY_SCALE,
    FREQUENCY_DROPOUT,
    FREQUENCY_HIDDEN_DIM,
    FREQUENCY_LEARNING_RATE,
    MAGNITUDE_FEATURE_DIM,
    PHASE_FEATURE_DIM,
    SPATIAL_FEATURE_DIM,
    SPATIAL_LEARNING_RATE,
    MagnitudeBranch,
    PhaseBranch,
    SharedFFTPreprocessor,
    build_frequency_head,
    checkpoint_state_dict,
    validated_feature_state,
    validated_spatial_classifier_state,
)
from .transforms import IMAGENET_MEAN, IMAGENET_STD


HYBRID_V31_MODEL_TYPE = "hybrid_v31"
V31_SPATIAL_FEATURE_DIM = SPATIAL_FEATURE_DIM
V31_MAGNITUDE_FEATURE_DIM = MAGNITUDE_FEATURE_DIM
V31_PHASE_FEATURE_DIM = PHASE_FEATURE_DIM
V31_FREQUENCY_HIDDEN_DIM = FREQUENCY_HIDDEN_DIM
V31_FREQUENCY_DROPOUT = FREQUENCY_DROPOUT
V31_FREQUENCY_LEARNING_RATE = FREQUENCY_LEARNING_RATE
V31_SPATIAL_LEARNING_RATE = SPATIAL_LEARNING_RATE
DEFAULT_RADIAL_BINS = 32
MIN_RADIAL_BINS = 4
V31_RADIAL_HIDDEN_DIM = 64
V31_RADIAL_DROPOUT = 0.3


RadialCacheKey = tuple[int, int, int, str, int | None]


@dataclass(frozen=True)
class V31InitializationResult:
    """Describe the strict spatial weights initialized before V3.1 training."""

    feature_entry_count: int
    classifier_loaded: bool
    classifier_source: str


class RadialProfileExtractor(nn.Module):
    """Average normalized log-magnitude values in low-to-high radial bins."""

    def __init__(self, radial_bins: int = DEFAULT_RADIAL_BINS) -> None:
        """Configure the number of low-to-high radial spectrum bins."""
        super().__init__()
        if isinstance(radial_bins, bool) or radial_bins < MIN_RADIAL_BINS:
            raise ValueError(f"radial_bins must be at least {MIN_RADIAL_BINS}.")
        self.radial_bins = int(radial_bins)
        self._bin_cache: dict[RadialCacheKey, tuple[torch.Tensor, torch.Tensor]] = {}

    def cache_key(self, height: int, width: int, device: torch.device) -> RadialCacheKey:
        """Identify geometry by resolution, bin count, and exact compute device."""
        return (
            int(height),
            int(width),
            self.radial_bins,
            device.type,
            device.index if device.type == "cuda" else None,
        )

    def _bin_geometry(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached flattened bin indices and safe per-bin pixel counts."""
        key = self.cache_key(height, width, device)
        cached = self._bin_cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(height, device=device, dtype=torch.float32)
        columns = torch.arange(width, device=device, dtype=torch.float32)
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        distance = torch.sqrt(
            (row_grid - float(height // 2)).square()
            + (column_grid - float(width // 2)).square()
        )
        max_distance = distance.max()
        if float(max_distance.item()) == 0.0:
            bin_indices = torch.zeros(height * width, device=device, dtype=torch.long)
        else:
            bin_indices = torch.floor(
                distance / max_distance * self.radial_bins
            ).to(torch.long)
            bin_indices = bin_indices.clamp_max(self.radial_bins - 1).reshape(-1)
        counts = torch.bincount(
            bin_indices,
            minlength=self.radial_bins,
        ).to(dtype=torch.float32)
        counts = counts.clamp_min(1.0)
        self._bin_cache[key] = (bin_indices, counts)
        return bin_indices, counts

    def forward(self, normalized_magnitude: torch.Tensor) -> torch.Tensor:
        """Average each image's normalized magnitude within cached radial bins."""
        if normalized_magnitude.ndim != 4 or normalized_magnitude.shape[1] != 1:
            raise ValueError(
                "Radial extraction expects magnitude shaped [batch, 1, height, width]."
            )
        with torch.autocast(
            device_type=normalized_magnitude.device.type,
            enabled=False,
        ):
            magnitude = normalized_magnitude.float()
            batch_size, _, height, width = magnitude.shape
            bin_indices, counts = self._bin_geometry(
                height,
                width,
                magnitude.device,
            )
            expanded_indices = bin_indices.unsqueeze(0).expand(batch_size, -1)
            sums = torch.zeros(
                batch_size,
                self.radial_bins,
                device=magnitude.device,
                dtype=torch.float32,
            )
            sums.scatter_add_(1, expanded_indices, magnitude.reshape(batch_size, -1))
            profiles = sums / counts.unsqueeze(0)
        return profiles


class LearnedFrequencyResidualClassifier(nn.Module):
    """Combine magnitude, phase, and radial logits with learned softmax weights."""

    def __init__(
        self,
        spatial_dim: int = V31_SPATIAL_FEATURE_DIM,
        magnitude_dim: int = V31_MAGNITUDE_FEATURE_DIM,
        phase_dim: int = V31_PHASE_FEATURE_DIM,
        frequency_hidden_dim: int = V31_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V31_FREQUENCY_DROPOUT,
        radial_bins: int = DEFAULT_RADIAL_BINS,
        radial_hidden_dim: int = V31_RADIAL_HIDDEN_DIM,
        radial_dropout: float = V31_RADIAL_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        """Build spatial and three forensic heads with controlled residual fusion."""
        super().__init__()
        if not math.isfinite(frequency_scale) or frequency_scale < 0.0:
            raise ValueError("frequency_scale must be greater than or equal to 0.")
        if not 0.0 <= branch_dropout <= 1.0:
            raise ValueError("frequency branch dropout must be between 0 and 1.")
        if not 0.0 <= frequency_dropout <= 1.0:
            raise ValueError("frequency head dropout must be between 0 and 1.")
        if not 0.0 <= radial_dropout <= 1.0:
            raise ValueError("radial head dropout must be between 0 and 1.")

        self.frequency_scale = float(frequency_scale)
        self.branch_dropout = float(branch_dropout)
        self.fft_preprocessor = SharedFFTPreprocessor()
        self.radial_extractor = RadialProfileExtractor(radial_bins)
        self.spatial_classifier = nn.Linear(spatial_dim, 1)
        self.magnitude_branch = MagnitudeBranch(magnitude_dim, mask_probability)
        self.phase_branch = PhaseBranch(phase_dim)
        self.magnitude_head = build_frequency_head(
            magnitude_dim,
            frequency_hidden_dim,
            frequency_dropout,
        )
        self.phase_head = build_frequency_head(
            phase_dim,
            frequency_hidden_dim,
            frequency_dropout,
        )
        self.radial_head = nn.Sequential(
            nn.Linear(radial_bins, radial_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(radial_dropout),
            nn.Linear(radial_hidden_dim, 1),
        )
        self.raw_frequency_weights = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def normalized_frequency_weights(self) -> torch.Tensor:
        """Return nonnegative branch weights that sum to one."""
        return torch.softmax(self.raw_frequency_weights, dim=0)

    def forward_components(
        self,
        spatial_features: torch.Tensor,
        raw_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spatial, magnitude, phase, and radial logits from one shared FFT."""
        spatial_logit = self.spatial_classifier(spatial_features)
        normalized_magnitude, phase = self.fft_preprocessor(raw_images)

        # Radial extraction intentionally precedes magnitude-only masking.
        radial_features = self.radial_extractor(normalized_magnitude)
        magnitude_features = self.magnitude_branch(normalized_magnitude)
        phase_features = self.phase_branch(phase)

        magnitude_logit = self.magnitude_head(magnitude_features)
        phase_logit = self.phase_head(phase_features)
        radial_logit = self.radial_head(radial_features)
        return spatial_logit, magnitude_logit, phase_logit, radial_logit

    def combine_logits(
        self,
        spatial_logit: torch.Tensor,
        magnitude_logit: torch.Tensor,
        phase_logit: torch.Tensor,
        radial_logit: torch.Tensor,
    ) -> torch.Tensor:
        """Softmax-mix forensic logits and add the scaled residual to spatial output."""
        weights = self.normalized_frequency_weights()
        frequency_logit = (
            weights[0] * magnitude_logit
            + weights[1] * phase_logit
            + weights[2] * radial_logit
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
        """Return one fused REAL-positive logit per image."""
        return self.combine_logits(*self.forward_components(spatial_features, raw_images))


class HybridV31AIGCDetector(nn.Module):
    """Use EfficientNet with learned magnitude, phase, and radial residual fusion."""

    model_type = HYBRID_V31_MODEL_TYPE
    expects_unnormalized_input = True

    def __init__(
        self,
        pretrained_spatial: bool = True,
        spatial_feature_dim: int = V31_SPATIAL_FEATURE_DIM,
        magnitude_feature_dim: int = V31_MAGNITUDE_FEATURE_DIM,
        phase_feature_dim: int = V31_PHASE_FEATURE_DIM,
        frequency_hidden_dim: int = V31_FREQUENCY_HIDDEN_DIM,
        frequency_dropout: float = V31_FREQUENCY_DROPOUT,
        radial_bins: int = DEFAULT_RADIAL_BINS,
        radial_hidden_dim: int = V31_RADIAL_HIDDEN_DIM,
        radial_dropout: float = V31_RADIAL_DROPOUT,
        frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
        frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    ) -> None:
        """Construct EfficientNet features and the final V3.1 classifier."""
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_spatial else None
        efficientnet = models.efficientnet_b0(weights=weights)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.classifier = LearnedFrequencyResidualClassifier(
            spatial_dim=spatial_feature_dim,
            magnitude_dim=magnitude_feature_dim,
            phase_dim=phase_feature_dim,
            frequency_hidden_dim=frequency_hidden_dim,
            frequency_dropout=frequency_dropout,
            radial_bins=radial_bins,
            radial_hidden_dim=radial_hidden_dim,
            radial_dropout=radial_dropout,
            frequency_scale=frequency_scale,
            branch_dropout=frequency_branch_dropout,
            mask_probability=frequency_mask_probability,
        )
        self.spatial_feature_dim = spatial_feature_dim
        self.magnitude_feature_dim = magnitude_feature_dim
        self.phase_feature_dim = phase_feature_dim
        self.frequency_hidden_dim = frequency_hidden_dim
        self.frequency_dropout = float(frequency_dropout)
        self.radial_bins = int(radial_bins)
        self.radial_hidden_dim = int(radial_hidden_dim)
        self.radial_dropout = float(radial_dropout)
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
        """Normalize raw RGB internally and produce pooled EfficientNet features."""
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        return torch.flatten(self.avgpool(self.features(normalized)), 1)

    def forward_components(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all branch logits while sharing the original input tensor."""
        spatial_features = self.extract_spatial_features(images)
        return self.classifier.forward_components(spatial_features, images)

    def forward_with_branch_logits(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Return one final forward and its unmodified component logits."""
        components = self.forward_components(images)
        output = self.classifier.combine_logits(*components)
        spatial, magnitude, phase, radial = components
        return output, {
            "spatial_logit": spatial,
            "magnitude_logit": magnitude,
            "phase_logit": phase,
            "radial_logit": radial,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return final logits shaped [batch,1] with P(REAL) sigmoid semantics."""
        return self.classifier.combine_logits(*self.forward_components(images))


def load_v31_spatial_checkpoint(
    model: HybridV31AIGCDetector,
    checkpoint_path: str | Path,
    device: torch.device,
) -> V31InitializationResult:
    """Strictly load all EfficientNet features and a compatible binary head."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint_state_dict(checkpoint)
    features = validated_feature_state(model, state)
    classifier_state, classifier_source = validated_spatial_classifier_state(model, state)

    model.features.load_state_dict(features, strict=True)
    if classifier_state is not None:
        model.classifier.spatial_classifier.load_state_dict(classifier_state, strict=True)
    model.spatial_classifier_loaded = classifier_state is not None
    model.spatial_classifier_source = classifier_source
    print(f"Loaded {len(features):,} spatial state entries from {path.resolve()}")
    if classifier_state is not None:
        print(f"Loaded spatial classifier from {classifier_source}")
    else:
        print("Initialized the Hybrid V3.1 spatial classifier randomly")
    return V31InitializationResult(
        feature_entry_count=len(features),
        classifier_loaded=classifier_state is not None,
        classifier_source=classifier_source,
    )


def build_hybrid_v31_model(
    device: torch.device,
    spatial_checkpoint: str | Path | None = None,
    frequency_scale: float = DEFAULT_FREQUENCY_SCALE,
    frequency_branch_dropout: float = DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    frequency_mask_probability: float = DEFAULT_FREQUENCY_MASK_PROB,
    radial_bins: int = DEFAULT_RADIAL_BINS,
) -> HybridV31AIGCDetector:
    """Build V3.1 from ImageNet or one strict spatial detector checkpoint."""
    model = HybridV31AIGCDetector(
        pretrained_spatial=spatial_checkpoint is None,
        frequency_scale=frequency_scale,
        frequency_branch_dropout=frequency_branch_dropout,
        frequency_mask_probability=frequency_mask_probability,
        radial_bins=radial_bins,
    )
    if spatial_checkpoint is None:
        print("Hybrid V3.1 spatial initialization: ImageNet EfficientNet-B0 features")
        print("Hybrid V3.1 spatial classifier initialization: random")
    else:
        load_v31_spatial_checkpoint(model, spatial_checkpoint, device)
    return model.to(device)


def configure_hybrid_v31_stage(
    model: nn.Module,
    stage: str,
) -> tuple[tuple[nn.Module, ...], list[dict[str, object]]]:
    """Configure V3.1 trainable parameters and optimizer groups per stage."""
    if not isinstance(model, HybridV31AIGCDetector):
        raise TypeError("Hybrid V3.1 stage configuration requires HybridV31AIGCDetector.")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported Hybrid V3.1 training stage: {stage!r}")

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
    radial_parameters = list(model.classifier.radial_head.parameters())
    fusion_parameters = [model.classifier.raw_frequency_weights]
    for parameter in (
        *magnitude_parameters,
        *phase_parameters,
        *radial_parameters,
        *fusion_parameters,
    ):
        parameter.requires_grad = True
    groups: list[dict[str, object]] = [
        {"params": magnitude_parameters, "lr": V31_FREQUENCY_LEARNING_RATE, "name": "magnitude"},
        {"params": phase_parameters, "lr": V31_FREQUENCY_LEARNING_RATE, "name": "phase"},
        {"params": radial_parameters, "lr": V31_FREQUENCY_LEARNING_RATE, "name": "radial"},
        {"params": fusion_parameters, "lr": V31_FREQUENCY_LEARNING_RATE, "name": "frequency_fusion"},
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
                {"params": spatial_parameters, "lr": V31_SPATIAL_LEARNING_RATE, "name": "spatial_classifier"}
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
            {"params": spatial_parameters, "lr": V31_SPATIAL_LEARNING_RATE, "name": "spatial_classifier"},
            {"params": backbone_parameters, "lr": V31_SPATIAL_LEARNING_RATE, "name": "backbone"},
        ]
    )
    return frozen_blocks, groups


def v31_epoch_metadata(model: nn.Module) -> Mapping[str, object]:
    """Print and return current learned fusion weights once per epoch."""
    if not isinstance(model, HybridV31AIGCDetector):
        raise TypeError("V3.1 epoch metadata requires HybridV31AIGCDetector.")
    weights = model.classifier.normalized_frequency_weights().detach().cpu()
    values = {
        "magnitude": float(weights[0]),
        "phase": float(weights[1]),
        "radial": float(weights[2]),
    }
    print("Frequency fusion weights:")
    print(f"  magnitude: {values['magnitude']:.4f}")
    print(f"  phase:     {values['phase']:.4f}")
    print(f"  radial:    {values['radial']:.4f}")
    return {"learned_frequency_weights": values}


def v31_validation_forward(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Provide final and component logits from one V3.1 validation forward."""
    if not isinstance(model, HybridV31AIGCDetector):
        raise TypeError("V3.1 validation forward requires HybridV31AIGCDetector.")
    return model.forward_with_branch_logits(images)
