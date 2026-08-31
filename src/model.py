"""EfficientNet model creation and checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from .hybrid_model import HYBRID_MODEL_TYPE, HybridAIGCDetector
from .hybrid_v2_model import (
    HYBRID_V2_MODEL_TYPE,
    V2_SPATIAL_FEATURE_DIM,
    HybridV2AIGCDetector,
)
from .hybrid_v3_model import (
    HYBRID_V3_MODEL_TYPE,
    V3_SPATIAL_FEATURE_DIM,
    HybridV3AIGCDetector,
)
from .hybrid_v31_model import (
    HYBRID_V31_MODEL_TYPE,
    V31_SPATIAL_FEATURE_DIM,
    HybridV31AIGCDetector,
)


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


def _required_checkpoint_number(
    checkpoint: Mapping[str, object],
    key: str,
    model_name: str = "Hybrid V2",
) -> float:
    value = checkpoint.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{model_name} checkpoint requires numeric metadata {key!r}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{model_name} checkpoint metadata {key!r} must be finite.")
    return float(value)


def load_model(checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Load saved model weights, move them to the device, and enable inference mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = checkpoint.get("model_type") if isinstance(checkpoint, Mapping) else None
    reported_v31_weights: Mapping[str, object] | None = None
    if model_type == HYBRID_MODEL_TYPE:
        model = HybridAIGCDetector(pretrained_spatial=False).to(device)
    elif model_type == HYBRID_V2_MODEL_TYPE:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Hybrid V2 checkpoint must be a metadata mapping.")
        spatial_feature_dim = int(
            _required_checkpoint_number(checkpoint, "spatial_feature_dim")
        )
        if spatial_feature_dim != V2_SPATIAL_FEATURE_DIM:
            raise ValueError(
                "Hybrid V2 EfficientNet-B0 requires spatial_feature_dim=1280; "
                f"found {spatial_feature_dim}."
            )
        model = HybridV2AIGCDetector(
            pretrained_spatial=False,
            spatial_feature_dim=spatial_feature_dim,
            frequency_feature_dim=int(
                _required_checkpoint_number(checkpoint, "frequency_feature_dim")
            ),
            frequency_hidden_dim=int(
                _required_checkpoint_number(checkpoint, "frequency_hidden_dim")
            ),
            frequency_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_dropout",
            ),
            frequency_scale=_required_checkpoint_number(checkpoint, "frequency_scale"),
            frequency_branch_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_branch_dropout",
            ),
            frequency_mask_probability=_required_checkpoint_number(
                checkpoint,
                "frequency_mask_prob",
            ),
        ).to(device)
        model.spatial_classifier_loaded = bool(
            checkpoint.get("spatial_classifier_loaded", False)
        )
        model.spatial_classifier_source = str(
            checkpoint.get("spatial_classifier_source", "checkpoint")
        )
    elif model_type == HYBRID_V3_MODEL_TYPE:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Hybrid V3 checkpoint must be a metadata mapping.")
        spatial_feature_dim = int(
            _required_checkpoint_number(
                checkpoint,
                "spatial_feature_dim",
                "Hybrid V3",
            )
        )
        if spatial_feature_dim != V3_SPATIAL_FEATURE_DIM:
            raise ValueError(
                "Hybrid V3 EfficientNet-B0 requires spatial_feature_dim=1280; "
                f"found {spatial_feature_dim}."
            )
        if checkpoint.get("phase_representation") != "sin_cos":
            raise ValueError("Hybrid V3 checkpoint requires phase_representation='sin_cos'.")
        if checkpoint.get("fft_normalization") != "ortho":
            raise ValueError("Hybrid V3 checkpoint requires fft_normalization='ortho'.")
        model = HybridV3AIGCDetector(
            pretrained_spatial=False,
            spatial_feature_dim=spatial_feature_dim,
            magnitude_feature_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "magnitude_feature_dim",
                    "Hybrid V3",
                )
            ),
            phase_feature_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "phase_feature_dim",
                    "Hybrid V3",
                )
            ),
            frequency_hidden_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "frequency_hidden_dim",
                    "Hybrid V3",
                )
            ),
            frequency_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_dropout",
                "Hybrid V3",
            ),
            frequency_scale=_required_checkpoint_number(
                checkpoint,
                "frequency_scale",
                "Hybrid V3",
            ),
            magnitude_weight=_required_checkpoint_number(
                checkpoint,
                "supplied_magnitude_weight",
                "Hybrid V3",
            ),
            phase_weight=_required_checkpoint_number(
                checkpoint,
                "supplied_phase_weight",
                "Hybrid V3",
            ),
            frequency_branch_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_branch_dropout",
                "Hybrid V3",
            ),
            frequency_mask_probability=_required_checkpoint_number(
                checkpoint,
                "frequency_mask_prob",
                "Hybrid V3",
            ),
        ).to(device)
        normalized_magnitude = _required_checkpoint_number(
            checkpoint,
            "normalized_magnitude_weight",
            "Hybrid V3",
        )
        normalized_phase = _required_checkpoint_number(
            checkpoint,
            "normalized_phase_weight",
            "Hybrid V3",
        )
        if not math.isclose(
            normalized_magnitude,
            model.magnitude_weight,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ) or not math.isclose(
            normalized_phase,
            model.phase_weight,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Hybrid V3 normalized magnitude/phase metadata does not match "
                "the supplied weights."
            )
        model.spatial_classifier_loaded = bool(
            checkpoint.get("spatial_classifier_loaded", False)
        )
        model.spatial_classifier_source = str(
            checkpoint.get("spatial_classifier_source", "checkpoint")
        )
        model.magnitude_initialized_from_v2 = bool(
            checkpoint.get("magnitude_initialized_from_v2", False)
        )
    elif model_type == HYBRID_V31_MODEL_TYPE:
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Hybrid V3.1 checkpoint must be a metadata mapping.")
        spatial_feature_dim = int(
            _required_checkpoint_number(
                checkpoint,
                "spatial_feature_dim",
                "Hybrid V3.1",
            )
        )
        if spatial_feature_dim != V31_SPATIAL_FEATURE_DIM:
            raise ValueError(
                "Hybrid V3.1 EfficientNet-B0 requires spatial_feature_dim=1280; "
                f"found {spatial_feature_dim}."
            )
        if checkpoint.get("phase_representation") != "sin_cos":
            raise ValueError(
                "Hybrid V3.1 checkpoint requires phase_representation='sin_cos'."
            )
        if checkpoint.get("fft_normalization") != "ortho":
            raise ValueError(
                "Hybrid V3.1 checkpoint requires fft_normalization='ortho'."
            )
        if checkpoint.get("frequency_fusion_type") != "learned_softmax":
            raise ValueError(
                "Hybrid V3.1 checkpoint requires frequency_fusion_type='learned_softmax'."
            )
        model = HybridV31AIGCDetector(
            pretrained_spatial=False,
            spatial_feature_dim=spatial_feature_dim,
            magnitude_feature_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "magnitude_feature_dim",
                    "Hybrid V3.1",
                )
            ),
            phase_feature_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "phase_feature_dim",
                    "Hybrid V3.1",
                )
            ),
            frequency_hidden_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "frequency_hidden_dim",
                    "Hybrid V3.1",
                )
            ),
            frequency_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_dropout",
                "Hybrid V3.1",
            ),
            radial_bins=int(
                _required_checkpoint_number(
                    checkpoint,
                    "radial_bins",
                    "Hybrid V3.1",
                )
            ),
            radial_hidden_dim=int(
                _required_checkpoint_number(
                    checkpoint,
                    "radial_hidden_dim",
                    "Hybrid V3.1",
                )
            ),
            radial_dropout=_required_checkpoint_number(
                checkpoint,
                "radial_dropout",
                "Hybrid V3.1",
            ),
            frequency_scale=_required_checkpoint_number(
                checkpoint,
                "frequency_scale",
                "Hybrid V3.1",
            ),
            frequency_branch_dropout=_required_checkpoint_number(
                checkpoint,
                "frequency_branch_dropout",
                "Hybrid V3.1",
            ),
            frequency_mask_probability=_required_checkpoint_number(
                checkpoint,
                "frequency_mask_prob",
                "Hybrid V3.1",
            ),
        ).to(device)
        model.spatial_classifier_loaded = bool(
            checkpoint.get("spatial_classifier_loaded", False)
        )
        model.spatial_classifier_source = str(
            checkpoint.get("spatial_classifier_source", "checkpoint")
        )
        candidate_weights = checkpoint.get("learned_frequency_weights")
        if not isinstance(candidate_weights, Mapping):
            raise ValueError(
                "Hybrid V3.1 checkpoint requires learned_frequency_weights metadata."
            )
        reported_v31_weights = candidate_weights
    elif model_type in (None, "efficientnet"):
        model = build_model(device, pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping.")
    model.load_state_dict(state_dict, strict=True)
    if reported_v31_weights is not None:
        learned = model.classifier.normalized_frequency_weights().detach().cpu()
        for index, name in enumerate(("magnitude", "phase", "radial")):
            reported = reported_v31_weights.get(name)
            if isinstance(reported, bool) or not isinstance(reported, (int, float)):
                raise ValueError(
                    "Hybrid V3.1 learned_frequency_weights must contain numeric "
                    f"{name!r}."
                )
            if not math.isclose(
                float(reported),
                float(learned[index]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "Hybrid V3.1 reported frequency weights do not match the "
                    "strictly loaded fusion parameters."
                )
    model.eval()
    return model
