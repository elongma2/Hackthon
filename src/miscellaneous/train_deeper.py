"""Progressive three-stage EfficientNet-B0 fine-tuning.

This module is intentionally separate from the existing staged trainer so the
new experiment cannot alter established training commands or checkpoint files.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluate import evaluate
from ..multisource_dataset import FAKE_LABEL, REAL_LABEL
from ..source_balanced import calculate_source_metrics
from ..train import train_one_epoch


EXPECTED_EFFICIENTNET_BLOCKS = 9
DEFAULT_STAGE1_EPOCHS = 2
DEFAULT_STAGE2_EPOCHS = 2
DEFAULT_STAGE3_EPOCHS = 3
DEFAULT_CLASSIFIER_LR = 1e-4
DEFAULT_LATE_BLOCKS_LR = 1e-5
DEFAULT_MIDDLE_BLOCKS_LR = 3e-6
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_PROBABILITY_THRESHOLD = 0.5


def get_efficientnet_feature_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    """Return the verified top-level EfficientNet-B0 feature blocks."""
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(features, nn.Module) or not isinstance(classifier, nn.Module):
        raise TypeError("Progressive training requires model.features and model.classifier.")
    blocks = tuple(features.children())
    if len(blocks) != EXPECTED_EFFICIENTNET_BLOCKS:
        raise ValueError(
            "Progressive EfficientNet-B0 training expects exactly nine top-level "
            f"feature blocks; found {len(blocks)}."
        )
    return blocks


def print_efficientnet_structure(model: nn.Module) -> None:
    """Print the actual feature modules used by the three stage groups."""
    blocks = get_efficientnet_feature_blocks(model)
    print("\n--- EfficientNet-B0 Progressive Fine-Tuning Structure ---")
    for index, block in enumerate(blocks):
        parameters = sum(parameter.numel() for parameter in block.parameters())
        print(f"Block {index}: {type(block).__name__} ({parameters:,} parameters)")
    classifier_parameters = sum(
        parameter.numel() for parameter in model.classifier.parameters()
    )
    print(f"Blocks 0-3: {[type(block).__name__ for block in blocks[0:4]]}")
    print(f"Blocks 4-5: {[type(block).__name__ for block in blocks[4:6]]}")
    print(f"Blocks 6-8: {[type(block).__name__ for block in blocks[6:9]]}")
    print(
        f"Classifier: {type(model.classifier).__name__} "
        f"({classifier_parameters:,} parameters)"
    )


def configure_progressive_stage(
    model: nn.Module,
    stage: str,
) -> tuple[tuple[nn.Module, ...], tuple[nn.Module, ...]]:
    """Apply one stage's exact freeze policy and return frozen/trainable blocks."""
    blocks = get_efficientnet_feature_blocks(model)
    trainable_indices_by_stage = {
        "stage1": (),
        "stage2": (6, 7, 8),
        "stage3": (4, 5, 6, 7, 8),
    }
    try:
        trainable_indices = trainable_indices_by_stage[stage]
    except KeyError as error:
        raise ValueError(f"Unknown progressive training stage: {stage!r}.") from error

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for index in trainable_indices:
        for parameter in blocks[index].parameters():
            parameter.requires_grad = True

    frozen = tuple(
        block for index, block in enumerate(blocks) if index not in trainable_indices
    )
    trainable = tuple(blocks[index] for index in trainable_indices)
    return frozen, trainable


def _group_parameters(modules: Sequence[nn.Module]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def create_progressive_optimizer(
    model: nn.Module,
    classifier_learning_rate: float = DEFAULT_CLASSIFIER_LR,
) -> torch.optim.AdamW:
    """Create the single AdamW instance that lives across all three stages."""
    return torch.optim.AdamW(
        [
            {
                "params": list(model.classifier.parameters()),
                "lr": classifier_learning_rate,
                "configured_lr": classifier_learning_rate,
                "name": "classifier",
            }
        ],
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )


def _add_optimizer_group(
    optimizer: torch.optim.AdamW,
    parameters: Sequence[nn.Parameter],
    *,
    name: str,
    learning_rate: float,
) -> None:
    existing_names = {str(group.get("name")) for group in optimizer.param_groups}
    if name in existing_names:
        raise ValueError(f"Optimizer group {name!r} already exists.")
    existing_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    overlap = [parameter for parameter in parameters if parameter in existing_parameters]
    if overlap:
        raise ValueError(f"Optimizer group {name!r} contains existing parameters.")
    optimizer.add_param_group(
        {
            "params": list(parameters),
            "lr": learning_rate,
            "configured_lr": learning_rate,
            "name": name,
        }
    )


def _restore_configured_learning_rates(optimizer: torch.optim.AdamW) -> None:
    """Reset every group before attaching a fresh stage-local scheduler."""
    for group in optimizer.param_groups:
        configured = float(group["configured_lr"])
        group["lr"] = configured
        group["initial_lr"] = configured


def transition_to_stage2(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    *,
    late_blocks_learning_rate: float = DEFAULT_LATE_BLOCKS_LR,
    stage_epochs: int = DEFAULT_STAGE2_EPOCHS,
) -> tuple[
    tuple[nn.Module, ...],
    tuple[nn.Module, ...],
    torch.optim.lr_scheduler.CosineAnnealingLR,
]:
    """Unfreeze blocks 6-8 without replacing existing AdamW state."""
    frozen, trainable = configure_progressive_stage(model, "stage2")
    _add_optimizer_group(
        optimizer,
        _group_parameters(trainable),
        name="blocks_6_8",
        learning_rate=late_blocks_learning_rate,
    )
    _restore_configured_learning_rates(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage_epochs,
    )
    return frozen, trainable, scheduler


def transition_to_stage3(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    *,
    middle_blocks_learning_rate: float = DEFAULT_MIDDLE_BLOCKS_LR,
    stage_epochs: int = DEFAULT_STAGE3_EPOCHS,
) -> tuple[
    tuple[nn.Module, ...],
    tuple[nn.Module, ...],
    torch.optim.lr_scheduler.CosineAnnealingLR,
]:
    """Add blocks 4-5 and restart only the scheduler state."""
    frozen, trainable = configure_progressive_stage(model, "stage3")
    middle_blocks = get_efficientnet_feature_blocks(model)[4:6]
    _add_optimizer_group(
        optimizer,
        _group_parameters(middle_blocks),
        name="blocks_4_5",
        learning_rate=middle_blocks_learning_rate,
    )
    _restore_configured_learning_rates(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage_epochs,
    )
    return frozen, trainable, scheduler


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Return current named learning rates for logs and metadata."""
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def clone_model_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Take a true snapshot that later live-model updates cannot mutate."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _display_validation_source(source_name: str) -> str:
    lowered = source_name.casefold()
    if lowered.endswith(" test"):
        return source_name[:-5] + " held-out validation"
    if " test " in lowered:
        position = lowered.index(" test ")
        return source_name[:position] + " held-out validation " + source_name[position + 6 :]
    return source_name


def verify_heldout_validation_contract(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    heldout_generator: str,
) -> dict[str, object]:
    """Prove the held-out FAKE source is absent from training before training."""
    training_samples = getattr(train_loader.dataset, "samples", None)
    validation_samples = getattr(validation_loader.dataset, "samples", None)
    training_sources = getattr(train_loader.dataset, "sources", None)
    validation_sources = getattr(validation_loader.dataset, "sources", None)
    if not isinstance(training_samples, list) or not isinstance(validation_samples, list):
        raise TypeError("Progressive training requires source-aware dataset samples.")
    if not isinstance(training_sources, Sequence) or not isinstance(
        validation_sources,
        Sequence,
    ):
        raise TypeError("Progressive training requires source-aware dataset sources.")

    heldout_train = f"WildFake {heldout_generator} train".casefold()
    heldout_validation = f"WildFake {heldout_generator} test".casefold()
    if any(str(sample[2]).casefold() == heldout_train for sample in training_samples):
        raise AssertionError(
            f"Held-out generator {heldout_generator!r} contributed training samples."
        )

    heldout_samples = [
        sample
        for sample in validation_samples
        if str(sample[2]).casefold() == heldout_validation
    ]
    if not heldout_samples or any(int(sample[1]) != FAKE_LABEL for sample in heldout_samples):
        raise ValueError(
            f"Held-out-generator validation has no unique FAKE source for "
            f"{heldout_generator!r}."
        )
    invalid_fake_sources = {
        str(sample[2])
        for sample in validation_samples
        if int(sample[1]) == FAKE_LABEL
        and str(sample[2]).casefold() != heldout_validation
    }
    if invalid_fake_sources:
        raise ValueError(
            "Held-out-generator validation contains unexpected FAKE sources: "
            + ", ".join(sorted(invalid_fake_sources, key=str.casefold))
        )
    if not any(int(sample[1]) == REAL_LABEL for sample in validation_samples):
        raise ValueError("Held-out-generator validation requires REAL images.")

    train_fake = sorted(
        (source.name for source in training_sources if source.label == FAKE_LABEL),
        key=str.casefold,
    )
    train_real = sorted(
        (source.name for source in training_sources if source.label == REAL_LABEL),
        key=str.casefold,
    )
    heldout_validation_sources = sorted(
        (_display_validation_source(source.name) for source in validation_sources),
        key=str.casefold,
    )
    print(
        f"Verified held-out generator {heldout_generator}: zero training samples; "
        f"{len(heldout_samples):,} held-out-validation images."
    )
    return {
        "train_fake_sources": train_fake,
        "train_real_sources": train_real,
        "heldout_validation_sources": heldout_validation_sources,
    }


def calculate_progressive_validation_metrics(
    validation: Mapping[str, object],
    heldout_generator: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> dict[str, object]:
    """Add held-out FAKE-positive and class-balanced metrics to evaluation."""
    probabilities = [float(value) for value in validation["probabilities"]]
    labels = [int(value) for value in validation["labels"]]
    source_names = [str(value) for value in validation["source_names"]]
    source_metrics = calculate_source_metrics(
        probabilities,
        labels,
        source_names,
        heldout_generator,
        probability_threshold,
    )
    real_positions = [index for index, label in enumerate(labels) if label == REAL_LABEL]
    if not real_positions:
        raise ValueError("Held-out-generator validation requires REAL images.")
    predictions = [int(probability >= probability_threshold) for probability in probabilities]
    real_recall = sum(predictions[index] == REAL_LABEL for index in real_positions) / len(
        real_positions
    )
    heldout_fake_recall = float(source_metrics["heldout_generator_recall"])
    balanced_accuracy = (heldout_fake_recall + real_recall) / 2.0
    pooled_auc = validation.get("auc_roc")
    if pooled_auc is None:
        raise ValueError("Held-out-generator validation AUC requires both classes.")
    return {
        **source_metrics,
        "pooled_validation_auc_roc": float(pooled_auc),
        "heldout_fake_recall": heldout_fake_recall,
        "real_recall": real_recall,
        "balanced_accuracy": balanced_accuracy,
    }


def _is_better_candidate(
    heldout_auc: float,
    balanced_accuracy: float,
    best_key: tuple[float, float],
) -> bool:
    return (heldout_auc, balanced_accuracy) > best_key


def _validate_settings(
    stage1_epochs: int,
    stage2_epochs: int,
    stage3_epochs: int,
    learning_rates: Sequence[float],
) -> None:
    if min(stage1_epochs, stage2_epochs, stage3_epochs) < 1:
        raise ValueError("Every progressive training stage must have at least one epoch.")
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in learning_rates):
        raise ValueError("Progressive learning rates must be finite and greater than zero.")


def _print_stage_configuration(
    model: nn.Module,
    stage_label: str,
    frozen: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
) -> None:
    frozen_parameters = sum(
        parameter.numel()
        for module in frozen
        for parameter in module.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    rates = ", ".join(
        f"{name} @ {rate:.6g}" for name, rate in optimizer_learning_rates(optimizer).items()
    )
    print(f"\n--- {stage_label} ---")
    print(f"Frozen feature parameters: {frozen_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Configured optimizer groups: {rates}")


def _write_checkpoint_exclusively(path: Path, payload: Mapping[str, object]) -> None:
    """Create one new checkpoint without an overwrite-capable code path."""
    try:
        with path.open("xb") as checkpoint_file:
            torch.save(dict(payload), checkpoint_file)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite checkpoint created during training: {path}"
        ) from error


def train_progressive_deeper(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    heldout_generator: str,
    checkpoint_path: str | Path,
    run_name: str,
    samples_per_epoch: int,
    seed: int,
    stage1_epochs: int = DEFAULT_STAGE1_EPOCHS,
    stage2_epochs: int = DEFAULT_STAGE2_EPOCHS,
    stage3_epochs: int = DEFAULT_STAGE3_EPOCHS,
    classifier_learning_rate: float = DEFAULT_CLASSIFIER_LR,
    late_blocks_learning_rate: float = DEFAULT_LATE_BLOCKS_LR,
    middle_blocks_learning_rate: float = DEFAULT_MIDDLE_BLOCKS_LR,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> dict[str, object]:
    """Run all progressive stages and create one isolated best checkpoint."""
    _validate_settings(
        stage1_epochs,
        stage2_epochs,
        stage3_epochs,
        (
            classifier_learning_rate,
            late_blocks_learning_rate,
            middle_blocks_learning_rate,
        ),
    )
    output_path = Path(checkpoint_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {output_path}"
        )

    source_metadata = verify_heldout_validation_contract(
        train_loader,
        validation_loader,
        heldout_generator,
    )
    print_efficientnet_structure(model)
    criterion = nn.BCEWithLogitsLoss()
    frozen, trainable = configure_progressive_stage(model, "stage1")
    optimizer = create_progressive_optimizer(model, classifier_learning_rate)
    _print_stage_configuration(model, "Stage 1 - classifier only", frozen, optimizer)

    total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
    global_epoch = 0
    history: list[dict[str, object]] = []
    best_key = (float("-inf"), float("-inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, object] | None = None

    def run_stage(
        stage_key: str,
        stage_label: str,
        stage_epochs: int,
        frozen_modules: Sequence[nn.Module],
        trainable_modules: Sequence[nn.Module],
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None,
    ) -> None:
        nonlocal global_epoch, best_key, best_state, best_metrics
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            learning_rates = optimizer_learning_rates(optimizer)
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                probability_threshold,
                frozen_modules=frozen_modules,
                trainable_modules=(*trainable_modules, model.classifier),
            )
            validation = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                probability_threshold,
                description=f"Held-out-generator validation ({stage_label})",
            )
            metrics = calculate_progressive_validation_metrics(
                validation,
                heldout_generator,
                probability_threshold,
            )
            epoch_metrics: dict[str, object] = {
                "stage": stage_key,
                "stage_label": stage_label,
                "stage_epoch": stage_epoch,
                "epoch": global_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "learning_rates": learning_rates,
                **metrics,
            }
            history.append(epoch_metrics)
            rate_text = ", ".join(
                f"{name}: {rate:.6g}" for name, rate in learning_rates.items()
            )
            print(
                f"Stage: {stage_label} | Epoch {stage_epoch}/{stage_epochs} "
                f"(global {global_epoch}/{total_epochs}) | "
                f"Train loss: {train_loss:.4f} | "
                f"Pooled validation AUC: {metrics['pooled_validation_auc_roc']:.4f} | "
                f"Held-out-generator AUC: "
                f"{metrics['heldout_generator_auc_roc']:.4f} | "
                f"Held-out FAKE recall: {metrics['heldout_fake_recall']:.4f} | "
                f"REAL recall: {metrics['real_recall']:.4f} | "
                f"Balanced accuracy: {metrics['balanced_accuracy']:.4f} | "
                f"Learning rates: {rate_text}"
            )
            candidate_auc = float(metrics["heldout_generator_auc_roc"])
            candidate_balanced = float(metrics["balanced_accuracy"])
            if _is_better_candidate(candidate_auc, candidate_balanced, best_key):
                best_key = (candidate_auc, candidate_balanced)
                best_state = clone_model_state_to_cpu(model)
                best_metrics = copy.deepcopy(epoch_metrics)
                print(
                    "Stored a new best in-memory snapshot "
                    f"(held-out AUC={candidate_auc:.4f}, "
                    f"balanced accuracy={candidate_balanced:.4f})."
                )
            if scheduler is not None:
                scheduler.step()

    run_stage(
        "stage1",
        "Stage 1 - classifier only",
        stage1_epochs,
        frozen,
        trainable,
        None,
    )

    frozen, trainable, stage2_scheduler = transition_to_stage2(
        model,
        optimizer,
        late_blocks_learning_rate=late_blocks_learning_rate,
        stage_epochs=stage2_epochs,
    )
    _print_stage_configuration(model, "Stage 2 - late blocks", frozen, optimizer)
    run_stage(
        "stage2",
        "Stage 2 - late blocks",
        stage2_epochs,
        frozen,
        trainable,
        stage2_scheduler,
    )

    frozen, trainable, stage3_scheduler = transition_to_stage3(
        model,
        optimizer,
        middle_blocks_learning_rate=middle_blocks_learning_rate,
        stage_epochs=stage3_epochs,
    )
    _print_stage_configuration(model, "Stage 3 - deeper partial", frozen, optimizer)
    run_stage(
        "stage3",
        "Stage 3 - deeper partial",
        stage3_epochs,
        frozen,
        trainable,
        stage3_scheduler,
    )

    if best_state is None or best_metrics is None:
        raise RuntimeError("Progressive training completed without a best model state.")

    optimizer_defaults = {
        key: value
        for key, value in optimizer.defaults.items()
        if key != "params"
    }
    payload: dict[str, object] = {
        "model_state_dict": best_state,
        "model_type": "efficientnet",
        "training_variant": "progressive_deeper",
        "initialization_source": "ImageNet-pretrained EfficientNet-B0",
        "run_name": run_name,
        "stage": best_metrics["stage"],
        "stage_label": best_metrics["stage_label"],
        "stage_epoch": best_metrics["stage_epoch"],
        "epoch": best_metrics["epoch"],
        "heldout_generator": heldout_generator,
        "heldout_validation_role": "held-out-generator validation",
        **source_metadata,
        "stage_epoch_counts": {
            "stage1": stage1_epochs,
            "stage2": stage2_epochs,
            "stage3": stage3_epochs,
        },
        "base_learning_rates": {
            "classifier": classifier_learning_rate,
            "blocks_6_8": late_blocks_learning_rate,
            "blocks_4_5": middle_blocks_learning_rate,
        },
        "learning_rates": best_metrics["learning_rates"],
        "optimizer_type": "AdamW",
        "optimizer_defaults": optimizer_defaults,
        "scheduler_configuration": {
            "stage1": None,
            "stage2": {
                "type": "CosineAnnealingLR",
                "t_max": stage2_epochs,
                "restart": True,
            },
            "stage3": {
                "type": "CosineAnnealingLR",
                "t_max": stage3_epochs,
                "restart": True,
            },
        },
        "seed": seed,
        "samples_per_epoch": samples_per_epoch,
        "probability_threshold": probability_threshold,
        "selection_metric": "heldout_generator_auc_roc_then_balanced_accuracy",
        "selection_metric_value": best_key[0],
        "best_heldout_generator_auc_roc": best_key[0],
        "best_balanced_accuracy": best_key[1],
        "heldout_generator_auc_roc": best_metrics["heldout_generator_auc_roc"],
        "heldout_fake_recall": best_metrics["heldout_fake_recall"],
        "real_recall": best_metrics["real_recall"],
        "balanced_accuracy": best_metrics["balanced_accuracy"],
        "validation_auc_roc": best_metrics["pooled_validation_auc_roc"],
        "validation_loss": best_metrics["validation_loss"],
        "validation_accuracy": best_metrics["validation_accuracy"],
    }
    _write_checkpoint_exclusively(output_path, payload)
    print(f"Created new progressive EfficientNet checkpoint: {output_path}")
    return {
        "checkpoint_path": output_path,
        "best_heldout_generator_auc_roc": best_key[0],
        "best_balanced_accuracy": best_key[1],
        "history": history,
    }
