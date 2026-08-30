"""Training loops and best-checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .evaluate import evaluate
from .source_balanced import calculate_source_metrics, print_source_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    probability_threshold: float = 0.5,
    frozen_modules: Sequence[nn.Module] = (),
) -> tuple[float, float]:
    """Train the currently unfrozen model parameters for one complete epoch."""
    model.train()
    # model.train() also enables BatchNorm updates and stochastic depth. Restore
    # fully frozen feature blocks to evaluation mode for stable frozen features.
    for module in frozen_modules:
        module.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = (torch.sigmoid(outputs) >= probability_threshold).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("The training DataLoader is empty.")
    return running_loss / total, correct / total


def configure_head_only(model: nn.Module) -> tuple[nn.Module, ...]:
    """Freeze the backbone and leave only the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return tuple(model.features.children())


def configure_partial_unfreezing(
    model: nn.Module,
    trainable_feature_blocks: int = 3,
) -> tuple[tuple[nn.Module, ...], tuple[nn.Module, ...]]:
    """Unfreeze the final top-level feature blocks and the classifier."""
    feature_blocks = tuple(model.features.children())
    if not 1 <= trainable_feature_blocks <= len(feature_blocks):
        raise ValueError(
            "trainable_feature_blocks must be between 1 and the number of feature blocks."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    frozen_blocks = feature_blocks[:-trainable_feature_blocks]
    trainable_blocks = feature_blocks[-trainable_feature_blocks:]
    for block in trainable_blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return frozen_blocks, trainable_blocks


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Read the named optimizer learning rates for logs and checkpoints."""
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _run_staged_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    frozen_modules: Sequence[nn.Module],
    stage: str,
    stage_epoch: int,
    stage_epochs: int,
    global_epoch: int,
    total_epochs: int,
    probability_threshold: float,
    heldout_generator: str | None = None,
) -> dict[str, object]:
    """Run one staged epoch, evaluate it, and assemble its logged metrics."""
    learning_rates = _learning_rates(optimizer)
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        probability_threshold,
        frozen_modules=frozen_modules,
    )
    validation = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        probability_threshold,
        description=f"Validation ({stage})",
    )
    validation_auc = validation["auc_roc"]
    if validation_auc is None:
        raise ValueError(
            "Validation ROC-AUC requires at least one sample from each class."
        )

    metrics: dict[str, object] = {
        "stage": stage,
        "stage_epoch": stage_epoch,
        "epoch": global_epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": float(validation["loss"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_auc_roc": float(validation_auc),
        "learning_rates": learning_rates,
    }
    if heldout_generator is not None:
        source_names = validation.get("source_names")
        if not isinstance(source_names, list):
            raise ValueError(
                "Held-out validation requires source names from its validation dataset."
            )
        source_metrics = calculate_source_metrics(
            list(validation["probabilities"]),
            list(validation["labels"]),
            source_names,
            heldout_generator,
            probability_threshold,
        )
        metrics.update(source_metrics)
    learning_rate_text = ", ".join(
        f"{name}: {learning_rate:.6g}" for name, learning_rate in learning_rates.items()
    )
    print(
        f"Stage: {stage} | Epoch {stage_epoch}/{stage_epochs} "
        f"(global {global_epoch}/{total_epochs}) | "
        f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f} | "
        f"Validation loss: {metrics['validation_loss']:.4f}, "
        f"accuracy: {metrics['validation_accuracy']:.4f}, "
        f"ROC-AUC: {metrics['validation_auc_roc']:.4f} | "
        f"Learning rate(s): {learning_rate_text}"
    )
    if heldout_generator is not None:
        print_source_metrics(metrics)
    return metrics


def _save_staged_checkpoint(
    model: nn.Module,
    metrics: dict[str, object],
    checkpoint_path: Path,
    selection_metric_name: str,
    best_selection_value: float,
    checkpoint_metadata: Mapping[str, object] | None = None,
) -> None:
    """Save model weights together with enough metadata to understand the run."""
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "stage": metrics["stage"],
        "stage_epoch": metrics["stage_epoch"],
        "epoch": metrics["epoch"],
        "validation_loss": metrics["validation_loss"],
        "validation_accuracy": metrics["validation_accuracy"],
        "validation_auc_roc": metrics["validation_auc_roc"],
        "learning_rates": metrics["learning_rates"],
        "selection_metric": selection_metric_name,
        "selection_metric_value": float(metrics[selection_metric_name]),
        "best_selection_metric_value": best_selection_value,
    }
    if selection_metric_name == "validation_auc_roc":
        payload["best_validation_auc_roc"] = best_selection_value
    else:
        payload.update(
            {
                "heldout_generator": metrics["heldout_generator"],
                "heldout_generator_recall": metrics["heldout_generator_recall"],
                "heldout_generator_auc_roc": metrics["heldout_generator_auc_roc"],
                "best_heldout_generator_auc_roc": best_selection_value,
                "macro_source_recall": metrics["macro_source_recall"],
                "overall_auc_roc": metrics["overall_auc_roc"],
                "source_recalls": metrics["source_recalls"],
            }
        )
    if checkpoint_metadata:
        conflicting = sorted(set(payload) & set(checkpoint_metadata))
        if conflicting:
            raise ValueError(
                "Checkpoint metadata cannot replace reserved keys: "
                + ", ".join(conflicting)
            )
        payload.update(checkpoint_metadata)
    torch.save(payload, checkpoint_path)


def train_staged_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    stage2_epochs: int = 5,
    checkpoint_path: str | Path = "checkpoints/efficientnet_staged_best.pt",
    probability_threshold: float = 0.5,
    heldout_generator: str | None = None,
    stage1_epochs: int = 2,
    stage1_classifier_learning_rate: float = 1e-3,
    stage2_classifier_learning_rate: float = 1e-4,
    stage2_backbone_learning_rate: float = 1e-5,
    checkpoint_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train the head and final feature blocks, saving the best validation model.

    Ordinary staged runs select pooled validation AUC. Source-balanced runs pass
    a held-out generator and instead select that generator's FAKE-positive AUC.
    """
    if stage2_epochs < 1:
        raise ValueError("stage2_epochs must be at least 1.")
    if stage1_epochs < 1:
        raise ValueError("stage1_epochs must be at least 1.")

    total_epochs = stage1_epochs + stage2_epochs
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selection_metric_name = (
        "heldout_generator_auc_roc"
        if heldout_generator is not None
        else "validation_auc_roc"
    )
    best_selection_value = float("-inf")
    history: list[dict[str, object]] = []
    global_epoch = 0

    frozen_blocks = configure_head_only(model)
    stage1_optimizer = torch.optim.AdamW(
        [
            {
                "params": model.classifier.parameters(),
                "lr": stage1_classifier_learning_rate,
                "name": "classifier",
            }
        ],
        weight_decay=1e-2,
    )

    for stage_epoch in range(1, stage1_epochs + 1):
        global_epoch += 1
        metrics = _run_staged_epoch(
            model,
            train_loader,
            validation_loader,
            criterion,
            stage1_optimizer,
            device,
            frozen_blocks,
            "head-only",
            stage_epoch,
            stage1_epochs,
            global_epoch,
            total_epochs,
            probability_threshold,
            heldout_generator,
        )
        history.append(metrics)
        selection_value = float(metrics[selection_metric_name])
        if selection_value > best_selection_value:
            best_selection_value = selection_value
            _save_staged_checkpoint(
                model,
                metrics,
                checkpoint_path,
                selection_metric_name,
                best_selection_value,
                checkpoint_metadata,
            )
            print(f"Saved best staged checkpoint to {checkpoint_path}")

    frozen_blocks, trainable_blocks = configure_partial_unfreezing(model, 3)
    backbone_parameters = [
        parameter
        for block in trainable_blocks
        for parameter in block.parameters()
        if parameter.requires_grad
    ]
    stage2_optimizer = torch.optim.AdamW(
        [
            {
                "params": model.classifier.parameters(),
                "lr": stage2_classifier_learning_rate,
                "name": "classifier",
            },
            {
                "params": backbone_parameters,
                "lr": stage2_backbone_learning_rate,
                "name": "backbone",
            },
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        stage2_optimizer,
        T_max=stage2_epochs,
    )

    for stage_epoch in range(1, stage2_epochs + 1):
        global_epoch += 1
        metrics = _run_staged_epoch(
            model,
            train_loader,
            validation_loader,
            criterion,
            stage2_optimizer,
            device,
            frozen_blocks,
            "partial-unfreezing",
            stage_epoch,
            stage2_epochs,
            global_epoch,
            total_epochs,
            probability_threshold,
            heldout_generator,
        )
        history.append(metrics)
        selection_value = float(metrics[selection_metric_name])
        if selection_value > best_selection_value:
            best_selection_value = selection_value
            _save_staged_checkpoint(
                model,
                metrics,
                checkpoint_path,
                selection_metric_name,
                best_selection_value,
                checkpoint_metadata,
            )
            print(f"Saved best staged checkpoint to {checkpoint_path}")
        scheduler.step()

    result: dict[str, object] = {
        "best_auc_roc": best_selection_value,
        "selection_metric": selection_metric_name,
        "history": history,
    }
    if heldout_generator is not None:
        result["best_heldout_generator_auc_roc"] = best_selection_value
    return result


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    probability_threshold: float = 0.5,
) -> dict[str, object]:
    """Run the original full-fine-tuning baseline and save best accuracy."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device, probability_threshold
        )
        validation = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            probability_threshold,
            description="Validation",
        )
        validation_loss = float(validation["loss"])
        validation_accuracy = float(validation["accuracy"])
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f} | "
            f"Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path}")

    return {"best_accuracy": best_accuracy, "history": history}
