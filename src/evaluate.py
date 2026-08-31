"""Model evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ComponentForward = Callable[
    [nn.Module, torch.Tensor],
    tuple[torch.Tensor, Mapping[str, torch.Tensor]],
]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    probability_threshold: float = 0.5,
    description: str = "Evaluating",
    component_forward: ComponentForward | None = None,
) -> dict[str, object]:
    """Run inference over a loader and return loss, accuracy, AUC, and scores.

    Source-aware validation batches may include a third item containing source
    names. Ordinary loaders still use the original ``(images, labels)`` shape.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    probabilities: list[float] = []
    labels_list: list[int] = []
    source_names: list[str] = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}
    expected_diagnostic_names: set[str] | None = None

    for batch in tqdm(dataloader, desc=description):
        if len(batch) == 2:
            images, labels = batch
            batch_sources = None
        elif len(batch) == 3:
            images, labels, batch_sources = batch
        else:
            raise ValueError("Evaluation batches must contain images, labels, and optional sources.")
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        if component_forward is None:
            outputs = model(images)
            component_logits: Mapping[str, torch.Tensor] = {}
        else:
            outputs, component_logits = component_forward(model, images)
            names = set(component_logits)
            if expected_diagnostic_names is None:
                expected_diagnostic_names = names
            elif names != expected_diagnostic_names:
                raise ValueError(
                    "Validation component names changed between batches."
                )
            for name, logits in component_logits.items():
                if not isinstance(name, str) or not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        "Validation components must map string names to tensors."
                    )
                if logits.numel() == 0:
                    raise ValueError(f"Validation component {name!r} is empty.")
                if logits.shape != outputs.shape:
                    raise ValueError(
                        f"Validation component {name!r} shape {tuple(logits.shape)} "
                        f"does not match output shape {tuple(outputs.shape)}."
                    )
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(
                    logits.detach().abs().sum().item()
                )
                diagnostic_counts[name] = (
                    diagnostic_counts.get(name, 0) + logits.numel()
                )
        loss = criterion(outputs, labels)
        probs = torch.sigmoid(outputs)
        predictions = (probs >= probability_threshold).float()

        running_loss += loss.item() * images.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        probabilities.extend(probs.view(-1).cpu().tolist())
        labels_list.extend(labels.view(-1).int().cpu().tolist())
        if batch_sources is not None:
            source_names.extend(str(source_name) for source_name in batch_sources)

    if total == 0:
        raise ValueError("The evaluation DataLoader is empty.")

    auc = roc_auc_score(labels_list, probabilities) if len(set(labels_list)) > 1 else None
    results: dict[str, object] = {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "auc_roc": auc,
        "probabilities": probabilities,
        "labels": labels_list,
    }
    if source_names:
        if len(source_names) != total:
            raise ValueError("Source-name count does not match evaluated samples.")
        results["source_names"] = source_names
    if diagnostic_sums:
        results["mean_absolute_branch_logits"] = {
            name: diagnostic_sums[name] / diagnostic_counts[name]
            for name in sorted(diagnostic_sums)
        }
    return results
