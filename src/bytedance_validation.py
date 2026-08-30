"""Inference-only evaluation on the ByteDance demonstration validation set."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .dataset import download_dataset
from .evaluate import evaluate
from .model import load_model
from .transforms import build_eval_transforms


EXPECTED_CLASSES = {"FAKE", "REAL"}
EXPECTED_FAKE_IMAGES = 8_843
EXPECTED_REAL_IMAGES = 4_998
EXPECTED_TOTAL_IMAGES = 13_841


def _validate_class_mapping(mapping: dict[str, int], dataset_name: str) -> None:
    """Stop evaluation when a dataset is not exactly binary FAKE versus REAL."""
    if set(mapping) != EXPECTED_CLASSES or set(mapping.values()) != {0, 1}:
        raise ValueError(
            f"{dataset_name} must contain exactly FAKE and REAL classes with binary "
            f"indices; found {mapping}."
        )


def get_class_counts(dataset: ImageFolder) -> dict[str, int]:
    """Return ImageFolder counts keyed by semantic class name."""
    _validate_class_mapping(dataset.class_to_idx, "Validation dataset")
    counts_by_index = Counter(dataset.targets)
    return {
        class_name: counts_by_index[class_index]
        for class_name, class_index in dataset.class_to_idx.items()
    }


def calculate_bytedance_metrics(
    training_mapping: dict[str, int],
    validation_mapping: dict[str, int],
    raw_probabilities: list[float],
    validation_labels: list[int],
    probability_threshold: float = 0.5,
) -> dict[str, object]:
    """Calculate semantic metrics with FAKE/AIGC as the positive class."""
    _validate_class_mapping(training_mapping, "CIFAKE training dataset")
    _validate_class_mapping(validation_mapping, "ByteDance validation dataset")
    if len(raw_probabilities) != len(validation_labels):
        raise ValueError("Probability and label counts do not match.")
    if not raw_probabilities:
        raise ValueError("The ByteDance validation dataset is empty.")

    training_class_by_index = {index: name for name, index in training_mapping.items()}
    validation_class_by_index = {index: name for name, index in validation_mapping.items()}
    raw_probability_class = training_class_by_index[1]

    if raw_probability_class == "FAKE":
        aigc_probabilities = list(raw_probabilities)
        aigc_probability_expression = "sigmoid(logit)"
    else:
        aigc_probabilities = [1.0 - probability for probability in raw_probabilities]
        aigc_probability_expression = "1 - sigmoid(logit)"

    actual_classes = [validation_class_by_index[label] for label in validation_labels]
    predicted_classes = [
        training_class_by_index[1 if probability >= probability_threshold else 0]
        for probability in raw_probabilities
    ]
    aigc_labels = [1 if class_name == "FAKE" else 0 for class_name in actual_classes]

    true_fake_predicted_fake = sum(
        actual == "FAKE" and predicted == "FAKE"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    true_fake_predicted_real = sum(
        actual == "FAKE" and predicted == "REAL"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    true_real_predicted_real = sum(
        actual == "REAL" and predicted == "REAL"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    true_real_predicted_fake = sum(
        actual == "REAL" and predicted == "FAKE"
        for actual, predicted in zip(actual_classes, predicted_classes)
    )
    correct = true_fake_predicted_fake + true_real_predicted_real
    total = len(validation_labels)

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "auc_roc_aigc": float(roc_auc_score(aigc_labels, aigc_probabilities)),
        "true_fake_predicted_fake": true_fake_predicted_fake,
        "true_fake_predicted_real": true_fake_predicted_real,
        "true_real_predicted_real": true_real_predicted_real,
        "true_real_predicted_fake": true_real_predicted_fake,
        "raw_probability_class": raw_probability_class,
        "aigc_probability_expression": aigc_probability_expression,
        "aigc_probabilities": aigc_probabilities,
        "aigc_labels": aigc_labels,
    }


def run_bytedance_validation(
    checkpoint_path: str | Path,
    validation_dir: str | Path,
    data_dir: str | Path,
    device: torch.device,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    probability_threshold: float = 0.5,
) -> dict[str, object]:
    """Inspect mappings and evaluate a checkpoint without any training state."""
    cifake_root = download_dataset(data_dir)
    train_dataset = ImageFolder(cifake_root / "train")
    _validate_class_mapping(train_dataset.class_to_idx, "CIFAKE training dataset")

    validation_path = Path(validation_dir).resolve()
    validation_dataset = ImageFolder(
        validation_path,
        transform=build_eval_transforms(image_size),
    )
    _validate_class_mapping(
        validation_dataset.class_to_idx,
        "ByteDance validation dataset",
    )
    class_counts = get_class_counts(validation_dataset)
    fake_count = class_counts["FAKE"]
    real_count = class_counts["REAL"]
    total_count = len(validation_dataset)

    training_class_by_index = {
        index: name for name, index in train_dataset.class_to_idx.items()
    }
    raw_probability_class = training_class_by_index[1]
    aigc_probability_expression = (
        "sigmoid(logit)" if raw_probability_class == "FAKE" else "1 - sigmoid(logit)"
    )

    print("\n--- ByteDance Validation Dataset ---\n")
    print("Training class mapping:")
    print(train_dataset.class_to_idx)
    print("\nValidation class mapping:")
    print(validation_dataset.class_to_idx)
    print(f"\nFAKE images: {fake_count}")
    print(f"REAL images: {real_count}")
    print(f"Total images: {total_count}")

    if (
        fake_count != EXPECTED_FAKE_IMAGES
        or real_count != EXPECTED_REAL_IMAGES
        or total_count != EXPECTED_TOTAL_IMAGES
    ):
        print("\nWARNING: ByteDance validation image count does not match expectations.")
        print(
            f"Expected FAKE={EXPECTED_FAKE_IMAGES}, REAL={EXPECTED_REAL_IMAGES}, "
            f"total={EXPECTED_TOTAL_IMAGES}."
        )
        print(f"Found FAKE={fake_count}, REAL={real_count}, total={total_count}.")
        print("Continuing with the images that were discovered.\n")

    print("Checkpoint:")
    print(checkpoint_path)
    print(f"\nRaw model probability represents: {raw_probability_class}")
    print("Evaluation positive class: FAKE / AIGC")
    print(
        "ByteDance AIGC probability used for evaluation: "
        f"{aigc_probability_expression}"
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(checkpoint_path, device)
    model.eval()
    inference = evaluate(
        model,
        validation_loader,
        nn.BCEWithLogitsLoss(),
        device,
        probability_threshold=probability_threshold,
        description="ByteDance validation",
    )
    metrics = calculate_bytedance_metrics(
        train_dataset.class_to_idx,
        validation_dataset.class_to_idx,
        list(inference["probabilities"]),
        list(inference["labels"]),
        probability_threshold,
    )

    print("\n--- Results ---\n")
    print(f"Total images: {metrics['total']}")
    print(f"Correctly classified: {metrics['correct']}")
    print(f"Incorrectly classified: {metrics['incorrect']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"ROC-AUC (AIGC positive): {metrics['auc_roc_aigc']:.4f}")
    print("\nConfusion Matrix:")
    print(
        "Actual FAKE -> Predicted FAKE: "
        f"{metrics['true_fake_predicted_fake']}"
    )
    print(
        "Actual FAKE -> Predicted REAL: "
        f"{metrics['true_fake_predicted_real']}"
    )
    print(
        "Actual REAL -> Predicted REAL: "
        f"{metrics['true_real_predicted_real']}"
    )
    print(
        "Actual REAL -> Predicted FAKE: "
        f"{metrics['true_real_predicted_fake']}"
    )

    return {
        "training_mapping": train_dataset.class_to_idx,
        "validation_mapping": validation_dataset.class_to_idx,
        "class_counts": class_counts,
        "checkpoint": str(checkpoint_path),
        **metrics,
    }
