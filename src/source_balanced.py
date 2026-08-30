"""Source-balanced training data and held-out-generator evaluation helpers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Sampler

from .multisource_dataset import (
    FAKE_LABEL,
    REAL_LABEL,
    ImageSource,
    MultiSourceImageDataset,
    _print_dataset_summary,
    build_multisource_sources,
)
from .transforms import build_eval_transforms, build_train_transforms


SUPPORTED_HOLDOUTS = ("ADM", "DDPM")
DEFAULT_BALANCED_SEED = 42
DEFAULT_SAMPLES_PER_EPOCH = 100_000
FAKE_REPORT_ORDER = ("CIFAKE test FAKE", "WildFake ADM test", "WildFake DDPM test")
REAL_REPORT_ORDER = (
    "CIFAKE test REAL",
    "WildFake COCO test",
    "WildFake LAION-5B test",
)


@dataclass
class _IndexCycle:
    """Serve shuffled indices from one source, refilling when it is exhausted."""

    indices: list[int]
    rng: random.Random
    position: int = 0

    def __post_init__(self) -> None:
        """Shuffle the first source-specific pass before sampling begins."""
        self.rng.shuffle(self.indices)

    def next(self) -> int:
        """Return one index and reshuffle after every complete source pass."""
        if self.position >= len(self.indices):
            self.rng.shuffle(self.indices)
            self.position = 0
        index = self.indices[self.position]
        self.position += 1
        return index


class SourceBalancedBatchSampler(Sampler[list[int]]):
    """Create fixed-size epochs balanced by class and configured data source."""

    def __init__(
        self,
        dataset: MultiSourceImageDataset,
        batch_size: int,
        samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
        seed: int = DEFAULT_BALANCED_SEED,
    ) -> None:
        """Index the dataset by class/source and validate sampler settings."""
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for class-balanced batches.")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be at least 1.")

        grouped: dict[int, dict[str, list[int]]] = {
            FAKE_LABEL: defaultdict(list),
            REAL_LABEL: defaultdict(list),
        }
        for index, (_, label, source_name) in enumerate(dataset.samples):
            grouped[label][source_name].append(index)
        for label, class_name in ((FAKE_LABEL, "FAKE"), (REAL_LABEL, "REAL")):
            if not grouped[label]:
                raise ValueError(f"Balanced sampling requires at least one {class_name} source.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._epoch = 0
        self._grouped = {
            label: {name: tuple(indices) for name, indices in sources.items()}
            for label, sources in grouped.items()
        }

    def __len__(self) -> int:
        """Return how many batches produce the requested samples per epoch."""
        return math.ceil(self.samples_per_epoch / self.batch_size)

    @staticmethod
    def _allocate_slots(
        total_slots: int,
        names: Sequence[str],
        offset: int,
    ) -> list[str]:
        """Spread class slots evenly across its active source names."""
        return [names[(offset + slot) % len(names)] for slot in range(total_slots)]

    def __iter__(self) -> Iterator[list[int]]:
        """Yield one deterministic but epoch-varying sequence of balanced batches."""
        epoch_rng = random.Random(self.seed + self._epoch)
        cycles: dict[tuple[int, str], _IndexCycle] = {}
        source_names: dict[int, list[str]] = {}
        for label, sources in self._grouped.items():
            names = sorted(sources)
            epoch_rng.shuffle(names)
            source_names[label] = names
            for source_name, indices in sources.items():
                source_seed = epoch_rng.randrange(0, 2**63)
                cycles[(label, source_name)] = _IndexCycle(
                    list(indices),
                    random.Random(source_seed),
                )

        emitted = 0
        batch_number = 0
        while emitted < self.samples_per_epoch:
            current_size = min(self.batch_size, self.samples_per_epoch - emitted)
            fake_slots = current_size // 2
            real_slots = current_size - fake_slots
            if current_size % 2 and (batch_number + self._epoch) % 2:
                fake_slots, real_slots = real_slots, fake_slots

            selected: list[int] = []
            for label, slots in ((FAKE_LABEL, fake_slots), (REAL_LABEL, real_slots)):
                names = source_names[label]
                chosen_sources = self._allocate_slots(slots, names, batch_number)
                selected.extend(cycles[(label, name)].next() for name in chosen_sources)
            epoch_rng.shuffle(selected)
            yield selected
            emitted += current_size
            batch_number += 1

        self._epoch += 1


def normalize_holdout(holdout: str) -> str:
    """Convert a CLI holdout name to the supported uppercase spelling."""
    normalized = holdout.upper()
    if normalized not in SUPPORTED_HOLDOUTS:
        choices = ", ".join(SUPPORTED_HOLDOUTS)
        raise ValueError(f"holdout must be one of: {choices}; received {holdout!r}.")
    return normalized


def build_heldout_sources(
    cifake_root: str | Path,
    wildfake_root: str | Path,
    holdout: str,
) -> tuple[list[ImageSource], list[ImageSource]]:
    """Exclude one fake generator from training and use its TEST images for validation."""
    normalized = normalize_holdout(holdout)
    training_sources, validation_sources = build_multisource_sources(
        cifake_root,
        wildfake_root,
    )
    heldout_train_name = f"WildFake {normalized} train"
    heldout_test_name = f"WildFake {normalized} test"
    selected_training = [
        source for source in training_sources if source.name != heldout_train_name
    ]
    selected_validation = [
        source
        for source in validation_sources
        if source.label == REAL_LABEL or source.name == heldout_test_name
    ]
    if len(selected_training) != len(training_sources) - 1:
        raise ValueError(f"Could not find configured training source for {normalized}.")
    if not any(source.name == heldout_test_name for source in selected_validation):
        raise ValueError(f"Could not find configured TEST source for {normalized}.")
    if any(source.name == heldout_train_name for source in selected_training):
        raise AssertionError(f"Held-out generator {normalized} entered training.")
    return selected_training, selected_validation


def get_source_balanced_data_loaders(
    cifake_root: str | Path,
    wildfake_root: str | Path,
    holdout: str,
    batch_size: int = 32,
    samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
    seed: int = DEFAULT_BALANCED_SEED,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build balanced training and complete held-out-generator validation loaders."""
    normalized = normalize_holdout(holdout)
    training_sources, validation_sources = build_heldout_sources(
        cifake_root,
        wildfake_root,
        normalized,
    )
    training_dataset = MultiSourceImageDataset(
        training_sources,
        transform=build_train_transforms(image_size),
    )
    validation_dataset = MultiSourceImageDataset(
        validation_sources,
        transform=build_eval_transforms(image_size),
        return_source=True,
    )
    batch_sampler = SourceBalancedBatchSampler(
        training_dataset,
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        seed=seed,
    )

    print("\n--- Source-Balanced Held-Out Generator Dataset ---")
    print(f"Held-out generator: {normalized}")
    print(f"Samples per epoch: {samples_per_epoch:,}")
    print(f"Deterministic sampler seed: {seed}")
    _print_dataset_summary("Training", training_dataset)
    _print_dataset_summary("Held-out validation", validation_dataset)

    pin_memory = torch.cuda.is_available()
    training_loader = DataLoader(
        training_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return training_loader, validation_loader


def calculate_source_metrics(
    raw_probabilities: Sequence[float],
    labels: Sequence[int],
    source_names: Sequence[str],
    holdout: str,
    probability_threshold: float = 0.5,
) -> dict[str, object]:
    """Calculate recalls and FAKE-positive AUC for one held-out generator."""
    normalized = normalize_holdout(holdout)
    if not (len(raw_probabilities) == len(labels) == len(source_names)):
        raise ValueError("Probability, label, and source counts must match.")
    if not labels:
        raise ValueError("Held-out validation contains no samples.")

    probabilities = list(map(float, raw_probabilities))
    numeric_labels = list(map(int, labels))
    predictions = [int(probability >= probability_threshold) for probability in probabilities]
    recalls: dict[str, float] = {}
    for source_name in sorted(set(source_names)):
        positions = [index for index, name in enumerate(source_names) if name == source_name]
        source_labels = {numeric_labels[index] for index in positions}
        if len(source_labels) != 1:
            raise ValueError(f"Source {source_name!r} contains mixed semantic labels.")
        label = next(iter(source_labels))
        recalls[source_name] = sum(
            predictions[index] == label for index in positions
        ) / len(positions)

    heldout_source = f"WildFake {normalized} test"
    heldout_positions = [
        index for index, name in enumerate(source_names) if name == heldout_source
    ]
    real_positions = [
        index for index, label in enumerate(numeric_labels) if label == REAL_LABEL
    ]
    if not heldout_positions:
        raise ValueError(f"Validation contains no samples for held-out {normalized}.")
    if not real_positions:
        raise ValueError("Held-out ROC-AUC requires REAL validation samples.")
    auc_positions = heldout_positions + real_positions
    aigc_labels = [1 if numeric_labels[index] == FAKE_LABEL else 0 for index in auc_positions]
    aigc_scores = [1.0 - probabilities[index] for index in auc_positions]
    overall_auc = roc_auc_score(numeric_labels, probabilities)

    return {
        "source_recalls": recalls,
        "macro_source_recall": sum(recalls.values()) / len(recalls),
        "overall_accuracy": sum(
            prediction == label for prediction, label in zip(predictions, numeric_labels)
        ) / len(numeric_labels),
        "overall_auc_roc": float(overall_auc),
        "heldout_generator": normalized,
        "heldout_generator_source": heldout_source,
        "heldout_generator_recall": recalls[heldout_source],
        "heldout_generator_auc_roc": float(roc_auc_score(aigc_labels, aigc_scores)),
    }


def print_source_metrics(metrics: dict[str, object]) -> None:
    """Print a stable, beginner-readable recall report for every known source."""
    recalls = metrics["source_recalls"]
    if not isinstance(recalls, dict):
        raise TypeError("source_recalls must be a dictionary.")

    print("\nPer-source validation recall:")
    print("FAKE:")
    for source_name in FAKE_REPORT_ORDER:
        value = recalls.get(source_name)
        text = "N/A - not part of this held-out validation" if value is None else f"{value:.4f}"
        print(f"  {source_name}: {text}")
    print("REAL:")
    for source_name in REAL_REPORT_ORDER:
        value = recalls.get(source_name)
        text = "N/A - not part of this held-out validation" if value is None else f"{value:.4f}"
        print(f"  {source_name}: {text}")
    print(f"Macro-average source recall: {metrics['macro_source_recall']:.4f}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Overall ROC-AUC: {metrics['overall_auc_roc']:.4f}")
    print(
        f"Held-out {metrics['heldout_generator']} recall: "
        f"{metrics['heldout_generator_recall']:.4f}"
    )
    print(
        f"Held-out {metrics['heldout_generator']} ROC-AUC (FAKE positive): "
        f"{metrics['heldout_generator_auc_roc']:.4f}"
    )
