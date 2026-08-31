"""Explicit-label CIFAKE and WildFake dataset composition."""

from __future__ import annotations

import os
import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from .dataset import _find_split_root
from .transforms import build_eval_transforms, build_train_transforms


FAKE_LABEL = 0
REAL_LABEL = 1
TRAIN_SAMPLING_SEED = 42
SUPPORTED_IMAGE_EXTENSIONS = frozenset(extension.lower() for extension in IMG_EXTENSIONS)
LEGACY_SOURCE_DISPLAY_NAMES = {
    "cocofolder": "COCO",
    "laion5b": "LAION-5B",
}


@dataclass(frozen=True)
class ImageSource:
    """Define one recursively scanned root with an explicit source name and label."""
    name: str
    root: Path
    label: int


@dataclass(frozen=True)
class WildFakeSource:
    """One prepared WildFake source with explicit train and test roots."""

    name: str
    label: int
    train_root: Path
    test_root: Path


class MultiSourceImageDataset(Dataset):
    """Image dataset assembled from recursively scanned, explicitly labelled roots."""

    def __init__(
        self,
        sources: list[ImageSource],
        transform: Callable | None = None,
        source_fraction: float = 1.0,
        sampling_seed: int = TRAIN_SAMPLING_SEED,
        return_source: bool = False,
    ) -> None:
        """Discover each source and build one explicit-label list of image paths.

        ``return_source`` is used only by source-aware validation. Ordinary
        training keeps returning the familiar ``(image, label)`` pair.
        """
        if not sources:
            raise ValueError("At least one image source is required.")
        if not 0.0 < source_fraction <= 1.0:
            raise ValueError("source_fraction must be greater than 0 and at most 1.")

        self.transform = transform
        self.sources = tuple(sources)
        self.source_fraction = source_fraction
        self.sampling_seed = sampling_seed
        self.return_source = return_source
        self.samples: list[tuple[Path, int, str]] = []
        self.source_original_counts: dict[str, int] = {}
        self.source_counts: dict[str, int] = {}

        for source in self.sources:
            if source.label not in (FAKE_LABEL, REAL_LABEL):
                raise ValueError(
                    f"Source {source.name!r} has invalid binary label {source.label}."
                )
            source_root = source.root.resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(
                    f"Configured image source does not exist: {source_root}"
                )

            discovered = self._discover_images(source_root)
            if not discovered:
                raise ValueError(
                    f"Configured image source contains no supported images: {source_root}"
                )

            original_count = len(discovered)
            selected_count = int(original_count * source_fraction)
            if selected_count == 0:
                raise ValueError(
                    f"source_fraction={source_fraction} selects no images from "
                    f"{source.name!r}, which contains {original_count} image(s)."
                )
            selected = self._select_images(
                discovered,
                selected_count,
                sampling_seed,
                source.name,
            )
            self.source_original_counts[source.name] = original_count
            self.source_counts[source.name] = len(selected)
            self.samples.extend(
                (image_path, source.label, source.name)
                for image_path in selected
            )

        self.class_counts = Counter(label for _, label, _ in self.samples)

    @staticmethod
    def _discover_images(source_root: Path) -> list[Path]:
        """Recursively find supported image files in a stable path order."""
        images: list[Path] = []
        for directory, child_directories, filenames in os.walk(
            source_root,
            followlinks=False,
        ):
            child_directories.sort(key=str.casefold)
            for filename in sorted(filenames, key=str.casefold):
                if Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    images.append(Path(directory) / filename)
        return images

    @staticmethod
    def _select_images(
        discovered: list[Path],
        selected_count: int,
        sampling_seed: int,
        source_name: str,
    ) -> list[Path]:
        """Choose a repeatable subset of one source without changing its files."""
        if selected_count == len(discovered):
            return discovered
        seed_material = f"{sampling_seed}:{source_name}".encode("utf-8")
        source_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8])
        generator = random.Random(source_seed)
        return sorted(generator.sample(discovered, selected_count))

    def __len__(self) -> int:
        """Return the number of image paths available to the DataLoader."""
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        """Open one image as RGB, transform it, and return its explicit label."""
        image_path, label, source_name = self.samples[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            if self.transform is not None:
                rgb_image = self.transform(rgb_image)
        if self.return_source:
            return rgb_image, label, source_name
        return rgb_image, label


def _prepared_split_counts(source_root: Path) -> tuple[int, int] | None:
    """Return nonempty TRAIN/TEST counts for one prepared source."""
    train_root = source_root / "TRAIN"
    test_root = source_root / "TEST"
    if not train_root.is_dir() or not test_root.is_dir():
        return None
    train_count = len(MultiSourceImageDataset._discover_images(train_root))
    test_count = len(MultiSourceImageDataset._discover_images(test_root))
    if train_count == 0 or test_count == 0:
        return None
    return train_count, test_count


def discover_wildfake_sources(
    wildfake_root: str | Path,
    *,
    warn_invalid: bool = True,
) -> list[WildFakeSource]:
    """Discover every direct FAKE/REAL child with a usable prepared split."""
    wildfake_path = Path(wildfake_root).resolve()
    if not wildfake_path.is_dir():
        raise FileNotFoundError(
            "WildFake dataset directory not found: "
            f"{wildfake_path}. Run `python main.py prepare-data` after extracting "
            "sources under WildFake/FAKE or WildFake/REAL."
        )

    discovered: list[WildFakeSource] = []
    seen_names: dict[str, Path] = {}
    for label_name, label in (("FAKE", FAKE_LABEL), ("REAL", REAL_LABEL)):
        label_root = wildfake_path / label_name
        if not label_root.is_dir():
            if warn_invalid:
                print(f"WARNING: WildFake label directory is missing: {label_root}")
            continue
        source_roots = sorted(
            (path for path in label_root.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for source_root in source_roots:
            source_key = source_root.name.casefold()
            previous = seen_names.get(source_key)
            if previous is not None:
                raise ValueError(
                    "WildFake source names must be unique ignoring case; found "
                    f"both {previous} and {source_root}."
                )
            seen_names[source_key] = source_root
            if _prepared_split_counts(source_root) is None:
                if warn_invalid:
                    print(
                        f"WARNING: Skipping WildFake {label_name} source "
                        f"{source_root.name!r}; it needs nonempty TRAIN and TEST folders."
                    )
                continue
            discovered.append(
                WildFakeSource(
                    name=source_root.name,
                    label=label,
                    train_root=source_root / "TRAIN",
                    test_root=source_root / "TEST",
                )
            )

    if not discovered:
        raise ValueError(
            f"No prepared WildFake sources were found under {wildfake_path}. "
            "Run `python main.py prepare-data` first."
        )
    return discovered


def resolve_wildfake_holdout(
    wildfake_root: str | Path,
    holdout: str,
) -> str:
    """Resolve an arbitrary FAKE holdout name and validate its prepared split."""
    requested = holdout.strip()
    if not requested:
        raise ValueError("--holdout must name a WildFake FAKE source.")
    wildfake_path = Path(wildfake_root).resolve()
    fake_root = wildfake_path / "FAKE"
    if not fake_root.is_dir():
        raise FileNotFoundError(f"WildFake FAKE directory not found: {fake_root}")

    matches = [
        path
        for path in fake_root.iterdir()
        if path.is_dir() and path.name.casefold() == requested.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Holdout {holdout!r} is ambiguous because multiple FAKE sources match it."
        )
    if not matches:
        real_root = wildfake_path / "REAL"
        if real_root.is_dir() and any(
            path.is_dir() and path.name.casefold() == requested.casefold()
            for path in real_root.iterdir()
        ):
            raise ValueError(
                f"Holdout {holdout!r} is a REAL source; --holdout requires a FAKE source."
            )
        available = sorted(
            (path.name for path in fake_root.iterdir() if path.is_dir()),
            key=str.casefold,
        )
        available_text = ", ".join(available) if available else "none"
        raise ValueError(
            f"WildFake FAKE holdout {holdout!r} was not found. "
            f"Available FAKE sources: {available_text}."
        )

    source_root = matches[0]
    if _prepared_split_counts(source_root) is None:
        raise ValueError(
            f"WildFake FAKE holdout {source_root.name!r} needs nonempty "
            "TRAIN and TEST folders. Run `python main.py prepare-data` or fix the "
            "source layout."
        )
    return source_root.name


def build_multisource_sources(
    cifake_root: str | Path,
    wildfake_root: str | Path,
) -> tuple[list[ImageSource], list[ImageSource]]:
    """Describe CIFAKE plus every dynamically discovered WildFake source."""
    cifake_split_root = _find_split_root(Path(cifake_root))
    wildfake_sources = discover_wildfake_sources(wildfake_root)
    fake_sources = [source for source in wildfake_sources if source.label == FAKE_LABEL]
    real_sources = [source for source in wildfake_sources if source.label == REAL_LABEL]

    def display_name(source: WildFakeSource) -> str:
        """Use a stable human-readable label while retaining arbitrary sources."""
        return LEGACY_SOURCE_DISPLAY_NAMES.get(source.name.casefold(), source.name)

    training_sources = [
        ImageSource("CIFAKE train FAKE", cifake_split_root / "train" / "FAKE", FAKE_LABEL),
        *[
            ImageSource(
                f"WildFake {display_name(source)} train",
                source.train_root,
                source.label,
            )
            for source in fake_sources
        ],
        ImageSource("CIFAKE train REAL", cifake_split_root / "train" / "REAL", REAL_LABEL),
        *[
            ImageSource(
                f"WildFake {display_name(source)} train",
                source.train_root,
                source.label,
            )
            for source in real_sources
        ],
    ]
    validation_sources = [
        ImageSource("CIFAKE test FAKE", cifake_split_root / "test" / "FAKE", FAKE_LABEL),
        *[
            ImageSource(
                f"WildFake {display_name(source)} test",
                source.test_root,
                source.label,
            )
            for source in fake_sources
        ],
        ImageSource("CIFAKE test REAL", cifake_split_root / "test" / "REAL", REAL_LABEL),
        *[
            ImageSource(
                f"WildFake {display_name(source)} test",
                source.test_root,
                source.label,
            )
            for source in real_sources
        ],
    ]
    return training_sources, validation_sources


def _print_dataset_summary(
    title: str,
    dataset: MultiSourceImageDataset,
    show_selection: bool = False,
) -> None:
    """Print source paths and counts so the team can verify the input data."""
    print(f"\n{title} sources:")
    for source in dataset.sources:
        class_name = "FAKE" if source.label == FAKE_LABEL else "REAL"
        original_count = dataset.source_original_counts[source.name]
        selected_count = dataset.source_counts[source.name]
        if show_selection:
            count_text = f"{original_count:,} -> {selected_count:,} images"
        else:
            count_text = f"{selected_count:,} images"
        print(f"  [{class_name}={source.label}] {source.name}: {count_text}")
        print(f"      {source.root.resolve()}")
    print(
        f"{title} totals: FAKE={dataset.class_counts[FAKE_LABEL]:,}, "
        f"REAL={dataset.class_counts[REAL_LABEL]:,}, total={len(dataset):,}"
    )


def get_multisource_data_loaders(
    cifake_root: str | Path,
    wildfake_root: str | Path,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    train_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """Build the existing randomly shuffled multisource training workflow."""
    training_sources, validation_sources = build_multisource_sources(
        cifake_root,
        wildfake_root,
    )
    training_dataset = MultiSourceImageDataset(
        training_sources,
        transform=build_train_transforms(image_size),
        source_fraction=train_fraction,
        sampling_seed=TRAIN_SAMPLING_SEED,
    )
    validation_dataset = MultiSourceImageDataset(
        validation_sources,
        transform=build_eval_transforms(image_size),
    )

    print("\n--- CIFAKE + WildFake Multisource Dataset ---")
    print(
        f"Training source fraction: {train_fraction:.4f} "
        f"(deterministic seed {TRAIN_SAMPLING_SEED})"
    )
    _print_dataset_summary("Training", training_dataset, show_selection=True)
    _print_dataset_summary("Internal validation", validation_dataset)

    pin_memory = torch.cuda.is_available()
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
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
