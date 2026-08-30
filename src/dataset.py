"""Dataset download and DataLoader construction."""

from __future__ import annotations

from pathlib import Path

import kagglehub
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import build_eval_transforms, build_train_transforms


DATASET_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
DATASET_DIRECTORY_NAME = "cifake"


def _find_split_root(dataset_path: Path) -> Path:
    """Find the directory that directly contains both train/ and test/."""
    if (dataset_path / "train").is_dir() and (dataset_path / "test").is_dir():
        return dataset_path

    for train_dir in dataset_path.rglob("train"):
        candidate = train_dir.parent
        if train_dir.is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find train/ and test/ under {dataset_path}")


def _find_existing_split_root(dataset_path: Path) -> Path | None:
    """Return an existing CIFAKE split directory, or None when it is absent."""
    try:
        return _find_split_root(dataset_path)
    except FileNotFoundError:
        return None


def download_dataset(data_dir: str | Path = "data/raw") -> Path:
    """Reuse local CIFAKE files or download them into a safe empty directory."""
    destination = Path(data_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    existing_dataset = _find_existing_split_root(destination)
    if existing_dataset is not None:
        print(f"Using existing dataset at {existing_dataset}")
        return existing_dataset

    download_destination = destination
    if any(destination.iterdir()):
        download_destination = destination / DATASET_DIRECTORY_NAME
        download_destination.mkdir(parents=True, exist_ok=True)

        existing_dataset = _find_existing_split_root(download_destination)
        if existing_dataset is not None:
            print(f"Using existing dataset at {existing_dataset}")
            return existing_dataset

        if any(download_destination.iterdir()):
            raise FileExistsError(
                f"Dataset download directory is not empty and does not contain "
                f"train/test splits: {download_destination}"
            )

    downloaded_path = Path(
        kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(download_destination))
    )
    return _find_split_root(downloaded_path)


def get_data_loaders(
    dataset_path: str | Path,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build the original shuffled CIFAKE train and ordered test loaders."""
    root = _find_split_root(Path(dataset_path))
    train_dataset = ImageFolder(root / "train", transform=build_train_transforms(image_size))
    test_dataset = ImageFolder(root / "test", transform=build_eval_transforms(image_size))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_test_directory(dataset_path: str | Path) -> Path:
    """Return the CIFAKE test folder used by the robustness benchmark."""
    return _find_split_root(Path(dataset_path)) / "test"
