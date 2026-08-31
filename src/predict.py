"""Single-image and batched folder inference for trained detector checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .model import expects_unnormalized_input, load_model
from .multisource_dataset import SUPPORTED_IMAGE_EXTENSIONS
from .transforms import build_eval_transforms


class UnlabelledImageDataset(Dataset):
    """Load a deterministic image list without requiring class directories."""

    def __init__(self, image_paths: Sequence[Path], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        """Store nonempty paths and the checkpoint-specific evaluation transform."""
        if not image_paths:
            raise ValueError("At least one image is required for prediction.")
        self.image_paths = tuple(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images that will be scored exactly once."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one RGB image and return its tensor with its stable list index."""
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError(f"Could not read image {image_path}: {error}") from error
        return tensor, index


def discover_input_images(input_dir: str | Path) -> list[Path]:
    """Recursively find supported images in deterministic relative-path order."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input image directory does not exist: {root}")
    discovered: list[Path] = []
    for directory, child_directories, filenames in os.walk(root, followlinks=False):
        child_directories.sort(key=str.casefold)
        for filename in sorted(filenames, key=str.casefold):
            if Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                discovered.append(Path(directory) / filename)
    discovered.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    if not discovered:
        extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Input directory contains no supported images: {root}. "
            f"Supported extensions: {extensions}"
        )
    return discovered


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device,
    image_size: tuple[int, int] = (224, 224),
    probability_threshold: float = 0.5,
) -> dict[str, float | str]:
    """Classify one image while preserving the legacy FAKE/REAL result format."""
    model.eval()
    with Image.open(image_path) as image:
        tensor = build_eval_transforms(
            image_size,
            normalize=not expects_unnormalized_input(model),
        )(image.convert("RGB"))
    probability_real = torch.sigmoid(model(tensor.unsqueeze(0).to(device))).item()
    label = "REAL" if probability_real >= probability_threshold else "FAKE"
    confidence = probability_real if label == "REAL" else 1.0 - probability_real
    return {
        "label": label,
        "confidence": confidence,
        "probability_real": probability_real,
        "probability_fake": 1.0 - probability_real,
    }


def predict_folder(
    *, input_dir: str | Path, checkpoint_path: str | Path, output_path: str | Path,
    device: torch.device, batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224), num_workers: int = 2,
) -> list[dict[str, object]]:
    """Score an unlabeled folder and write relative paths with threshold-free P(AIGC)."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if min(image_size) < 1:
        raise ValueError("image_size dimensions must be positive.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    root = Path(input_dir).resolve()
    image_paths = discover_input_images(root)

    # One generic load reconstructs EfficientNet or any supported hybrid from
    # strict metadata. It never falls back to random weights.
    model = load_model(checkpoint, device)
    model.eval()
    transform = build_eval_transforms(image_size, normalize=not expects_unnormalized_input(model))
    dataset = UnlabelledImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    results: list[dict[str, object] | None] = [None] * len(dataset)
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Scoring images"):
            logits = model(images.to(device))
            if logits.numel() != images.shape[0]:
                raise ValueError(
                    "Model must return exactly one logit per image; "
                    f"received {tuple(logits.shape)} for batch size {images.shape[0]}."
                )
            # sigmoid(logit)=P(REAL), so the submission confidence is P(AIGC).
            probabilities = 1.0 - torch.sigmoid(logits.reshape(-1))
            for index, probability in zip(indices.tolist(), probabilities.cpu().tolist()):
                results[int(index)] = {
                    "image_path": image_paths[int(index)].relative_to(root).as_posix(),
                    "pred": float(probability),
                }
    if any(result is None for result in results):
        raise RuntimeError("Prediction output is incomplete.")
    completed = [result for result in results if result is not None]
    destination = Path(output_path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"JSON output path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        json.dump(completed, output, indent=2)
        output.write("\n")
    print(f"Scored {len(completed):,} image(s).")
    print(f"Saved AIGC probabilities to {destination.resolve()}")
    return completed
