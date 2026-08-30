"""Robustness benchmark under common image degradations."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .evaluate import evaluate
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AddGaussianNoise,
    RandomDownscaleUpscale,
    RandomJPEGCompression,
)


def _test_transforms(image_size: tuple[int, int]):
    """Build the fixed clean and degraded image scenarios used for robustness tests."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return {
        "clean": transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor(), normalize]
        ),
        "jpeg_quality_50": transforms.Compose(
            [
                transforms.Resize(image_size),
                RandomJPEGCompression(quality_range=(50, 50), p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "gaussian_blur_sigma_2": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.GaussianBlur(kernel_size=3, sigma=(2.0, 2.0)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "downscale_upscale_025": transforms.Compose(
            [
                transforms.Resize(image_size),
                RandomDownscaleUpscale(scale_factors=(0.25,), p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "gaussian_noise_sigma_010": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                AddGaussianNoise(std_range=(0.10, 0.10), p=1.0),
                normalize,
            ]
        ),
    }


def run_robustness_benchmark(
    model: nn.Module,
    test_dir: str | Path,
    device: torch.device,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    results_path: str | Path = "results/robustness.json",
) -> dict[str, dict[str, float | None]]:
    """Evaluate one model under every unchanged robustness scenario and save results."""
    criterion = nn.BCEWithLogitsLoss()
    results: dict[str, dict[str, float | None]] = {}

    for name, transform in _test_transforms(image_size).items():
        dataset = ImageFolder(test_dir, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        metrics = evaluate(model, loader, criterion, device, description=name)
        results[name] = {
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "auc_roc": None if metrics["auc_roc"] is None else float(metrics["auc_roc"]),
        }
        print(
            f"{name:<28} | accuracy: {results[name]['accuracy'] * 100:.2f}% | "
            f"AUC-ROC: {results[name]['auc_roc']}"
        )

    output_path = Path(results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved robustness results to {output_path}")
    return results
