"""Image augmentation and preprocessing pipelines."""

from __future__ import annotations

import io

import torch
import torchvision.transforms as transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEGCompression:
    """Randomly apply JPEG compression."""

    def __init__(self, quality_range: tuple[int, int] = (30, 90), p: float = 0.4):
        """Store the JPEG quality range and chance of applying compression."""
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        """Optionally recompress one image in memory to imitate social-media JPEG."""
        if torch.rand(1).item() >= self.p:
            return image

        quality = int(
            torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item()
        )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
        output.seek(0)
        # Loading the pixels before BytesIO closes avoids a dangling file handle.
        compressed = Image.open(output).convert("RGB")
        compressed.load()
        return compressed


class RandomDownscaleUpscale:
    """Randomly downscale an image and restore it to its original size."""

    def __init__(self, scale_factors: tuple[float, ...] = (0.25, 0.5), p: float = 0.3):
        """Store possible resize factors and the chance of applying one."""
        self.scale_factors = scale_factors
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        """Optionally shrink and restore an image to imitate online resizing."""
        if torch.rand(1).item() >= self.p:
            return image

        width, height = image.size
        index = int(torch.randint(0, len(self.scale_factors), (1,)).item())
        scale = self.scale_factors[index]
        low_resolution = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return low_resolution.resize((width, height), Image.Resampling.BILINEAR)


class AddGaussianNoise:
    """Randomly add Gaussian noise to an image tensor."""

    def __init__(self, std_range: tuple[float, float] = (0.02, 0.10), p: float = 0.3):
        """Store the noise-strength range and chance of adding tensor noise."""
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optionally add random Gaussian noise while keeping values valid."""
        if torch.rand(1).item() >= self.p:
            return tensor

        std = torch.empty(1).uniform_(*self.std_range).item()
        return torch.clamp(tensor + torch.randn_like(tensor) * std, 0.0, 1.0)


def build_train_transforms(image_size: tuple[int, int] = (224, 224)):
    """Build the one shared augmentation pipeline used for every training source."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomDownscaleUpscale(scale_factors=(0.25, 0.5), p=0.3),
            RandomJPEGCompression(quality_range=(30, 90), p=0.4),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.3
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            AddGaussianNoise(std_range=(0.02, 0.10), p=0.3),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: tuple[int, int] = (224, 224)):
    """Build deterministic resize and normalization steps for validation images."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
