"""Generate the exact robustness transformations from the specification.

Every required setting is applied independently to the original image. Derived
images retain the ImageFolder layout:

    INPUT/<split>/<class>/<image>
    OUTPUT/<split>/<class>/<derived image>

Install:
    uv add pillow numpy

Run from the project root:
    uv run python -m src.distort_dataset \
        --input data/raw --output data/distorted --splits train test --seed 42

Split original source images before running this program. Do not let variants
of one source image appear in different train/test/validation splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)  # Standard deviation on [0, 1] RGB values.
COLOR_FACTORS = (0.8, 1.2)  # -20% and +20%.
CROP_FRACTION = 0.8


@dataclass(frozen=True)
class Variant:
    transform: str
    tag: str
    parameters: dict[str, Any]
    image: Image.Image
    suffix: str
    jpeg_quality: int | None = None


def _tag_number(value: float) -> str:
    return str(value).replace(".", "p")


def _resize_down_then_up(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = image.resize(small_size, Image.Resampling.LANCZOS)
    return small.resize((width, height), Image.Resampling.LANCZOS)


def _gaussian_noise(
    image: Image.Image, sigma: float, seed: int
) -> Image.Image:
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=sigma, size=pixels.shape)
    result = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(result * 255).astype(np.uint8), mode="RGB")


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Retain the centered 80%; normal model preprocessing can resize it later."""
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _stable_seed(global_seed: int, source_key: str, transform_tag: str) -> int:
    value = f"{global_seed}\0{source_key}\0{transform_tag}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _source_group_id(source_key: str) -> str:
    return hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]


def required_variants(
    original: Image.Image, global_seed: int, source_key: str
) -> Iterator[Variant]:
    """Yield all 19 required variants, each made from the untouched original."""
    image = original.convert("RGB")

    # JPEG compression is performed exactly once when the variant is saved.
    for quality in JPEG_QUALITIES:
        yield Variant(
            "jpeg_compression",
            f"jpeg_q{quality}",
            {"quality": quality, "subsampling": "4:2:0"},
            image.copy(),
            ".jpg",
            jpeg_quality=quality,
        )

    for sigma in BLUR_SIGMAS:
        yield Variant(
            "gaussian_blur",
            f"blur_sigma_{_tag_number(sigma)}",
            {"sigma": sigma},
            image.filter(ImageFilter.GaussianBlur(radius=sigma)),
            ".png",
        )

    for scale in RESIZE_SCALES:
        yield Variant(
            "resize",
            f"resize_{_tag_number(scale)}x",
            {"downscale": scale, "upscale_to_original": True},
            _resize_down_then_up(image, scale),
            ".png",
        )

    for sigma in NOISE_SIGMAS:
        tag = f"noise_sigma_{_tag_number(sigma)}"
        variant_seed = _stable_seed(global_seed, source_key, tag)
        yield Variant(
            "gaussian_noise",
            tag,
            {"sigma": sigma, "value_range": "[0,1]"},
            _gaussian_noise(image, sigma, variant_seed),
            ".png",
        )

    # Test both stated boundaries independently for brightness, contrast and
    # saturation. Factors 0.8 and 1.2 mean -20% and +20% respectively.
    enhancers = {
        "brightness": ImageEnhance.Brightness,
        "contrast": ImageEnhance.Contrast,
        "saturation": ImageEnhance.Color,
    }
    for property_name, enhancer in enhancers.items():
        for factor in COLOR_FACTORS:
            yield Variant(
                "color_jitter",
                f"{property_name}_{_tag_number(factor)}",
                {property_name: factor},
                enhancer(image).enhance(factor),
                ".png",
            )

    yield Variant(
        "center_crop",
        "center_crop_0p8",
        {"retained_fraction": CROP_FRACTION, "resize_to_original": False},
        _center_crop(image, CROP_FRACTION),
        ".png",
    )


def _output_name(source: Path, variant: Variant) -> str:
    source_extension = source.suffix.lower().lstrip(".") or "none"
    return f"{source.stem}__src_{source_extension}__{variant.tag}{variant.suffix}"


def _save_variant(variant: Variant, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if variant.jpeg_quality is not None:
        variant.image.save(
            destination,
            format="JPEG",
            quality=variant.jpeg_quality,
            subsampling="4:2:0",
            optimize=False,
        )
    else:
        variant.image.save(destination, format="PNG", compress_level=6)


def _class_names(split_root: Path) -> list[str]:
    names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    folded: dict[str, str] = {}
    for name in names:
        key = name.casefold()
        if key in folded:
            raise ValueError(
                f"Duplicate class folders differing only by case: "
                f"{folded[key]!r} and {name!r}. Keep one spelling only."
            )
        folded[key] = name
    return names


def generate(
    input_root: Path,
    output_root: Path,
    splits: list[str],
    seed: int,
    include_original: bool,
    overwrite: bool,
) -> tuple[int, int]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_root}")
    if input_root == output_root:
        raise ValueError("Input and output directories must be different")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "distortions.csv"
    fieldnames = [
        "split",
        "class_label",
        "source",
        "source_group_id",
        "output",
        "transform",
        "parameters",
        "seed",
        "width",
        "height",
    ]
    written = 0
    failures = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fieldnames)
        writer.writeheader()

        for split in splits:
            split_root = input_root / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"Missing split directory: {split_root}")

            class_names = _class_names(split_root)
            if not class_names:
                raise ValueError(f"No class directories found in {split_root}")

            sources = sorted(
                path
                for path in split_root.rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            )
            for source in sources:
                relative = source.relative_to(split_root)
                if len(relative.parts) < 2:
                    raise ValueError(
                        f"Image must be inside a class folder: {source}. Expected "
                        f"{split_root}/<class>/<image>."
                    )
                class_label = relative.parts[0]
                source_key = f"{split}/{relative.as_posix()}"
                group_id = _source_group_id(source_key)

                try:
                    with Image.open(source) as opened:
                        original = ImageOps.exif_transpose(opened).convert("RGB")
                        original.load()
                except (OSError, UnidentifiedImageError) as exc:
                    failures += 1
                    print(f"WARNING: skipping unreadable image {source}: {exc}")
                    continue

                if include_original:
                    clean_name = (
                        f"{source.stem}__src_{source.suffix.lower().lstrip('.')}"
                        "__clean.png"
                    )
                    clean_destination = output_root / split / relative.parent / clean_name
                    if overwrite or not clean_destination.exists():
                        clean_destination.parent.mkdir(parents=True, exist_ok=True)
                        original.save(clean_destination, format="PNG", compress_level=6)
                        written += 1
                    writer.writerow(
                        {
                            "split": split,
                            "class_label": class_label,
                            "source": source.as_posix(),
                            "source_group_id": group_id,
                            "output": clean_destination.as_posix(),
                            "transform": "none",
                            "parameters": "{}",
                            "seed": seed,
                            "width": original.width,
                            "height": original.height,
                        }
                    )

                for variant in required_variants(original, seed, source_key):
                    destination = (
                        output_root
                        / split
                        / relative.parent
                        / _output_name(source, variant)
                    )
                    if overwrite or not destination.exists():
                        _save_variant(variant, destination)
                        written += 1
                    writer.writerow(
                        {
                            "split": split,
                            "class_label": class_label,
                            "source": source.as_posix(),
                            "source_group_id": group_id,
                            "output": destination.as_posix(),
                            "transform": variant.transform,
                            "parameters": json.dumps(
                                variant.parameters,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            "seed": _stable_seed(seed, source_key, variant.tag),
                            "width": variant.image.width,
                            "height": variant.image.height,
                        }
                    )

    return written, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Root containing split folders"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Existing split folder names, for example: train val test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Also save one clean PNG copy per source image",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output images"
    )
    args = parser.parse_args()

    try:
        generated, failures = generate(
            args.input,
            args.output,
            args.splits,
            args.seed,
            args.include_original,
            args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Generated {generated:,} images with {failures:,} failures. "
        f"Manifest: {args.output / 'distortions.csv'}"
    )


if __name__ == "__main__":
    main()
