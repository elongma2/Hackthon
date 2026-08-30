"""Create controlled image distortions while preserving ImageFolder labels.

Example:
    python -m src.distort_dataset --input data/raw --output data/distorted \
        --splits train test --copies 2 --seed 42

Important: use the same distortion policy for REAL and AI-generated images.
Otherwise a classifier can learn the augmentation artifact instead of the
image origin.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    result = Image.open(buffer).convert("RGB")
    result.load()
    return result


def distort(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    """Apply 1--3 randomly selected, realistic image corruptions."""
    image = image.convert("RGB")
    operations = ["jpeg", "blur", "downsample", "noise", "color", "perspective"]
    selected = rng.sample(operations, k=rng.randint(1, 3))
    applied: list[str] = []
    width, height = image.size

    for operation in selected:
        if operation == "jpeg":
            quality = rng.randint(35, 90)
            image = _jpeg(image, quality)
            applied.append(f"jpeg_quality={quality}")
        elif operation == "blur":
            radius = round(rng.uniform(0.3, 1.5), 2)
            image = image.filter(ImageFilter.GaussianBlur(radius))
            applied.append(f"blur_radius={radius}")
        elif operation == "downsample":
            scale = rng.choice((0.25, 0.5, 0.75))
            small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR)
            image = small.resize((width, height), Image.Resampling.BILINEAR)
            applied.append(f"downsample_scale={scale}")
        elif operation == "noise":
            # Pixel noise is generated in RGB space, with a modest amplitude.
            pixels = image.load()
            amount = rng.randint(3, 14)
            for y in range(height):
                for x in range(width):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in (r, g, b))
            applied.append(f"noise_amount={amount}")
        elif operation == "color":
            factor = round(rng.uniform(0.75, 1.25), 2)
            image = ImageEnhance.Contrast(image).enhance(factor)
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
            applied.append(f"color_factor={factor}")
        elif operation == "perspective":
            margin = rng.uniform(0.02, 0.08)
            dx, dy = width * margin, height * margin
            quad = (dx, dy, width - dx, 0, width, height - dy, 0, height)
            image = image.transform((width, height), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC)
            applied.append(f"perspective_margin={margin:.3f}")

    return image, "+".join(applied)


def generate(input_root: Path, output_root: Path, splits: list[str], copies: int, seed: int) -> int:
    rng = random.Random(seed)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    count = 0
    for split in splits:
        split_root = input_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        for source in sorted(p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
            relative = source.relative_to(split_root)
            with Image.open(source) as original:
                for copy_index in range(copies):
                    distorted, recipe = distort(original, rng)
                    destination = output_root / split / relative.parent / f"{relative.stem}__distorted_{copy_index:02d}.jpg"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    distorted.save(destination, format="JPEG", quality=95)
                    records.append({"split": split, "source": str(source), "output": str(destination), "copy": copy_index, "recipe": recipe})
                    count += 1
    with (output_root / "distortions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "source", "output", "copy", "recipe"])
        writer.writeheader()
        writer.writerows(records)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Root containing train/ and test/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "test"])
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.copies < 1:
        parser.error("--copies must be at least 1")
    print(f"Generated {generate(args.input, args.output, args.splits, args.copies, args.seed):,} images")


if __name__ == "__main__":
    main()
