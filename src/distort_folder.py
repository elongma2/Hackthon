"""Distort every image in one input folder; no labels or dataset splits needed.

This thin runner reuses the exact 19 transformations defined in
``src/distort_dataset2.py``.

Expected layout:
    data/input/example.jpg

Run from the project root:
    uv run python -m src.distort_folder \
        --input data/input --output data/output --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .distort_dataset2 import (
    EXTENSIONS,
    _output_name,
    _save_variant,
    _source_group_id,
    _stable_seed,
    required_variants,
)


def distort_folder(
    input_folder: Path,
    output_folder: Path,
    seed: int = 42,
    include_original: bool = False,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Generate all 19 required variants for every supported input image."""
    input_folder = input_folder.resolve()
    output_folder = output_folder.resolve()

    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if output_folder == input_folder or input_folder in output_folder.parents:
        raise ValueError("Output folder must not be inside the input folder")

    sources = sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not sources:
        raise ValueError(f"No supported images found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = output_folder / "distortions.csv"
    fieldnames = [
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

        for source in sources:
            relative = source.relative_to(input_folder)
            source_key = relative.as_posix()
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
                source_extension = source.suffix.lower().lstrip(".") or "none"
                clean_name = f"{source.stem}__src_{source_extension}__clean.png"
                clean_destination = output_folder / relative.parent / clean_name
                if overwrite or not clean_destination.exists():
                    clean_destination.parent.mkdir(parents=True, exist_ok=True)
                    original.save(clean_destination, format="PNG", compress_level=6)
                    written += 1
                writer.writerow(
                    {
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
                    output_folder
                    / relative.parent
                    / _output_name(source, variant)
                )
                if overwrite or not destination.exists():
                    _save_variant(variant, destination)
                    written += 1
                writer.writerow(
                    {
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
    """Parse folder-distortion arguments and generate the requested manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        generated, failures = distort_folder(
            input_folder=args.input,
            output_folder=args.output,
            seed=args.seed,
            include_original=args.include_original,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Generated {generated:,} images with {failures:,} failures. "
        f"Manifest: {args.output / 'distortions.csv'}"
    )


if __name__ == "__main__":
    main()
