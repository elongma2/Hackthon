"""Safely prepare recursively extracted WildFake-style image sources."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_PREPARATION_SEED = 42
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SPLIT_NAMES = frozenset({"TRAIN", "TEST"})


@dataclass(frozen=True)
class PreparationResult:
    """Describe how one labelled source was handled."""

    label: str
    source: str
    status: str
    train_count: int = 0
    test_count: int = 0


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _stable_relative_key(path: Path, root: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    return relative.casefold(), relative


def _discover_images(root: Path) -> list[Path]:
    """Recursively find supported images in a reproducible order."""
    return sorted(
        (path for path in root.rglob("*") if _is_supported_image(path)),
        key=lambda path: _stable_relative_key(path, root),
    )


def _split_directories(source_root: Path) -> list[Path]:
    """Find split-like directories without treating their contents as unsplit."""
    return sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_dir() and path.name.upper() in SPLIT_NAMES
        ),
        key=lambda path: _stable_relative_key(path, source_root),
    )


def _candidate_split_pairs(
    split_directories: list[Path],
) -> list[tuple[Path, Path, Path]]:
    by_parent: dict[Path, dict[str, list[Path]]] = {}
    for directory in split_directories:
        by_name = by_parent.setdefault(directory.parent, {"TRAIN": [], "TEST": []})
        by_name[directory.name.upper()].append(directory)

    candidates: list[tuple[Path, Path, Path]] = []
    for parent, by_name in by_parent.items():
        if len(by_name["TRAIN"]) == 1 and len(by_name["TEST"]) == 1:
            candidates.append((parent, by_name["TRAIN"][0], by_name["TEST"][0]))
    return sorted(candidates, key=lambda pair: str(pair[0]).casefold())


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_move_plan(moves: list[tuple[Path, Path]]) -> None:
    """Reject every known overwrite or collision before the first move."""
    seen_destinations: dict[str, Path] = {}
    seen_sources: set[Path] = set()
    for source, destination in moves:
        source_absolute = source.absolute()
        destination_absolute = destination.absolute()
        if source_absolute in seen_sources:
            raise ValueError(f"The move plan contains the source twice: {source}")
        seen_sources.add(source_absolute)
        if source_absolute == destination_absolute:
            raise ValueError(f"Source and destination are identical: {source}")

        collision_key = str(destination_absolute).casefold()
        previous = seen_destinations.get(collision_key)
        if previous is not None:
            raise FileExistsError(
                "Two images would use the same destination: "
                f"{previous} and {destination}"
            )
        seen_destinations[collision_key] = destination
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing destination: {destination}"
            )


def _execute_move_plan(moves: list[tuple[Path, Path]]) -> None:
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _result_warning(label: str, source: str, message: str) -> PreparationResult:
    print(f"[WARN] {source} [{label}]: {message}")
    return PreparationResult(label, source, "warning")


def _promote_nested_split(
    source_root: Path,
    label: str,
    source_name: str,
    pair: tuple[Path, Path, Path],
    all_images: list[Path],
) -> PreparationResult:
    parent, train_directory, test_directory = pair
    train_images = [path for path in all_images if _path_is_within(path, train_directory)]
    test_images = [path for path in all_images if _path_is_within(path, test_directory)]
    pair_images = set(train_images) | set(test_images)
    extra_split_directories = [
        path
        for path in _split_directories(source_root)
        if path not in (train_directory, test_directory)
    ]
    if not train_images or not test_images:
        return _result_warning(
            label,
            source_name,
            "the detected TRAIN/TEST pair has an empty split; files were left untouched.",
        )
    if pair_images != set(all_images) or extra_split_directories:
        return _result_warning(
            label,
            source_name,
            "images or additional split folders exist outside the detected pair; "
            "files were left untouched.",
        )

    wrapper = parent.relative_to(source_root)
    moves: list[tuple[Path, Path]] = []
    for split_name, split_directory, images in (
        ("TRAIN", train_directory, train_images),
        ("TEST", test_directory, test_images),
    ):
        for image in images:
            relative = image.relative_to(split_directory)
            destination = source_root / split_name
            if wrapper.parts:
                destination /= wrapper
            destination /= relative
            moves.append((image, destination))

    try:
        _validate_move_plan(moves)
        _execute_move_plan(moves)
    except (OSError, ValueError) as error:
        return _result_warning(
            label,
            source_name,
            f"could not safely promote the existing split: {error}",
        )

    print(f"[PROMOTE] {source_name} [{label}] existing TRAIN/TEST moved to source root")
    print(f"TRAIN: {len(train_images):,}")
    print(f"TEST: {len(test_images):,}")
    return PreparationResult(
        label,
        source_name,
        "promoted",
        len(train_images),
        len(test_images),
    )


def _prepare_source(
    source_root: Path,
    label: str,
    train_ratio: float,
    seed: int,
) -> PreparationResult:
    source_name = source_root.name
    all_images = _discover_images(source_root)
    split_directories = _split_directories(source_root)
    candidates = _candidate_split_pairs(split_directories)

    top_level_candidates = [pair for pair in candidates if pair[0] == source_root]
    if top_level_candidates:
        pair = top_level_candidates[0]
        _, train_directory, test_directory = pair
        train_images = [path for path in all_images if _path_is_within(path, train_directory)]
        test_images = [path for path in all_images if _path_is_within(path, test_directory)]
        pair_images = set(train_images) | set(test_images)
        canonical = train_directory.name == "TRAIN" and test_directory.name == "TEST"
        if (
            canonical
            and len(top_level_candidates) == 1
            and len(candidates) == 1
            and train_images
            and test_images
            and pair_images == set(all_images)
            and len(split_directories) == 2
        ):
            print(f"[SKIP] {source_name} already has TRAIN/TEST")
            print(f"TRAIN: {len(train_images):,}")
            print(f"TEST: {len(test_images):,}")
            return PreparationResult(
                label,
                source_name,
                "skipped",
                len(train_images),
                len(test_images),
            )
        return _result_warning(
            label,
            source_name,
            "the top-level split is empty, non-canonical, ambiguous, or has images "
            "outside it; files were left untouched.",
        )

    if split_directories:
        if len(candidates) != 1:
            return _result_warning(
                label,
                source_name,
                "a partial or ambiguous nested TRAIN/TEST structure was found; "
                "files were left untouched.",
            )
        return _promote_nested_split(
            source_root,
            label,
            source_name,
            candidates[0],
            all_images,
        )

    if not all_images:
        return _result_warning(label, source_name, "no supported images were found.")
    if len(all_images) < 2:
        return _result_warning(
            label,
            source_name,
            "at least two images are required to create nonempty TRAIN and TEST splits.",
        )

    shuffled = list(all_images)
    random.Random(seed).shuffle(shuffled)
    train_count = min(max(int(len(shuffled) * train_ratio), 1), len(shuffled) - 1)
    training_images = set(shuffled[:train_count])
    moves = [
        (
            image,
            source_root
            / ("TRAIN" if image in training_images else "TEST")
            / image.relative_to(source_root),
        )
        for image in shuffled
    ]

    print(f"Preparing {source_name} [{label}]")
    print(f"Found: {len(all_images):,} images")
    print(f"TRAIN: {train_count:,}")
    print(f"TEST: {len(all_images) - train_count:,}")
    try:
        _validate_move_plan(moves)
        _execute_move_plan(moves)
    except (OSError, ValueError) as error:
        return _result_warning(
            label,
            source_name,
            f"could not safely create the split: {error}",
        )
    return PreparationResult(
        label,
        source_name,
        "prepared",
        train_count,
        len(all_images) - train_count,
    )


def _print_summary(results: list[PreparationResult]) -> None:
    print("\n--- Dataset Preparation Summary ---")
    for label in ("FAKE", "REAL"):
        print(f"\n{label}:")
        label_results = [result for result in results if result.label == label]
        if not label_results:
            print("  No sources found")
            continue
        for result in label_results:
            print(f"  {result.source} [{result.status}]")
            print(f"    TRAIN: {result.train_count:,}")
            print(f"    TEST: {result.test_count:,}")
    print(f"\nTotal TRAIN: {sum(result.train_count for result in results):,}")
    print(f"Total TEST: {sum(result.test_count for result in results):,}")


def prepare_wildfake_data(
    wildfake_root: str | Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_PREPARATION_SEED,
) -> list[PreparationResult]:
    """Prepare every WildFake source or explain why it was safely skipped."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be greater than 0 and less than 1.")

    root = Path(wildfake_root)
    if not root.is_dir():
        print(f"WildFake dataset directory not found:\n{root}")
        print("\nDownload/extract your datasets first and place sources under:")
        print(f"{root / 'FAKE' / '<source_name>'}")
        print(f"{root / 'REAL' / '<source_name>'}")
        return []

    labelled_sources: list[tuple[str, Path]] = []
    seen_names: dict[str, Path] = {}
    for label in ("FAKE", "REAL"):
        label_root = root / label
        if not label_root.is_dir():
            print(f"[WARN] Missing label directory: {label_root}")
            continue
        for source_root in sorted(
            (path for path in label_root.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        ):
            key = source_root.name.casefold()
            previous = seen_names.get(key)
            if previous is not None:
                raise ValueError(
                    "WildFake source names must be unique ignoring case; found "
                    f"both {previous} and {source_root}."
                )
            seen_names[key] = source_root
            labelled_sources.append((label, source_root))

    results = [
        _prepare_source(source_root, label, train_ratio, seed)
        for label, source_root in labelled_sources
    ]
    _print_summary(results)
    return results
