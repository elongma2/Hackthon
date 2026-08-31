"""Evaluate prepared robustness variants described by ``distortions.csv``."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .evaluate import evaluate
from .distort_dataset2 import (
    _center_crop,
    _gaussian_noise,
    _resize_down_then_up,
    _stable_seed,
)
from .model import expects_unnormalized_input, load_model
from .transforms import build_eval_transforms


FAKE_LABEL = 0
REAL_LABEL = 1
SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class ConditionSpec:
    """Describe one canonical robustness condition and its display metadata."""
    condition_id: str
    transform: str
    parameter: str
    expected_tag: str


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("clean", "Clean", "none", "clean"),
    ConditionSpec("jpeg_90", "JPEG", "quality 90", "jpeg_q90"),
    ConditionSpec("jpeg_70", "JPEG", "quality 70", "jpeg_q70"),
    ConditionSpec("jpeg_50", "JPEG", "quality 50", "jpeg_q50"),
    ConditionSpec("jpeg_30", "JPEG", "quality 30", "jpeg_q30"),
    ConditionSpec("blur_05", "Gaussian blur", "sigma 0.5", "blur_sigma_0p5"),
    ConditionSpec("blur_1", "Gaussian blur", "sigma 1.0", "blur_sigma_1p0"),
    ConditionSpec("blur_2", "Gaussian blur", "sigma 2.0", "blur_sigma_2p0"),
    ConditionSpec("resize_05", "Resize", "0.5x down/up", "resize_0p5x"),
    ConditionSpec("resize_025", "Resize", "0.25x down/up", "resize_0p25x"),
    ConditionSpec("noise_002", "Gaussian noise", "sigma 0.02", "noise_sigma_0p02"),
    ConditionSpec("noise_005", "Gaussian noise", "sigma 0.05", "noise_sigma_0p05"),
    ConditionSpec("noise_010", "Gaussian noise", "sigma 0.10", "noise_sigma_0p1"),
    ConditionSpec("brightness_m20", "Brightness", "-20%", "brightness_0p8"),
    ConditionSpec("brightness_p20", "Brightness", "+20%", "brightness_1p2"),
    ConditionSpec("contrast_m20", "Contrast", "-20%", "contrast_0p8"),
    ConditionSpec("contrast_p20", "Contrast", "+20%", "contrast_1p2"),
    ConditionSpec("saturation_m20", "Saturation", "-20%", "saturation_0p8"),
    ConditionSpec("saturation_p20", "Saturation", "+20%", "saturation_1p2"),
    ConditionSpec("crop_80", "Center crop", "retain 80%", "center_crop_0p8"),
)
CONDITION_BY_ID = {condition.condition_id: condition for condition in CONDITIONS}
ROBUSTNESS_CONDITION_IDS = tuple(condition.condition_id for condition in CONDITIONS)
DISTORTION_CONDITION_IDS = ROBUSTNESS_CONDITION_IDS[1:]


@dataclass(frozen=True)
class ImageRecord:
    """Identify one labelled clean or distorted image by stable source ID."""
    path: Path
    source_id: str
    source_group_id: str | None
    label: int


@dataclass(frozen=True)
class ManifestRow:
    """Store one parsed and resolved distortions.csv record."""
    row_number: int
    condition_id: str
    source_id: str
    source_group_id: str | None
    label: int
    output_path: Path | None
    recorded_output: str
    split: str | None
    manifest_width: int | None
    manifest_height: int | None


@dataclass
class AuditResult:
    """Contain audited condition records, counts, schema, and nonfatal warnings."""
    schema: str
    manifest_path: Path | None
    clean_records: list[ImageRecord]
    condition_records: dict[str, list[ImageRecord]]
    present_conditions: list[str]
    missing_conditions: list[str]
    unexpected_conditions: list[str]
    counts: dict[str, dict[str, int]]
    warnings: list[str]

    def serializable(self) -> dict[str, object]:
        """Return JSON-safe audit metadata without heavyweight image records."""
        return {
            "schema": self.schema,
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "present_conditions": self.present_conditions,
            "missing_conditions": self.missing_conditions,
            "unexpected_conditions": self.unexpected_conditions,
            "counts": self.counts,
            "warnings": self.warnings,
        }


def sanitize_run_name(run_name: str) -> str:
    """Return a filesystem-safe run name consistent with checkpoint naming."""
    slug = re.sub(r"[^a-z0-9]+", "_", run_name.strip().casefold()).strip("_")
    if not slug:
        raise ValueError("--run-name must contain at least one letter or number")
    return slug


def _number(parameters: Mapping[str, object], key: str) -> float:
    """Read one finite numeric transform parameter from manifest JSON."""
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"manifest parameter {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"manifest parameter {key!r} must be finite")
    return number


def _match_number(value: float, expected: Sequence[tuple[float, str]]) -> str:
    """Map a numeric setting to exactly one supported canonical condition."""
    matches = [condition for number, condition in expected if math.isclose(value, number, abs_tol=1e-9)]
    if len(matches) != 1:
        raise ValueError(f"unsupported or ambiguous robustness parameter value {value!r}")
    return matches[0]


def map_manifest_condition(transform: str, parameters: Mapping[str, object]) -> str:
    """Map one manifest semantic description to a canonical condition ID."""
    normalized = transform.strip().casefold()
    if normalized == "none":
        return "clean"
    if normalized == "jpeg_compression":
        return _match_number(
            _number(parameters, "quality"),
            ((90, "jpeg_90"), (70, "jpeg_70"), (50, "jpeg_50"), (30, "jpeg_30")),
        )
    if normalized == "gaussian_blur":
        return _match_number(
            _number(parameters, "sigma"),
            ((0.5, "blur_05"), (1.0, "blur_1"), (2.0, "blur_2")),
        )
    if normalized == "resize":
        return _match_number(
            _number(parameters, "downscale"),
            ((0.5, "resize_05"), (0.25, "resize_025")),
        )
    if normalized == "gaussian_noise":
        return _match_number(
            _number(parameters, "sigma"),
            ((0.02, "noise_002"), (0.05, "noise_005"), (0.10, "noise_010")),
        )
    if normalized == "center_crop":
        return _match_number(_number(parameters, "retained_fraction"), ((0.8, "crop_80"),))
    if normalized == "color_jitter":
        keys = [key for key in ("brightness", "contrast", "saturation") if key in parameters]
        if len(keys) != 1:
            raise ValueError("color_jitter must contain exactly one supported property")
        key = keys[0]
        suffix = _match_number(_number(parameters, key), ((0.8, "m20"), (1.2, "p20")))
        return f"{key}_{suffix}"
    raise ValueError(f"unsupported robustness transform {transform!r}")


def _normalized_parts(path_text: str) -> tuple[str, ...]:
    """Normalize copied Windows or POSIX manifest paths into portable parts."""
    normalized = path_text.strip().replace("\\", "/")
    return tuple(part for part in PurePosixPath(normalized).parts if part not in ("/", ""))


def _clean_records(validation_dir: Path) -> tuple[list[ImageRecord], dict[str, ImageRecord]]:
    """Discover clean FAKE/REAL images and build collision-safe source IDs."""
    validation_dir = validation_dir.resolve()
    if not validation_dir.is_dir():
        raise FileNotFoundError(f"Clean validation directory not found: {validation_dir}")
    direct_classes = {path.name.casefold(): path.name for path in validation_dir.iterdir() if path.is_dir()}
    if direct_classes != {"fake": "FAKE", "real": "REAL"}:
        raise ValueError(
            "Clean validation directory must contain exactly FAKE and REAL class folders."
        )

    records: list[ImageRecord] = []
    lookup: dict[str, ImageRecord] = {}
    for class_name, label in (("FAKE", FAKE_LABEL), ("REAL", REAL_LABEL)):
        class_root = validation_dir / class_name
        paths = sorted(
            (
                path
                for path in class_root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.relative_to(validation_dir).as_posix().casefold(),
        )
        for path in paths:
            relative = path.relative_to(validation_dir).as_posix()
            source_id = f"{class_name}/{relative.split('/', 1)[1]}"
            key = source_id.casefold()
            if key in lookup:
                raise ValueError(f"Clean source IDs collide ignoring case: {source_id!r}")
            record = ImageRecord(path.resolve(), source_id, None, label)
            lookup[key] = record
            records.append(record)
    if not records or not any(record.label == FAKE_LABEL for record in records) or not any(
        record.label == REAL_LABEL for record in records
    ):
        raise ValueError("Clean validation requires at least one FAKE and one REAL image.")
    return records, lookup


def _stable_source_record(source: str, clean_lookup: Mapping[str, ImageRecord]) -> ImageRecord:
    """Resolve a copied absolute manifest source by its relative class-path suffix."""
    parts = _normalized_parts(source)
    matches: dict[str, ImageRecord] = {}
    for index, part in enumerate(parts):
        if part.casefold() not in {"fake", "real"}:
            continue
        candidate = "/".join(parts[index:]).casefold()
        record = clean_lookup.get(candidate)
        if record is not None:
            matches[record.source_id.casefold()] = record
    if len(matches) != 1:
        raise ValueError(
            f"manifest source {source!r} must match exactly one clean relative class path"
        )
    return next(iter(matches.values()))


def _resolve_output_path(
    recorded_output: str,
    distorted_dir: Path,
    split: str | None,
    source_id: str,
) -> Path | None:
    """Resolve a prepared file without trusting machine-specific absolute prefixes."""
    raw_path = Path(recorded_output)
    candidates: list[Path] = []
    if raw_path.is_file():
        candidates.append(raw_path.resolve())
    if not raw_path.is_absolute():
        candidates.append((distorted_dir / raw_path).resolve())
    filename = PurePosixPath(recorded_output.replace("\\", "/")).name
    parent = PurePosixPath(source_id).parent
    candidates.append((distorted_dir / Path(parent.as_posix()) / filename).resolve())
    if split:
        candidates.append(
            (distorted_dir / split / Path(parent.as_posix()) / filename).resolve()
        )
    existing = {path for path in candidates if path.is_file()}
    if len(existing) > 1:
        raise ValueError(
            f"manifest output {recorded_output!r} resolves ambiguously under {distorted_dir}"
        )
    return next(iter(existing)) if existing else None


def _optional_int(row: Mapping[str, str], key: str, row_number: int) -> int | None:
    """Parse an optional positive integer manifest field with row diagnostics."""
    text = (row.get(key) or "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"manifest row {row_number} has invalid {key}: {text!r}") from exc
    if value < 1:
        raise ValueError(f"manifest row {row_number} has nonpositive {key}: {value}")
    return value


def _iter_manifest_rows(manifest_path: Path) -> Iterable[dict[str, str]]:
    """Stream a potentially large manifest without retaining every CSV row."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
        yield from csv.DictReader(manifest_file)


def _readable_image(path: Path) -> tuple[tuple[int, int], str | None]:
    """Fully decode an image and return EXIF-corrected dimensions and format."""
    try:
        with Image.open(path) as image:
            image_format = image.format
            transposed = ImageOps.exif_transpose(image)
            size = transposed.size
            transposed.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image {path}: {exc}") from exc
    return size, image_format


def _jpeg_in_memory(image: Image.Image, quality: int) -> Image.Image:
    """Apply the same one-pass JPEG settings used by the prepared generator."""
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling="4:2:0",
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        result = compressed.convert("RGB")
        result.load()
    return result


def apply_live_condition(
    image: Image.Image,
    condition_id: str,
    seed: int,
    source_id: str,
) -> Image.Image:
    """Apply one official condition in memory without writing an image file."""
    if condition_id not in CONDITION_BY_ID:
        raise ValueError(f"Unknown robustness condition: {condition_id!r}")
    rgb = image.convert("RGB")
    if condition_id == "clean":
        return rgb
    if condition_id.startswith("jpeg_"):
        return _jpeg_in_memory(rgb, int(condition_id.split("_", 1)[1]))
    blur_sigmas = {"blur_05": 0.5, "blur_1": 1.0, "blur_2": 2.0}
    if condition_id in blur_sigmas:
        return rgb.filter(ImageFilter.GaussianBlur(radius=blur_sigmas[condition_id]))
    resize_scales = {"resize_05": 0.5, "resize_025": 0.25}
    if condition_id in resize_scales:
        return _resize_down_then_up(rgb, resize_scales[condition_id])
    noise_sigmas = {"noise_002": 0.02, "noise_005": 0.05, "noise_010": 0.10}
    if condition_id in noise_sigmas:
        tag = CONDITION_BY_ID[condition_id].expected_tag
        variant_seed = _stable_seed(seed, source_id, tag)
        return _gaussian_noise(rgb, noise_sigmas[condition_id], variant_seed)
    color_conditions = {
        "brightness_m20": (ImageEnhance.Brightness, 0.8),
        "brightness_p20": (ImageEnhance.Brightness, 1.2),
        "contrast_m20": (ImageEnhance.Contrast, 0.8),
        "contrast_p20": (ImageEnhance.Contrast, 1.2),
        "saturation_m20": (ImageEnhance.Color, 0.8),
        "saturation_p20": (ImageEnhance.Color, 1.2),
    }
    if condition_id in color_conditions:
        enhancer, factor = color_conditions[condition_id]
        return enhancer(rgb).enhance(factor)
    if condition_id == "crop_80":
        return _center_crop(rgb, 0.8)
    raise AssertionError(f"Missing live implementation for {condition_id}")


def _check_condition_image(
    condition_id: str,
    manifest_row: ManifestRow,
    clean_size: tuple[int, int],
    warnings: list[str],
) -> None:
    """Validate readability and generator-consistent dimensions/format with warnings."""
    if manifest_row.output_path is None:
        raise FileNotFoundError(
            f"Prepared output is missing for {condition_id}/{manifest_row.source_id}: "
            f"{manifest_row.recorded_output}"
        )
    output_size, output_format = _readable_image(manifest_row.output_path)
    expected_size = clean_size
    if condition_id == "crop_80":
        expected_size = (max(1, round(clean_size[0] * 0.8)), max(1, round(clean_size[1] * 0.8)))
    if output_size != expected_size:
        warnings.append(
            f"Dimension warning for {condition_id}/{manifest_row.source_id}: "
            f"found {output_size}, expected {expected_size} from the current generator."
        )
    if manifest_row.manifest_width is not None and manifest_row.manifest_height is not None:
        manifest_size = (manifest_row.manifest_width, manifest_row.manifest_height)
        if output_size != manifest_size:
            warnings.append(
                f"Manifest dimension warning for {condition_id}/{manifest_row.source_id}: "
                f"file {output_size}, manifest {manifest_size}."
            )
    if condition_id.startswith("jpeg_") and output_format != "JPEG":
        warnings.append(
            f"Format warning for {condition_id}/{manifest_row.source_id}: "
            f"expected JPEG, found {output_format or 'unknown'}."
        )


def audit_prepared_robustness_data(
    validation_dir: str | Path,
    distorted_dir: str | Path,
    only: str | None = None,
) -> AuditResult:
    """Validate prepared files and return deterministic condition datasets."""
    if only is not None and only not in CONDITION_BY_ID:
        raise ValueError(f"Unknown robustness condition: {only!r}")
    validation_path = Path(validation_dir).resolve()
    distorted_path = Path(distorted_dir).resolve()
    manifest_path = distorted_path / "distortions.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared distortion manifest not found: {manifest_path}")
    clean_records, clean_lookup = _clean_records(validation_path)

    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = set(reader.fieldnames or ())
        common = {"source", "output", "transform", "parameters"}
        if not common <= fieldnames:
            missing = ", ".join(sorted(common - fieldnames))
            raise ValueError(f"Unsupported distortions.csv schema; missing columns: {missing}")
        labelled = {"split", "class_label"} <= fieldnames
        schema = "distort_dataset2" if labelled else "distort_folder"
    rows = _iter_manifest_rows(manifest_path)

    warnings: list[str] = []
    manifest_rows: dict[str, dict[str, ManifestRow]] = {
        condition_id: {} for condition_id in ROBUSTNESS_CONDITION_IDS
    }
    supplied_groups: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    row_count = 0
    for row_number, row in enumerate(rows, start=2):
        row_count += 1
        try:
            parameters = json.loads(row.get("parameters") or "")
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest row {row_number} has malformed parameters JSON") from exc
        if not isinstance(parameters, dict):
            raise ValueError(f"manifest row {row_number} parameters must be a JSON object")
        try:
            condition_id = map_manifest_condition(row.get("transform") or "", parameters)
        except ValueError as exc:
            raise ValueError(f"manifest row {row_number}: {exc}") from exc
        clean_record = _stable_source_record(row.get("source") or "", clean_lookup)
        class_label = (row.get("class_label") or "").strip().casefold()
        if labelled:
            expected_class = "fake" if clean_record.label == FAKE_LABEL else "real"
            if class_label != expected_class:
                raise ValueError(
                    f"manifest row {row_number} class label {class_label!r} does not match "
                    f"clean source {clean_record.source_id!r}"
                )
        source_key = clean_record.source_id.casefold()
        if source_key in manifest_rows[condition_id]:
            raise ValueError(
                f"duplicate manifest rows for {condition_id}/{clean_record.source_id}"
            )
        group_id = (row.get("source_group_id") or "").strip() or None
        if group_id is not None:
            previous_source = supplied_groups.get(group_id)
            if previous_source is not None and previous_source != source_key:
                raise ValueError(f"source_group_id {group_id!r} maps to multiple clean sources")
            supplied_groups[group_id] = source_key
            previous_group = source_groups.get(source_key)
            if previous_group is not None and previous_group != group_id:
                raise ValueError(
                    f"clean source {clean_record.source_id!r} has inconsistent "
                    "source_group_id values"
                )
            source_groups[source_key] = group_id
        split = (row.get("split") or "").strip() or None
        recorded_output = (row.get("output") or "").strip()
        if not recorded_output:
            raise ValueError(f"manifest row {row_number} has an empty output path")
        output_path = _resolve_output_path(
            recorded_output,
            distorted_path,
            split,
            clean_record.source_id,
        )
        manifest_row = ManifestRow(
            row_number=row_number,
            condition_id=condition_id,
            source_id=clean_record.source_id,
            source_group_id=group_id,
            label=clean_record.label,
            output_path=output_path,
            recorded_output=recorded_output,
            split=split,
            manifest_width=_optional_int(row, "width", row_number),
            manifest_height=_optional_int(row, "height", row_number),
        )
        manifest_rows[condition_id][source_key] = manifest_row
        filename = PurePosixPath(recorded_output.replace("\\", "/")).name.casefold()
        expected_tag = CONDITION_BY_ID[condition_id].expected_tag.casefold()
        if expected_tag not in filename:
            warnings.append(
                f"Naming warning on manifest row {row_number}: {recorded_output!r} "
                f"does not contain expected tag {expected_tag!r}; manifest semantics were used."
            )
        folder_conditions = {
            re.sub(r"[^a-z0-9]+", "_", part.casefold()).strip("_")
            for part in PurePosixPath(recorded_output.replace("\\", "/")).parent.parts
        } & set(CONDITION_BY_ID)
        disagreeing_folders = sorted(folder_conditions - {condition_id})
        if disagreeing_folders:
            warnings.append(
                f"Folder naming warning on manifest row {row_number}: "
                f"folder condition(s) {disagreeing_folders} disagree with manifest "
                f"condition {condition_id!r}; manifest semantics were used."
            )
    if row_count == 0:
        raise ValueError("distortions.csv contains no records")

    required_for_files = (
        set(DISTORTION_CONDITION_IDS)
        if only is None
        else ({only} if only not in (None, "clean") else set())
    )
    for condition_id, condition_map in manifest_rows.items():
        if condition_id == "clean":
            continue
        missing_outputs = [
            row.source_id for row in condition_map.values() if row.output_path is None
        ]
        if missing_outputs and condition_id not in required_for_files:
            warnings.append(
                f"Prepared files missing for unselected {condition_id}: "
                f"{len(missing_outputs):,} file(s)."
            )

    present = [
        condition.condition_id
        for condition in CONDITIONS[1:]
        if manifest_rows[condition.condition_id]
    ]
    missing = [condition for condition in DISTORTION_CONDITION_IDS if condition not in present]
    required = [] if only == "clean" else ([only] if only is not None else list(DISTORTION_CONDITION_IDS))
    required = [condition for condition in required if condition is not None]
    missing_required = [condition for condition in required if condition in missing]
    if missing_required:
        raise ValueError("Required prepared conditions are missing: " + ", ".join(missing_required))

    clean_keys = set(clean_lookup)
    clean_sizes = {
        source_key: _readable_image(record.path)[0]
        for source_key, record in clean_lookup.items()
    }
    selected_conditions = required
    condition_records: dict[str, list[ImageRecord]] = {"clean": clean_records}
    for condition_id in selected_conditions:
        condition_map = manifest_rows[condition_id]
        condition_keys = set(condition_map)
        if condition_keys != clean_keys:
            missing_sources = sorted(clean_keys - condition_keys)
            unexpected_sources = sorted(condition_keys - clean_keys)
            raise ValueError(
                f"Source-set mismatch for {condition_id}; missing={missing_sources[:5]}, "
                f"unexpected={unexpected_sources[:5]}"
            )
        records: list[ImageRecord] = []
        for source_key in sorted(clean_keys):
            manifest_row = condition_map[source_key]
            clean_record = clean_lookup[source_key]
            _check_condition_image(
                condition_id,
                manifest_row,
                clean_sizes[source_key],
                warnings,
            )
            assert manifest_row.output_path is not None
            if manifest_row.output_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported prepared image extension: {manifest_row.output_path}")
            records.append(
                ImageRecord(
                    manifest_row.output_path,
                    manifest_row.source_id,
                    manifest_row.source_group_id,
                    manifest_row.label,
                )
            )
        condition_records[condition_id] = records

    counts: dict[str, dict[str, int]] = {}
    for condition_id in ("clean", *present):
        if condition_id == "clean":
            labels = [record.label for record in clean_records]
        else:
            labels = [row.label for row in manifest_rows[condition_id].values()]
        counter = Counter(labels)
        counts[condition_id] = {
            "total": len(labels),
            "FAKE": counter[FAKE_LABEL],
            "REAL": counter[REAL_LABEL],
        }

    return AuditResult(
        schema=schema,
        manifest_path=manifest_path,
        clean_records=clean_records,
        condition_records=condition_records,
        present_conditions=present,
        missing_conditions=missing,
        unexpected_conditions=[],
        counts=counts,
        warnings=warnings,
    )


def audit_live_robustness_data(validation_dir: str | Path) -> AuditResult:
    """Verify the clean source set used for deterministic in-memory conditions."""
    clean_records, _ = _clean_records(Path(validation_dir))
    for record in clean_records:
        _readable_image(record.path)
    counter = Counter(record.label for record in clean_records)
    clean_counts = {
        "total": len(clean_records),
        "FAKE": counter[FAKE_LABEL],
        "REAL": counter[REAL_LABEL],
    }
    return AuditResult(
        schema="on_the_fly",
        manifest_path=None,
        clean_records=clean_records,
        condition_records={
            condition_id: clean_records for condition_id in ROBUSTNESS_CONDITION_IDS
        },
        present_conditions=list(DISTORTION_CONDITION_IDS),
        missing_conditions=[],
        unexpected_conditions=[],
        counts={
            condition_id: dict(clean_counts)
            for condition_id in ROBUSTNESS_CONDITION_IDS
        },
        warnings=[],
    )


class PreparedConditionDataset(Dataset):
    """Read one audited condition without changing its order or membership."""

    def __init__(self, records: Sequence[ImageRecord], transform) -> None:
        """Store audited records and the model-aware deterministic transform."""
        self.records = tuple(records)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of prepared images evaluated exactly once."""
        return len(self.records)

    def __getitem__(self, index: int):
        """Load one prepared RGB image with its label and stable source ID."""
        record = self.records[index]
        with Image.open(record.path) as image:
            value = self.transform(image.convert("RGB"))
        return value, record.label, record.source_id


class LiveConditionDataset(Dataset):
    """Apply one deterministic condition to clean images without saving variants."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform,
        condition_id: str,
        seed: int,
    ) -> None:
        """Configure one deterministic in-memory condition over clean records."""
        self.records = tuple(records)
        self.transform = transform
        self.condition_id = condition_id
        self.seed = seed

    def __len__(self) -> int:
        """Return the unchanged number of clean source images."""
        return len(self.records)

    def __getitem__(self, index: int):
        """Apply one condition transiently and return the transformed labelled image."""
        record = self.records[index]
        with Image.open(record.path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
            original.load()
        conditioned = apply_live_condition(
            original,
            self.condition_id,
            self.seed,
            record.source_id,
        )
        return self.transform(conditioned), record.label, record.source_id


def calculate_robustness_metrics(
    labels: Sequence[int],
    p_real: Sequence[float],
    probability_threshold: float,
) -> dict[str, object]:
    """Calculate binary metrics with FAKE/AIGC explicitly treated as positive."""
    if len(labels) != len(p_real) or not labels:
        raise ValueError("Metric labels and probabilities must be nonempty and aligned.")
    if set(labels) != {FAKE_LABEL, REAL_LABEL}:
        raise ValueError("Robustness metrics require both FAKE=0 and REAL=1 images.")
    predicted = [REAL_LABEL if probability >= probability_threshold else FAKE_LABEL for probability in p_real]
    p_aigc = [1.0 - probability for probability in p_real]
    aigc_targets = [1 if label == FAKE_LABEL else 0 for label in labels]
    fake_fake = sum(a == FAKE_LABEL and p == FAKE_LABEL for a, p in zip(labels, predicted))
    fake_real = sum(a == FAKE_LABEL and p == REAL_LABEL for a, p in zip(labels, predicted))
    real_real = sum(a == REAL_LABEL and p == REAL_LABEL for a, p in zip(labels, predicted))
    real_fake = sum(a == REAL_LABEL and p == FAKE_LABEL for a, p in zip(labels, predicted))
    total = len(labels)
    fake_recall = fake_fake / (fake_fake + fake_real)
    real_recall = real_real / (real_real + real_fake)
    accuracy = (fake_fake + real_real) / total
    return {
        "total": total,
        "correct": fake_fake + real_real,
        "incorrect": fake_real + real_fake,
        "accuracy": accuracy,
        "auc_aigc": float(roc_auc_score(aigc_targets, p_aigc)),
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "balanced_accuracy": (fake_recall + real_recall) / 2.0,
        "confusion_matrix": {
            "fake_to_fake": fake_fake,
            "fake_to_real": fake_real,
            "real_to_real": real_real,
            "real_to_fake": real_fake,
        },
    }


def _with_deltas(metrics: dict[str, object], clean: Mapping[str, object]) -> dict[str, object]:
    """Add raw metric differences against the clean baseline."""
    result = dict(metrics)
    delta_names = {
        "accuracy": "delta_accuracy",
        "auc_aigc": "delta_auc",
        "balanced_accuracy": "delta_balanced_accuracy",
        "fake_recall": "delta_fake_recall",
        "real_recall": "delta_real_recall",
    }
    for metric, delta_name in delta_names.items():
        result[delta_name] = float(metrics[metric]) - float(clean[metric])
    return result


def _format_percent(value: object) -> str:
    """Format a fractional metric as a percentage."""
    return f"{float(value) * 100:.2f}%"


def _format_delta(value: object) -> str:
    """Format a raw fractional delta as signed percentage points."""
    return f"{float(value) * 100:+.2f} pp"


def _table_rows(results: Mapping[str, Mapping[str, object]]) -> list[list[str]]:
    """Create display rows in the official robustness-condition order."""
    rows: list[list[str]] = []
    for condition in CONDITIONS:
        metrics = results.get(condition.condition_id)
        if metrics is None:
            continue
        rows.append(
            [
                condition.transform,
                condition.parameter,
                _format_percent(metrics["accuracy"]),
                f"{float(metrics['auc_aigc']):.4f}",
                _format_percent(metrics["fake_recall"]),
                _format_percent(metrics["real_recall"]),
                _format_percent(metrics["balanced_accuracy"]),
                _format_delta(metrics["delta_accuracy"]),
                _format_delta(metrics["delta_auc"]),
                _format_delta(metrics["delta_balanced_accuracy"]),
            ]
        )
    return rows


TABLE_HEADERS = [
    "Transform",
    "Parameter",
    "Accuracy",
    "AUC",
    "FAKE Recall",
    "REAL Recall",
    "Balanced Acc",
    "Delta Accuracy",
    "Delta AUC",
    "Delta Balanced Acc",
]


def _markdown_table(rows: Sequence[Sequence[str]]) -> str:
    """Render robustness result rows as a compact Markdown table."""
    lines = [
        "| " + " | ".join(TABLE_HEADERS) + " |",
        "| " + " | ".join("---" for _ in TABLE_HEADERS) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _write_outputs(
    output_dir: Path,
    summary: Mapping[str, object],
    results: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Atomically create the run directory's JSON, CSV, Markdown, and predictions."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = _table_rows(results)
    (output_dir / "robustness_summary.md").write_text(
        _markdown_table(rows), encoding="utf-8"
    )
    csv_path = output_dir / "robustness_summary.csv"
    csv_fields = [
        "condition_id", "transform", "parameter", "total", "correct", "incorrect",
        "accuracy", "auc_aigc", "fake_recall", "real_recall", "balanced_accuracy",
        "delta_accuracy", "delta_auc", "delta_balanced_accuracy", "delta_fake_recall",
        "delta_real_recall", "fake_to_fake", "fake_to_real", "real_to_real", "real_to_fake",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for condition in CONDITIONS:
            metrics = results.get(condition.condition_id)
            if metrics is None:
                continue
            confusion = metrics["confusion_matrix"]
            writer.writerow(
                {
                    "condition_id": condition.condition_id,
                    "transform": condition.transform,
                    "parameter": condition.parameter,
                    **{key: metrics[key] for key in csv_fields[3:16]},
                    **confusion,
                }
            )
    (output_dir / "robustness_predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )


def print_audit(audit: AuditResult) -> None:
    """Print the prepared-data findings before checkpoint loading."""
    print("\n--- Robustness Matrix Preflight Audit ---\n")
    print(
        "Manifest: not used (conditions are generated in memory)"
        if audit.manifest_path is None
        else f"Manifest: {audit.manifest_path}"
    )
    print(f"Schema: {audit.schema}")
    print("Present conditions: " + (", ".join(audit.present_conditions) or "none"))
    print("Missing conditions: " + (", ".join(audit.missing_conditions) or "none"))
    for condition_id in ("clean", *audit.present_conditions):
        counts = audit.counts[condition_id]
        print(
            f"  {condition_id:<16} total={counts['total']:,} "
            f"FAKE={counts['FAKE']:,} REAL={counts['REAL']:,}"
        )
    for warning in audit.warnings:
        print(f"WARNING: {warning}")


def run_robustness_matrix(
    checkpoint_path: str | Path,
    validation_dir: str | Path,
    distorted_dir: str | Path | None,
    device: torch.device,
    probability_threshold: float = 0.5,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    run_name: str = "robustness",
    only: str | None = None,
    results_root: str | Path = "results/robustness",
    seed: int = 42,
) -> dict[str, object]:
    """Audit inputs, load one checkpoint, and evaluate the selected matrix."""
    if not math.isfinite(probability_threshold) or not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("--probability-threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if only is not None and only not in CONDITION_BY_ID:
        raise ValueError(f"Unknown robustness condition: {only!r}")
    run_slug = sanitize_run_name(run_name)
    output_dir = Path(results_root) / run_slug
    if output_dir.exists():
        raise FileExistsError(
            f"Robustness result directory already exists: {output_dir}. Choose a new --run-name."
        )

    live_mode = distorted_dir is None
    if live_mode:
        audit = audit_live_robustness_data(validation_dir)
    else:
        audit = audit_prepared_robustness_data(validation_dir, distorted_dir, only=only)
    print_audit(audit)
    model = load_model(checkpoint_path, device)
    transform = build_eval_transforms(
        image_size,
        normalize=not expects_unnormalized_input(model),
    )
    condition_ids = ["clean"] if only == "clean" else ["clean", *(
        [only] if only is not None else DISTORTION_CONDITION_IDS
    )]
    criterion = nn.BCEWithLogitsLoss()
    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, list[dict[str, object]]] = {}
    for condition_id in condition_ids:
        records = audit.condition_records[condition_id]
        dataset: Dataset
        if live_mode:
            dataset = LiveConditionDataset(
                records,
                transform,
                condition_id=condition_id,
                seed=seed,
            )
        else:
            dataset = PreparedConditionDataset(records, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        inference = evaluate(
            model,
            loader,
            criterion,
            device,
            probability_threshold=probability_threshold,
            description=f"Robustness {condition_id}",
        )
        source_names = list(inference.get("source_names", ()))
        expected_sources = [record.source_id for record in records]
        if source_names != expected_sources:
            raise RuntimeError(f"Per-image prediction order changed for {condition_id}")
        labels = list(inference["labels"])
        probabilities = list(inference["probabilities"])
        metrics = calculate_robustness_metrics(labels, probabilities, probability_threshold)
        results[condition_id] = metrics
        spec = CONDITION_BY_ID[condition_id]
        condition_predictions: list[dict[str, object]] = []
        for record, label, p_real in zip(records, labels, probabilities):
            predicted_label = REAL_LABEL if p_real >= probability_threshold else FAKE_LABEL
            condition_predictions.append(
                {
                    "image_path": str(record.path),
                    "source_id": record.source_id,
                    "source_group_id": record.source_group_id or record.source_id,
                    "true_label": "FAKE" if label == FAKE_LABEL else "REAL",
                    "transform": spec.transform,
                    "parameter": spec.parameter,
                    "p_real": float(p_real),
                    "p_aigc": float(1.0 - p_real),
                    "predicted_label": "FAKE" if predicted_label == FAKE_LABEL else "REAL",
                    "correct": predicted_label == label,
                }
            )
        predictions[condition_id] = condition_predictions

    clean_metrics = results["clean"]
    results = {
        condition_id: _with_deltas(metrics, clean_metrics)
        for condition_id, metrics in results.items()
    }
    summary: dict[str, object] = {
        "run_name": run_name,
        "run_slug": run_slug,
        "checkpoint": str(Path(checkpoint_path)),
        "validation_dir": str(Path(validation_dir)),
        "distorted_dir": None if distorted_dir is None else str(Path(distorted_dir)),
        "evaluation_mode": "on_the_fly" if live_mode else "prepared_manifest",
        "distortion_seed": seed,
        "probability_threshold": probability_threshold,
        "label_semantics": {
            "FAKE": FAKE_LABEL,
            "REAL": REAL_LABEL,
            "p_real": "sigmoid(logit)",
            "p_aigc": "1 - sigmoid(logit)",
        },
        "only": only,
        "audit": audit.serializable(),
        "conditions": results,
    }
    _write_outputs(output_dir, summary, results, predictions)
    table = _markdown_table(_table_rows(results))
    print("\n--- Robustness Matrix Results ---\n")
    print(table)
    print(f"Saved robustness matrix results to {output_dir}")
    return summary
