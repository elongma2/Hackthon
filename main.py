"""Command-line entry point for training and evaluating the CIFAKE detector."""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

from src.bytedance_validation import run_bytedance_validation
from src.data_preparation import DEFAULT_TRAIN_RATIO, prepare_wildfake_data
from src.dataset import download_dataset, get_data_loaders, get_test_directory
from src.evaluate import evaluate
from src.hybrid_model import (
    FREQUENCY_FEATURE_DIM,
    FUSION_DROPOUT,
    FUSION_HIDDEN_DIM,
    HYBRID_MODEL_TYPE,
    SPATIAL_FEATURE_DIM,
    build_hybrid_model,
)
from src.hybrid_v2_model import (
    DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    DEFAULT_FREQUENCY_MASK_PROB,
    DEFAULT_FREQUENCY_SCALE,
    HYBRID_V2_MODEL_TYPE,
    V2_FREQUENCY_DROPOUT,
    V2_FREQUENCY_FEATURE_DIM,
    V2_FREQUENCY_HIDDEN_DIM,
    V2_FREQUENCY_LEARNING_RATE,
    V2_SPATIAL_FEATURE_DIM,
    V2_SPATIAL_LEARNING_RATE,
    build_hybrid_v2_model,
    configure_hybrid_v2_stage,
)
from src.hybrid_v3_model import (
    DEFAULT_MAGNITUDE_WEIGHT,
    DEFAULT_PHASE_WEIGHT,
    HYBRID_V3_MODEL_TYPE,
    V3_FREQUENCY_DROPOUT,
    V3_FREQUENCY_HIDDEN_DIM,
    V3_FREQUENCY_LEARNING_RATE,
    V3_MAGNITUDE_FEATURE_DIM,
    V3_PHASE_FEATURE_DIM,
    V3_SPATIAL_FEATURE_DIM,
    V3_SPATIAL_LEARNING_RATE,
    build_hybrid_v3_model,
    configure_hybrid_v3_stage,
)
from src.hybrid_v31_model import (
    DEFAULT_RADIAL_BINS,
    HYBRID_V31_MODEL_TYPE,
    MIN_RADIAL_BINS,
    V31_FREQUENCY_DROPOUT,
    V31_FREQUENCY_HIDDEN_DIM,
    V31_FREQUENCY_LEARNING_RATE,
    V31_MAGNITUDE_FEATURE_DIM,
    V31_PHASE_FEATURE_DIM,
    V31_RADIAL_DROPOUT,
    V31_RADIAL_HIDDEN_DIM,
    V31_SPATIAL_FEATURE_DIM,
    V31_SPATIAL_LEARNING_RATE,
    build_hybrid_v31_model,
    configure_hybrid_v31_stage,
    v31_epoch_metadata,
    v31_validation_forward,
)
from src.model import build_model, expects_unnormalized_input, get_device, load_model
from src.multisource_dataset import (
    get_multisource_data_loaders,
    resolve_wildfake_holdout,
)
from src.predict import predict_image
from src.robustness import run_robustness_benchmark
from src.source_balanced import (
    DEFAULT_BALANCED_SEED,
    DEFAULT_SAMPLES_PER_EPOCH,
    get_source_balanced_data_loaders,
)
from src.train import train_model, train_staged_model


DEFAULT_CHECKPOINT = Path("checkpoints/best_model.pt")
DEFAULT_STAGED_CHECKPOINT = Path("checkpoints/efficientnet_staged_best.pt")
DEFAULT_MULTISOURCE_CHECKPOINT = Path(
    "checkpoints/efficientnet_staged_multisource_best.pt"
)
DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT = Path(
    "checkpoints/efficientnet_balanced_all_sources_best.pt"
)
DEFAULT_ALL_SOURCE_HYBRID_CHECKPOINT = Path(
    "checkpoints/hybrid_balanced_all_sources_best.pt"
)
DEFAULT_ALL_SOURCE_HYBRID_V2_CHECKPOINT = Path(
    "checkpoints/hybrid_v2_balanced_all_sources_best.pt"
)
DEFAULT_ALL_SOURCE_HYBRID_V3_CHECKPOINT = Path(
    "checkpoints/hybrid_v3_balanced_all_sources_best.pt"
)
DEFAULT_ALL_SOURCE_HYBRID_V31_CHECKPOINT = Path(
    "checkpoints/hybrid_v31_balanced_all_sources_best.pt"
)


def build_parser() -> argparse.ArgumentParser:
    """Describe every supported command-line command and option."""
    parser = argparse.ArgumentParser(description="Train and test an AI-image detector.")
    parser.add_argument(
        "command",
        choices=(
            "train",
            "train-staged",
            "train-multisource",
            "train-source-balanced",
            "train-hybrid",
            "train-hybrid-v2",
            "train-hybrid-v3",
            "train-hybrid-v31",
            "evaluate",
            "robustness",
            "predict",
            "validate-bytedance",
            "prepare-data",
        ),
        nargs="?",
        default="train",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Uses the multisource checkpoint for train-multisource, "
            "an all-source or holdout-specific checkpoint for source-balanced and "
            "hybrid training, the staged checkpoint for train-staged and "
            "validate-bytedance, and the baseline checkpoint otherwise."
        ),
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=None,
        help=(
            "Optional WildFake FAKE source excluded from train-source-balanced or "
            "a hybrid command. Omit it to train on every prepared source."
        ),
    )
    parser.add_argument(
        "--spatial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional EfficientNet checkpoint used to initialize train-hybrid. "
            "It also initializes train-hybrid-v2, train-hybrid-v3, and "
            "train-hybrid-v31. When omitted, the spatial branch uses ImageNet weights."
        ),
    )
    parser.add_argument(
        "--v2-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional exact Hybrid V2 warm start for train-hybrid-v3. Loads spatial "
            "and magnitude paths; mutually exclusive with --spatial-checkpoint."
        ),
    )
    parser.add_argument(
        "--frequency-scale",
        type=float,
        default=DEFAULT_FREQUENCY_SCALE,
        help="Fixed Hybrid V2/V3/V3.1 FFT residual scale (default: 0.25).",
    )
    parser.add_argument(
        "--frequency-branch-dropout",
        type=float,
        default=DEFAULT_FREQUENCY_BRANCH_DROPOUT,
        help="Hybrid V2/V3/V3.1 training-only FFT branch dropout (default: 0.20).",
    )
    parser.add_argument(
        "--frequency-mask-prob",
        type=float,
        default=DEFAULT_FREQUENCY_MASK_PROB,
        help="Hybrid V2/V3/V3.1 magnitude masking chance (default: 0.0).",
    )
    parser.add_argument(
        "--magnitude-weight",
        type=float,
        default=DEFAULT_MAGNITUDE_WEIGHT,
        help="Hybrid V3 supplied magnitude mixture weight (default: 0.5).",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=DEFAULT_PHASE_WEIGHT,
        help="Hybrid V3 supplied phase mixture weight (default: 0.5).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional Hybrid V2/V3/V3.1 run name used in its checkpoint path.",
    )
    parser.add_argument(
        "--radial-bins",
        type=int,
        default=DEFAULT_RADIAL_BINS,
        help="Hybrid V3.1 radial profile bin count (default: 32; minimum: 4).",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=DEFAULT_SAMPLES_PER_EPOCH,
        help="Fixed number of balanced training samples used in each epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BALANCED_SEED,
        help="Deterministic sampler or prepare-data split seed.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--wildfake-dir",
        type=Path,
        default=None,
        help="WildFake root. Defaults to <data-dir>/WildFake.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("validation"),
        help="ByteDance validation ImageFolder root.",
    )
    parser.add_argument("--image", type=Path, help="Image to classify with the predict command.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for baseline training.")
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=2,
        help="Frozen-spatial warm-up epochs for hybrid training (default: 2).",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=5,
        help=(
            "Partial-unfreezing epochs (default: 5). EfficientNet-only staged "
            "training still uses its fixed two head-only epochs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Per-source training fraction for train-multisource. "
            "Uses deterministic seed 42; internal validation remains complete."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="TRAIN share created by prepare-data (default: 0.9).",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def resolve_checkpoint_path(
    command: str,
    checkpoint: Path | None,
    holdout: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Choose a command-specific checkpoint unless the user supplied one."""
    if checkpoint is not None:
        return checkpoint
    if command == "train-source-balanced":
        if holdout is None:
            return DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT
        normalized = holdout.strip()
        slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
        if not slug:
            raise ValueError(
                f"Cannot create a safe checkpoint name from holdout {holdout!r}; "
                "pass --checkpoint explicitly."
            )
        return Path(f"checkpoints/efficientnet_balanced_holdout_{slug}_best.pt")
    if command == "train-hybrid":
        if holdout is None:
            return DEFAULT_ALL_SOURCE_HYBRID_CHECKPOINT
        normalized = holdout.strip()
        slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
        if not slug:
            raise ValueError(
                f"Cannot create a safe checkpoint name from holdout {holdout!r}; "
                "pass --checkpoint explicitly."
            )
        return Path(f"checkpoints/hybrid_balanced_holdout_{slug}_best.pt")
    if command == "train-hybrid-v2":
        holdout_slug: str | None = None
        if holdout is not None:
            holdout_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                holdout.strip().casefold(),
            ).strip("_")
            if not holdout_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from holdout {holdout!r}; "
                    "pass --checkpoint explicitly."
                )
        if run_name is not None:
            run_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                run_name.strip().casefold(),
            ).strip("_")
            if not run_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from run name {run_name!r}."
                )
            if holdout_slug is None:
                return Path(f"checkpoints/hybrid_v2_{run_slug}_all_sources_best.pt")
            return Path(
                f"checkpoints/hybrid_v2_{run_slug}_holdout_{holdout_slug}_best.pt"
            )
        if holdout_slug is None:
            return DEFAULT_ALL_SOURCE_HYBRID_V2_CHECKPOINT
        return Path(f"checkpoints/hybrid_v2_balanced_holdout_{holdout_slug}_best.pt")
    if command == "train-hybrid-v3":
        holdout_slug: str | None = None
        if holdout is not None:
            holdout_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                holdout.strip().casefold(),
            ).strip("_")
            if not holdout_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from holdout {holdout!r}; "
                    "pass --checkpoint explicitly."
                )
        if run_name is not None:
            run_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                run_name.strip().casefold(),
            ).strip("_")
            if not run_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from run name {run_name!r}."
                )
            if holdout_slug is None:
                return Path(f"checkpoints/hybrid_v3_{run_slug}_all_sources_best.pt")
            return Path(
                f"checkpoints/hybrid_v3_{run_slug}_holdout_{holdout_slug}_best.pt"
            )
        if holdout_slug is None:
            return DEFAULT_ALL_SOURCE_HYBRID_V3_CHECKPOINT
        return Path(f"checkpoints/hybrid_v3_balanced_holdout_{holdout_slug}_best.pt")
    if command == "train-hybrid-v31":
        holdout_slug: str | None = None
        if holdout is not None:
            holdout_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                holdout.strip().casefold(),
            ).strip("_")
            if not holdout_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from holdout {holdout!r}; "
                    "pass --checkpoint explicitly."
                )
        if run_name is not None:
            run_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                run_name.strip().casefold(),
            ).strip("_")
            if not run_slug:
                raise ValueError(
                    f"Cannot create a safe checkpoint name from run name {run_name!r}."
                )
            if holdout_slug is None:
                return Path(f"checkpoints/hybrid_v31_{run_slug}_all_sources_best.pt")
            return Path(
                f"checkpoints/hybrid_v31_{run_slug}_holdout_{holdout_slug}_best.pt"
            )
        if holdout_slug is None:
            return DEFAULT_ALL_SOURCE_HYBRID_V31_CHECKPOINT
        return Path(f"checkpoints/hybrid_v31_balanced_holdout_{holdout_slug}_best.pt")
    if command == "train-multisource":
        return DEFAULT_MULTISOURCE_CHECKPOINT
    if command in ("train-staged", "validate-bytedance"):
        return DEFAULT_STAGED_CHECKPOINT
    return DEFAULT_CHECKPOINT


def main() -> None:
    """Parse the command and run exactly one training or evaluation workflow."""
    parser = build_parser()
    args = parser.parse_args()
    if args.samples_per_epoch < 1:
        parser.error("--samples-per-epoch must be at least 1")
    if args.stage1_epochs < 1:
        parser.error("--stage1-epochs must be at least 1")
    if args.command in (
        "train-hybrid-v2",
        "train-hybrid-v3",
        "train-hybrid-v31",
    ):
        if not math.isfinite(args.frequency_scale) or args.frequency_scale < 0.0:
            parser.error("--frequency-scale must be greater than or equal to 0")
        if not 0.0 <= args.frequency_branch_dropout <= 1.0:
            parser.error("--frequency-branch-dropout must be between 0 and 1")
        if not 0.0 <= args.frequency_mask_prob <= 1.0:
            parser.error("--frequency-mask-prob must be between 0 and 1")
        if args.run_name is not None and not re.sub(
            r"[^a-z0-9]+",
            "_",
            args.run_name.strip().casefold(),
        ).strip("_"):
            parser.error("--run-name must contain at least one letter or number")
    if args.command == "train-hybrid-v3":
        if args.spatial_checkpoint is not None and args.v2_checkpoint is not None:
            parser.error(
                "--spatial-checkpoint and --v2-checkpoint are mutually exclusive"
            )
        if (
            not math.isfinite(args.magnitude_weight)
            or args.magnitude_weight < 0.0
            or not math.isfinite(args.phase_weight)
            or args.phase_weight < 0.0
        ):
            parser.error("--magnitude-weight and --phase-weight must be nonnegative")
        if args.magnitude_weight == 0.0 and args.phase_weight == 0.0:
            parser.error("--magnitude-weight and --phase-weight cannot both be zero")
    if args.command == "train-hybrid-v31":
        if args.radial_bins < MIN_RADIAL_BINS:
            parser.error(f"--radial-bins must be at least {MIN_RADIAL_BINS}")
        if args.v2_checkpoint is not None:
            parser.error("--v2-checkpoint is supported by train-hybrid-v3 only")
    if args.command == "prepare-data" and not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be greater than 0 and less than 1")

    wildfake_path = (
        args.wildfake_dir
        if args.wildfake_dir is not None
        else args.data_dir / "WildFake"
    )
    if args.command == "prepare-data":
        try:
            prepare_wildfake_data(
                wildfake_path,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    canonical_holdout = args.holdout
    if args.command in (
        "train-source-balanced",
        "train-hybrid",
        "train-hybrid-v2",
        "train-hybrid-v3",
        "train-hybrid-v31",
    ) and args.holdout is not None:
        try:
            canonical_holdout = resolve_wildfake_holdout(
                wildfake_path,
                args.holdout,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    try:
        checkpoint_path = resolve_checkpoint_path(
            args.command,
            args.checkpoint,
            canonical_holdout,
            args.run_name,
        )
    except ValueError as error:
        parser.error(str(error))
    if (
        args.command in (
            "train-hybrid-v2",
            "train-hybrid-v3",
            "train-hybrid-v31",
        )
        and args.checkpoint is None
        and checkpoint_path.exists()
    ):
        version = {
            "train-hybrid-v2": "V2",
            "train-hybrid-v3": "V3",
            "train-hybrid-v31": "V3.1",
        }[args.command]
        parser.error(
            f"Refusing to overwrite existing Hybrid {version} checkpoint: {checkpoint_path}. "
            "Choose a new --run-name or pass an explicit --checkpoint path."
        )
    device = get_device()
    image_size = (args.image_size, args.image_size)
    print(f"Using device: {device}")

    if args.command == "predict":
        if args.image is None:
            raise SystemExit("--image is required for the predict command")
        model = load_model(checkpoint_path, device)
        result = predict_image(model, args.image, device, image_size=image_size)
        print(
            f"Prediction: {result['label']} | confidence: {result['confidence']:.2%} | "
            f"P(real): {result['probability_real']:.4f}"
        )
        return

    if args.command == "validate-bytedance":
        run_bytedance_validation(
            checkpoint_path=checkpoint_path,
            validation_dir=args.validation_dir,
            data_dir=args.data_dir,
            device=device,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
        )
        return

    if args.command == "train-multisource":
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_multisource_data_loaders(
            dataset_path,
            wildfake_path,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
            train_fraction=args.train_fraction,
        )
        model = build_model(device)
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
        )
        _ = load_model(checkpoint_path, device)
        run_bytedance_validation(
            checkpoint_path=checkpoint_path,
            validation_dir=args.validation_dir,
            data_dir=args.data_dir,
            device=device,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
        )
        return

    if args.command == "train-source-balanced":
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path,
            wildfake_path,
            holdout=canonical_holdout,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            image_size=image_size,
            num_workers=args.num_workers,
        )
        model = build_model(device)
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
            heldout_generator=canonical_holdout,
        )
        return

    if args.command == "train-hybrid":
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path,
            wildfake_path,
            holdout=canonical_holdout,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=False,
        )
        model = build_hybrid_model(
            device,
            spatial_checkpoint=args.spatial_checkpoint,
        )
        spatial_initialization = (
            "imagenet"
            if args.spatial_checkpoint is None
            else str(args.spatial_checkpoint.resolve())
        )
        checkpoint_metadata = {
            "model_type": HYBRID_MODEL_TYPE,
            "spatial_feature_dim": SPATIAL_FEATURE_DIM,
            "frequency_feature_dim": FREQUENCY_FEATURE_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "fusion_dropout": FUSION_DROPOUT,
            "fft_preprocessing": "float32 luminance fft2 ortho fftshift log1p per-image-standardization",
            "spatial_initialization": spatial_initialization,
            "heldout_generator_config": canonical_holdout,
            "training_sources": [source.name for source in train_loader.dataset.sources],
            "validation_sources": [source.name for source in validation_loader.dataset.sources],
            "training_config": {
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage1_classifier_learning_rate": 1e-4,
                "stage2_classifier_learning_rate": 1e-4,
                "stage2_backbone_learning_rate": 1e-5,
                "samples_per_epoch": args.samples_per_epoch,
                "seed": args.seed,
            },
        }
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
            heldout_generator=canonical_holdout,
            stage1_epochs=args.stage1_epochs,
            stage1_classifier_learning_rate=1e-4,
            stage2_classifier_learning_rate=1e-4,
            stage2_backbone_learning_rate=1e-5,
            checkpoint_metadata=checkpoint_metadata,
        )
        return

    if args.command == "train-hybrid-v2":
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path,
            wildfake_path,
            holdout=canonical_holdout,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=False,
        )
        model = build_hybrid_v2_model(
            device,
            spatial_checkpoint=args.spatial_checkpoint,
            frequency_scale=args.frequency_scale,
            frequency_branch_dropout=args.frequency_branch_dropout,
            frequency_mask_probability=args.frequency_mask_prob,
        )
        if model.spatial_classifier_loaded:
            print(
                "Hybrid V2 Stage 1: EfficientNet features and the loaded spatial "
                "classifier are frozen; FFT CNN/head train at 5e-5."
            )
        else:
            print(
                "Hybrid V2 Stage 1: EfficientNet features are frozen; the random "
                "spatial classifier trains at 1e-5 and FFT CNN/head train at 5e-5."
            )
        print(
            "Hybrid V2 Stage 2: EfficientNet blocks 0-5 stay frozen; blocks 6-8 "
            "and the spatial classifier train at 1e-5; FFT CNN/head train at 5e-5."
        )
        spatial_initialization = (
            "imagenet"
            if args.spatial_checkpoint is None
            else str(args.spatial_checkpoint.resolve())
        )
        checkpoint_metadata = {
            "model_type": HYBRID_V2_MODEL_TYPE,
            "spatial_feature_dim": V2_SPATIAL_FEATURE_DIM,
            "frequency_feature_dim": V2_FREQUENCY_FEATURE_DIM,
            "frequency_hidden_dim": V2_FREQUENCY_HIDDEN_DIM,
            "frequency_dropout": V2_FREQUENCY_DROPOUT,
            "frequency_scale": args.frequency_scale,
            "frequency_branch_dropout": args.frequency_branch_dropout,
            "frequency_mask_prob": args.frequency_mask_prob,
            "fft_preprocessing": (
                "float32 luminance fft2 ortho fftshift log1p "
                "per-image-standardization"
            ),
            "spatial_initialization": spatial_initialization,
            "spatial_classifier_loaded": model.spatial_classifier_loaded,
            "spatial_classifier_source": model.spatial_classifier_source,
            "heldout_generator_config": canonical_holdout,
            "run_name": None if args.run_name is None else args.run_name.strip(),
            "seed": args.seed,
            "training_sources": [source.name for source in train_loader.dataset.sources],
            "validation_sources": [
                source.name for source in validation_loader.dataset.sources
            ],
            "training_config": {
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage1_frequency_learning_rate": V2_FREQUENCY_LEARNING_RATE,
                "stage1_spatial_classifier_learning_rate": (
                    None
                    if model.spatial_classifier_loaded
                    else V2_SPATIAL_LEARNING_RATE
                ),
                "stage2_frequency_learning_rate": V2_FREQUENCY_LEARNING_RATE,
                "stage2_spatial_learning_rate": V2_SPATIAL_LEARNING_RATE,
                "samples_per_epoch": args.samples_per_epoch,
                "seed": args.seed,
            },
        }
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
            heldout_generator=canonical_holdout,
            stage1_epochs=args.stage1_epochs,
            checkpoint_metadata=checkpoint_metadata,
            stage_configurator=configure_hybrid_v2_stage,
        )
        return

    if args.command == "train-hybrid-v3":
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path,
            wildfake_path,
            holdout=canonical_holdout,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=False,
        )
        model = build_hybrid_v3_model(
            device,
            spatial_checkpoint=args.spatial_checkpoint,
            v2_checkpoint=args.v2_checkpoint,
            frequency_scale=args.frequency_scale,
            magnitude_weight=args.magnitude_weight,
            phase_weight=args.phase_weight,
            frequency_branch_dropout=args.frequency_branch_dropout,
            frequency_mask_probability=args.frequency_mask_prob,
        )
        if model.spatial_classifier_loaded:
            print(
                "Hybrid V3 Stage 1: EfficientNet features and the loaded spatial "
                "classifier are frozen; magnitude and phase paths train at 5e-5."
            )
        else:
            print(
                "Hybrid V3 Stage 1: EfficientNet features are frozen; the random "
                "spatial classifier trains at 1e-5 and magnitude/phase paths train "
                "at 5e-5."
            )
        print(
            "Hybrid V3 Stage 2: EfficientNet blocks 0-5 stay frozen; blocks 6-8 "
            "and the spatial classifier train at 1e-5; magnitude/phase paths train "
            "at 5e-5."
        )
        if args.v2_checkpoint is not None:
            spatial_initialization = str(args.v2_checkpoint.resolve())
            initialization_mode = "hybrid_v2_warm_start"
        elif args.spatial_checkpoint is not None:
            spatial_initialization = str(args.spatial_checkpoint.resolve())
            initialization_mode = "spatial_checkpoint"
        else:
            spatial_initialization = "imagenet"
            initialization_mode = "imagenet"
        checkpoint_metadata = {
            "model_type": HYBRID_V3_MODEL_TYPE,
            "spatial_feature_dim": V3_SPATIAL_FEATURE_DIM,
            "magnitude_feature_dim": V3_MAGNITUDE_FEATURE_DIM,
            "phase_feature_dim": V3_PHASE_FEATURE_DIM,
            "frequency_hidden_dim": V3_FREQUENCY_HIDDEN_DIM,
            "frequency_dropout": V3_FREQUENCY_DROPOUT,
            "frequency_scale": args.frequency_scale,
            "supplied_magnitude_weight": args.magnitude_weight,
            "supplied_phase_weight": args.phase_weight,
            "normalized_magnitude_weight": model.magnitude_weight,
            "normalized_phase_weight": model.phase_weight,
            "frequency_branch_dropout": args.frequency_branch_dropout,
            "frequency_mask_prob": args.frequency_mask_prob,
            "fft_normalization": "ortho",
            "phase_representation": "sin_cos",
            "fft_preprocessing": (
                "single float32 luminance fft2 ortho fftshift; log1p per-image "
                "standardized magnitude and sine/cosine phase"
            ),
            "spatial_initialization": spatial_initialization,
            "initialization_mode": initialization_mode,
            "spatial_classifier_loaded": model.spatial_classifier_loaded,
            "spatial_classifier_source": model.spatial_classifier_source,
            "magnitude_initialized_from_v2": model.magnitude_initialized_from_v2,
            "heldout_generator_config": canonical_holdout,
            "run_name": None if args.run_name is None else args.run_name.strip(),
            "seed": args.seed,
            "training_sources": [source.name for source in train_loader.dataset.sources],
            "validation_sources": [
                source.name for source in validation_loader.dataset.sources
            ],
            "training_config": {
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage1_magnitude_learning_rate": V3_FREQUENCY_LEARNING_RATE,
                "stage1_phase_learning_rate": V3_FREQUENCY_LEARNING_RATE,
                "stage1_spatial_classifier_learning_rate": (
                    None
                    if model.spatial_classifier_loaded
                    else V3_SPATIAL_LEARNING_RATE
                ),
                "stage2_magnitude_learning_rate": V3_FREQUENCY_LEARNING_RATE,
                "stage2_phase_learning_rate": V3_FREQUENCY_LEARNING_RATE,
                "stage2_spatial_learning_rate": V3_SPATIAL_LEARNING_RATE,
                "samples_per_epoch": args.samples_per_epoch,
                "seed": args.seed,
            },
        }
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
            heldout_generator=canonical_holdout,
            stage1_epochs=args.stage1_epochs,
            checkpoint_metadata=checkpoint_metadata,
            stage_configurator=configure_hybrid_v3_stage,
        )
        return

    if args.command == "train-hybrid-v31":
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dataset_path = download_dataset(args.data_dir)
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path,
            wildfake_path,
            holdout=canonical_holdout,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=False,
        )
        model = build_hybrid_v31_model(
            device,
            spatial_checkpoint=args.spatial_checkpoint,
            frequency_scale=args.frequency_scale,
            frequency_branch_dropout=args.frequency_branch_dropout,
            frequency_mask_probability=args.frequency_mask_prob,
            radial_bins=args.radial_bins,
        )
        if model.spatial_classifier_loaded:
            print(
                "Hybrid V3.1 Stage 1: EfficientNet features and the loaded spatial "
                "classifier are frozen; magnitude, phase, radial, and learned "
                "fusion paths train at 5e-5."
            )
        else:
            print(
                "Hybrid V3.1 Stage 1: EfficientNet features are frozen; the random "
                "spatial classifier trains at 1e-5 and all frequency paths train "
                "at 5e-5."
            )
        print(
            "Hybrid V3.1 Stage 2: EfficientNet blocks 0-5 stay frozen; blocks 6-8 "
            "and the spatial classifier train at 1e-5; magnitude, phase, radial, "
            "and learned fusion paths train at 5e-5."
        )
        spatial_initialization = (
            "imagenet"
            if args.spatial_checkpoint is None
            else str(args.spatial_checkpoint.resolve())
        )
        checkpoint_metadata = {
            "model_type": HYBRID_V31_MODEL_TYPE,
            "spatial_feature_dim": V31_SPATIAL_FEATURE_DIM,
            "magnitude_feature_dim": V31_MAGNITUDE_FEATURE_DIM,
            "phase_feature_dim": V31_PHASE_FEATURE_DIM,
            "frequency_hidden_dim": V31_FREQUENCY_HIDDEN_DIM,
            "frequency_dropout": V31_FREQUENCY_DROPOUT,
            "radial_bins": args.radial_bins,
            "radial_hidden_dim": V31_RADIAL_HIDDEN_DIM,
            "radial_dropout": V31_RADIAL_DROPOUT,
            "frequency_scale": args.frequency_scale,
            "frequency_branch_dropout": args.frequency_branch_dropout,
            "frequency_mask_prob": args.frequency_mask_prob,
            "frequency_fusion_type": "learned_softmax",
            "initial_frequency_weights": {
                "magnitude": 1.0 / 3.0,
                "phase": 1.0 / 3.0,
                "radial": 1.0 / 3.0,
            },
            "fft_normalization": "ortho",
            "phase_representation": "sin_cos",
            "radial_representation": (
                "mean unmasked normalized log-magnitude in cached low-to-high "
                "annular bins"
            ),
            "fft_preprocessing": (
                "single float32 luminance fft2 ortho fftshift; log1p per-image "
                "standardized magnitude and sine/cosine phase"
            ),
            "spatial_initialization": spatial_initialization,
            "spatial_classifier_loaded": model.spatial_classifier_loaded,
            "spatial_classifier_source": model.spatial_classifier_source,
            "heldout_generator_config": canonical_holdout,
            "run_name": None if args.run_name is None else args.run_name.strip(),
            "seed": args.seed,
            "training_sources": [source.name for source in train_loader.dataset.sources],
            "validation_sources": [
                source.name for source in validation_loader.dataset.sources
            ],
            "training_config": {
                "stage1_epochs": args.stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "stage1_frequency_learning_rate": V31_FREQUENCY_LEARNING_RATE,
                "stage1_spatial_classifier_learning_rate": (
                    None
                    if model.spatial_classifier_loaded
                    else V31_SPATIAL_LEARNING_RATE
                ),
                "stage2_frequency_learning_rate": V31_FREQUENCY_LEARNING_RATE,
                "stage2_spatial_learning_rate": V31_SPATIAL_LEARNING_RATE,
                "samples_per_epoch": args.samples_per_epoch,
                "seed": args.seed,
            },
        }
        train_staged_model(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path,
            heldout_generator=canonical_holdout,
            stage1_epochs=args.stage1_epochs,
            checkpoint_metadata=checkpoint_metadata,
            stage_configurator=configure_hybrid_v31_stage,
            validation_component_forward=v31_validation_forward,
            epoch_metadata_provider=v31_epoch_metadata,
        )
        return

    dataset_path = download_dataset(args.data_dir)
    if args.command == "evaluate":
        model = load_model(checkpoint_path, device)
        _, test_loader = get_data_loaders(
            dataset_path,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=not expects_unnormalized_input(model),
        )
        metrics = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
        auc_text = "N/A" if metrics["auc_roc"] is None else f"{metrics['auc_roc']:.4f}"
        print(f"Loss: {metrics['loss']:.4f} | accuracy: {metrics['accuracy']:.2%} | AUC-ROC: {auc_text}")
        return
    if args.command == "robustness":
        model = load_model(checkpoint_path, device)
        run_robustness_benchmark(
            model,
            get_test_directory(dataset_path),
            device,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
        )
        return
    train_loader, test_loader = get_data_loaders(
        dataset_path,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    if args.command in ("train", "train-staged"):
        model = build_model(device)
        if args.command == "train":
            train_model(
                model,
                train_loader,
                test_loader,
                device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                checkpoint_path=checkpoint_path,
            )
        else:
            train_staged_model(
                model,
                train_loader,
                test_loader,
                device,
                stage2_epochs=args.stage2_epochs,
                checkpoint_path=checkpoint_path,
            )
        model = load_model(checkpoint_path, device)
        run_robustness_benchmark(
            model,
            get_test_directory(dataset_path),
            device,
            batch_size=args.batch_size,
            image_size=image_size,
            num_workers=args.num_workers,
        )

if __name__ == "__main__":
    main()
