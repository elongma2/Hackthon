"""Public CLI for final training, inference, preparation, and evaluation."""

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
from src.hybrid_v31_model import (
    DEFAULT_FREQUENCY_BRANCH_DROPOUT,
    DEFAULT_FREQUENCY_MASK_PROB,
    DEFAULT_FREQUENCY_SCALE,
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
from src.multisource_dataset import resolve_wildfake_holdout
from src.predict import predict_folder
from src.robustness import run_robustness_benchmark
from src.robustness_matrix import ROBUSTNESS_CONDITION_IDS, run_robustness_matrix
from src.source_balanced import (
    DEFAULT_BALANCED_SEED,
    DEFAULT_SAMPLES_PER_EPOCH,
    get_source_balanced_data_loaders,
)
from src.train import train_staged_model


DEFAULT_STAGED_CHECKPOINT = Path("checkpoints/efficientnet_staged_best.pt")
DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT = Path(
    "checkpoints/efficientnet_balanced_all_sources_best.pt"
)
DEFAULT_ALL_SOURCE_HYBRID_V31_CHECKPOINT = Path(
    "checkpoints/hybrid_v31_balanced_all_sources_best.pt"
)
PUBLIC_COMMANDS = (
    "prepare-data",
    "train-source-balanced",
    "train-hybrid-v31",
    "predict",
    "evaluate",
    "validate-bytedance",
    "robustness",
    "robustness-matrix",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the focused public command-line interface for the final solution."""
    parser = argparse.ArgumentParser(
        description="Train and run the final source-balanced Hybrid V3.1 detector."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_runtime(command: argparse.ArgumentParser) -> None:
        """Add common deterministic DataLoader options to one command."""
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--image-size", type=int, default=224)
        command.add_argument("--num-workers", type=int, default=2)

    def add_data(command: argparse.ArgumentParser, *, wildfake: bool = False) -> None:
        """Add CIFAKE and optional WildFake path options to one command."""
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        if wildfake:
            command.add_argument("--wildfake-dir", type=Path, default=None)

    prepare = commands.add_parser("prepare-data", help="Prepare WildFake TRAIN/TEST roots.")
    add_data(prepare, wildfake=True)
    prepare.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    prepare.add_argument("--seed", type=int, default=DEFAULT_BALANCED_SEED)

    spatial = commands.add_parser("train-source-balanced", help="Train final EfficientNet-B0.")
    add_data(spatial, wildfake=True)
    add_runtime(spatial)
    spatial.add_argument("--checkpoint", type=Path, default=None)
    spatial.add_argument("--holdout", type=str, default=None)
    spatial.add_argument("--stage1-epochs", type=int, default=2)
    spatial.add_argument("--stage2-epochs", type=int, default=5)
    spatial.add_argument("--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH)
    spatial.add_argument("--seed", type=int, default=DEFAULT_BALANCED_SEED)

    hybrid = commands.add_parser("train-hybrid-v31", help="Train final Hybrid V3.1.")
    add_data(hybrid, wildfake=True)
    add_runtime(hybrid)
    hybrid.add_argument("--checkpoint", type=Path, default=None)
    hybrid.add_argument("--spatial-checkpoint", type=Path, default=None)
    hybrid.add_argument("--holdout", type=str, default=None)
    hybrid.add_argument("--run-name", type=str, default=None)
    hybrid.add_argument("--stage1-epochs", type=int, default=2)
    hybrid.add_argument("--stage2-epochs", type=int, default=5)
    hybrid.add_argument("--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH)
    hybrid.add_argument("--seed", type=int, default=DEFAULT_BALANCED_SEED)
    hybrid.add_argument("--frequency-scale", type=float, default=DEFAULT_FREQUENCY_SCALE)
    hybrid.add_argument("--frequency-branch-dropout", type=float, default=DEFAULT_FREQUENCY_BRANCH_DROPOUT)
    hybrid.add_argument("--frequency-mask-prob", type=float, default=DEFAULT_FREQUENCY_MASK_PROB)
    hybrid.add_argument("--radial-bins", type=int, default=DEFAULT_RADIAL_BINS)

    predict = commands.add_parser("predict", help="Write P(AIGC) JSON for an unlabeled folder.")
    add_runtime(predict)
    predict.add_argument("--input-dir", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("predictions.json"))

    evaluate_command = commands.add_parser("evaluate", help="Evaluate on CIFAKE test data.")
    add_data(evaluate_command)
    add_runtime(evaluate_command)
    evaluate_command.add_argument("--checkpoint", type=Path, required=True)

    validate = commands.add_parser("validate-bytedance", help="Evaluate ByteDance labelled data.")
    add_data(validate)
    add_runtime(validate)
    validate.add_argument("--validation-dir", type=Path, default=Path("validation"))
    validate.add_argument("--checkpoint", type=Path, default=None)

    robustness = commands.add_parser("robustness", help="Run the compact CIFAKE robustness test.")
    add_data(robustness)
    add_runtime(robustness)
    robustness.add_argument("--checkpoint", type=Path, required=True)

    matrix = commands.add_parser("robustness-matrix", help="Evaluate the official robustness matrix.")
    add_runtime(matrix)
    matrix.add_argument("--checkpoint", type=Path, required=True)
    matrix.add_argument("--validation-dir", type=Path, default=Path("validation"))
    matrix.add_argument("--distorted-dir", type=Path, default=None)
    matrix.add_argument("--probability-threshold", type=float, default=0.5)
    matrix.add_argument("--only", choices=ROBUSTNESS_CONDITION_IDS, default=None)
    matrix.add_argument("--run-name", type=str, required=True)
    matrix.add_argument("--seed", type=int, default=DEFAULT_BALANCED_SEED)
    return parser


def _slug(value: str, option: str) -> str:
    """Sanitize a name for safe automatic checkpoint and result paths."""
    result = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    if not result:
        raise ValueError(f"{option} must contain at least one letter or number")
    return result


def resolve_checkpoint_path(
    command: str,
    checkpoint: Path | None,
    holdout: str | None = None,
    run_name: str | None = None,
) -> Path:
    """Resolve final training outputs while honoring an explicit checkpoint path."""
    if checkpoint is not None:
        return checkpoint
    if command == "train-source-balanced":
        if holdout is None:
            return DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT
        return Path(f"checkpoints/efficientnet_balanced_holdout_{_slug(holdout, '--holdout')}_best.pt")
    if command == "train-hybrid-v31":
        holdout_slug = None if holdout is None else _slug(holdout, "--holdout")
        if run_name is not None:
            run_slug = _slug(run_name, "--run-name")
            suffix = "all_sources" if holdout_slug is None else f"holdout_{holdout_slug}"
            return Path(f"checkpoints/hybrid_v31_{run_slug}_{suffix}_best.pt")
        if holdout_slug is None:
            return DEFAULT_ALL_SOURCE_HYBRID_V31_CHECKPOINT
        return Path(f"checkpoints/hybrid_v31_balanced_holdout_{holdout_slug}_best.pt")
    if command == "validate-bytedance":
        return DEFAULT_STAGED_CHECKPOINT
    raise ValueError(f"--checkpoint is required for {command}")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid command-specific values before loading data or checkpoints."""
    if (
        getattr(args, "batch_size", 1) < 1
        or getattr(args, "image_size", 1) < 1
        or getattr(args, "num_workers", 0) < 0
    ):
        parser.error("batch size/image size must be positive and workers cannot be negative")
    if args.command in {"train-source-balanced", "train-hybrid-v31"}:
        if min(args.stage1_epochs, args.stage2_epochs, args.samples_per_epoch) < 1:
            parser.error("stage epochs and samples-per-epoch must be at least 1")
    if args.command == "prepare-data" and not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio must be greater than 0 and less than 1")
    if args.command == "train-hybrid-v31":
        if not math.isfinite(args.frequency_scale) or args.frequency_scale < 0.0:
            parser.error("--frequency-scale must be finite and nonnegative")
        if not 0.0 <= args.frequency_branch_dropout <= 1.0:
            parser.error("--frequency-branch-dropout must be between 0 and 1")
        if not 0.0 <= args.frequency_mask_prob <= 1.0:
            parser.error("--frequency-mask-prob must be between 0 and 1")
        if args.radial_bins < MIN_RADIAL_BINS:
            parser.error(f"--radial-bins must be at least {MIN_RADIAL_BINS}")
    if args.command == "predict":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for predict")
        if args.input_dir is None:
            parser.error("--input-dir is required for predict")
    if args.command in {"evaluate", "robustness", "robustness-matrix"} and args.checkpoint is None:
        parser.error(f"--checkpoint is required for {args.command}")
    if args.command == "robustness-matrix":
        if args.run_name is None:
            parser.error("--run-name is required for robustness-matrix")
        try:
            _slug(args.run_name, "--run-name")
        except ValueError as error:
            parser.error(str(error))
        if not math.isfinite(args.probability_threshold) or not 0.0 <= args.probability_threshold <= 1.0:
            parser.error("--probability-threshold must be between 0 and 1")


def _hybrid_metadata(model, args, train_loader, validation_loader, holdout, spatial_source) -> dict[str, object]:
    """Build complete architecture and training metadata for a strict V3.1 checkpoint."""
    return {
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
        "initial_frequency_weights": {"magnitude": 1 / 3, "phase": 1 / 3, "radial": 1 / 3},
        "fft_normalization": "ortho",
        "phase_representation": "sin_cos",
        "radial_representation": "mean unmasked normalized log-magnitude in low-to-high annular bins",
        "fft_preprocessing": "single float32 luminance fft2 ortho fftshift; standardized log-magnitude and sin/cos phase",
        "spatial_initialization": spatial_source,
        "spatial_classifier_loaded": model.spatial_classifier_loaded,
        "spatial_classifier_source": model.spatial_classifier_source,
        "heldout_generator_config": holdout,
        "run_name": None if args.run_name is None else args.run_name.strip(),
        "seed": args.seed,
        "training_sources": [source.name for source in train_loader.dataset.sources],
        "validation_sources": [source.name for source in validation_loader.dataset.sources],
        "training_config": {
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage1_frequency_learning_rate": V31_FREQUENCY_LEARNING_RATE,
            "stage1_spatial_classifier_learning_rate": None if model.spatial_classifier_loaded else V31_SPATIAL_LEARNING_RATE,
            "stage2_frequency_learning_rate": V31_FREQUENCY_LEARNING_RATE,
            "stage2_spatial_learning_rate": V31_SPATIAL_LEARNING_RATE,
            "samples_per_epoch": args.samples_per_epoch,
            "seed": args.seed,
        },
    }


def main() -> None:
    """Dispatch one final production command without exposing archived experiments."""
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    data_dir = getattr(args, "data_dir", Path("data/raw"))
    wildfake_path = getattr(args, "wildfake_dir", None) or data_dir / "WildFake"

    if args.command == "prepare-data":
        try:
            prepare_wildfake_data(wildfake_path, train_ratio=args.train_ratio, seed=args.seed)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    holdout = getattr(args, "holdout", None)
    if args.command in {"train-source-balanced", "train-hybrid-v31"} and holdout is not None:
        try:
            holdout = resolve_wildfake_holdout(wildfake_path, holdout)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    try:
        checkpoint_path = resolve_checkpoint_path(
            args.command,
            args.checkpoint,
            holdout,
            getattr(args, "run_name", None),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.command == "train-hybrid-v31" and args.checkpoint is None and checkpoint_path.exists():
        parser.error(
            f"Refusing to overwrite existing Hybrid V3.1 checkpoint: {checkpoint_path}. "
            "Choose a new --run-name or an explicit --checkpoint path."
        )

    device = get_device()
    image_size = (args.image_size, args.image_size)
    print(f"Using device: {device}")

    if args.command == "predict":
        try:
            predict_folder(
                input_dir=args.input_dir,
                checkpoint_path=checkpoint_path,
                output_path=args.output,
                device=device,
                batch_size=args.batch_size,
                image_size=image_size,
                num_workers=args.num_workers,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))
        return
    if args.command == "validate-bytedance":
        run_bytedance_validation(
            checkpoint_path=checkpoint_path, validation_dir=args.validation_dir,
            data_dir=args.data_dir, device=device, batch_size=args.batch_size,
            image_size=image_size, num_workers=args.num_workers,
        )
        return
    if args.command == "robustness-matrix":
        try:
            run_robustness_matrix(
                checkpoint_path=checkpoint_path, validation_dir=args.validation_dir,
                distorted_dir=args.distorted_dir, device=device,
                probability_threshold=args.probability_threshold,
                batch_size=args.batch_size, image_size=image_size,
                num_workers=args.num_workers, run_name=args.run_name,
                only=args.only, seed=args.seed,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    dataset_path = download_dataset(args.data_dir)
    if args.command in {"train-source-balanced", "train-hybrid-v31"}:
        hybrid = args.command == "train-hybrid-v31"
        train_loader, validation_loader = get_source_balanced_data_loaders(
            dataset_path, wildfake_path, holdout=holdout,
            batch_size=args.batch_size, samples_per_epoch=args.samples_per_epoch,
            seed=args.seed, image_size=image_size, num_workers=args.num_workers,
            normalize_inputs=not hybrid,
        )
        if not hybrid:
            model = build_model(device)
            train_staged_model(
                model, train_loader, validation_loader, device,
                stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
                checkpoint_path=checkpoint_path, heldout_generator=holdout,
            )
            return

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_hybrid_v31_model(
            device, spatial_checkpoint=args.spatial_checkpoint,
            frequency_scale=args.frequency_scale,
            frequency_branch_dropout=args.frequency_branch_dropout,
            frequency_mask_probability=args.frequency_mask_prob,
            radial_bins=args.radial_bins,
        )
        spatial_source = "imagenet" if args.spatial_checkpoint is None else str(args.spatial_checkpoint.resolve())
        metadata = _hybrid_metadata(model, args, train_loader, validation_loader, holdout, spatial_source)
        train_staged_model(
            model, train_loader, validation_loader, device,
            stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
            checkpoint_path=checkpoint_path, heldout_generator=holdout,
            checkpoint_metadata=metadata,
            stage_configurator=configure_hybrid_v31_stage,
            validation_component_forward=v31_validation_forward,
            epoch_metadata_provider=v31_epoch_metadata,
        )
        return

    model = load_model(checkpoint_path, device)
    if args.command == "evaluate":
        _, loader = get_data_loaders(
            dataset_path, batch_size=args.batch_size, image_size=image_size,
            num_workers=args.num_workers,
            normalize_inputs=not expects_unnormalized_input(model),
        )
        metrics = evaluate(model, loader, nn.BCEWithLogitsLoss(), device)
        auc_text = "N/A" if metrics["auc_roc"] is None else f"{metrics['auc_roc']:.4f}"
        print(f"Loss: {metrics['loss']:.4f} | accuracy: {metrics['accuracy']:.2%} | AUC-ROC: {auc_text}")
        return
    if args.command == "robustness":
        run_robustness_benchmark(
            model, get_test_directory(dataset_path), device,
            batch_size=args.batch_size, image_size=image_size,
            num_workers=args.num_workers,
        )
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
