"""Command-line entry point for training and evaluating the CIFAKE detector."""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import torch.nn as nn

from src.bytedance_validation import run_bytedance_validation
from src.data_preparation import DEFAULT_TRAIN_RATIO, prepare_wildfake_data
from src.dataset import download_dataset, get_data_loaders, get_test_directory
from src.evaluate import evaluate
from src.model import build_model, get_device, load_model
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
            "a holdout-specific checkpoint for train-source-balanced, the staged "
            "checkpoint for train-staged and validate-bytedance, and the baseline "
            "checkpoint otherwise."
        ),
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=None,
        help=(
            "Optional WildFake FAKE source excluded from train-source-balanced. "
            "Omit it to train on every prepared source."
        ),
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
        "--stage2-epochs",
        type=int,
        default=5,
        help="Partial-unfreezing epochs after two fixed head-only epochs.",
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
    if args.command == "train-source-balanced" and args.holdout is not None:
        try:
            canonical_holdout = resolve_wildfake_holdout(
                wildfake_path,
                args.holdout,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    checkpoint_path = resolve_checkpoint_path(
        args.command,
        args.checkpoint,
        canonical_holdout,
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

    dataset_path = download_dataset(args.data_dir)
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
    elif args.command == "evaluate":
        model = load_model(checkpoint_path, device)
        metrics = evaluate(model, test_loader, nn.BCEWithLogitsLoss(), device)
        auc_text = "N/A" if metrics["auc_roc"] is None else f"{metrics['auc_roc']:.4f}"
        print(f"Loss: {metrics['loss']:.4f} | accuracy: {metrics['accuracy']:.2%} | AUC-ROC: {auc_text}")
    else:
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
