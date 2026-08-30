from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from main import DEFAULT_STAGED_CHECKPOINT, build_parser, resolve_checkpoint_path
from src.bytedance_validation import (
    calculate_bytedance_metrics,
    get_class_counts,
    run_bytedance_validation,
)
from src.evaluate import evaluate
from src.model import load_model
from src.transforms import build_eval_transforms


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def create_cifake_fixture(root: Path) -> Path:
    for split in ("train", "test"):
        write_image(root / split / "FAKE" / "fake.jpg", (255, 0, 0))
        write_image(root / split / "REAL" / "real.jpg", (0, 255, 0))
    return root


def create_bytedance_fixture(root: Path) -> Path:
    write_image(
        root / "FAKE" / "Advanced" / "DALLE3" / "dalle3" / "nested" / "fake.jpg",
        (255, 0, 0),
    )
    write_image(root / "REAL" / "val2017" / "real.jpg", (0, 255, 0))
    return root


class ConstantLogitModel(nn.Module):
    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(logit))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(inputs.shape[0], 1)


class ByteDanceValidationTests(unittest.TestCase):
    def test_imagefolder_recurses_into_nested_fake_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_bytedance_fixture(Path(directory) / "validation")
            dataset = ImageFolder(root, transform=build_eval_transforms((8, 8)))

            self.assertEqual(dataset.class_to_idx, {"FAKE": 0, "REAL": 1})
            self.assertEqual(len(dataset), 2)
            self.assertEqual(get_class_counts(dataset), {"FAKE": 1, "REAL": 1})
            fake_path = next(path for path, label in dataset.samples if label == 0)
            self.assertIn(str(Path("Advanced") / "DALLE3" / "dalle3"), fake_path)

    def test_metrics_use_fake_as_positive_when_raw_probability_is_real(self) -> None:
        metrics = calculate_bytedance_metrics(
            {"FAKE": 0, "REAL": 1},
            {"FAKE": 0, "REAL": 1},
            raw_probabilities=[0.1, 0.8, 0.9, 0.2],
            validation_labels=[0, 0, 1, 1],
        )

        self.assertEqual(metrics["raw_probability_class"], "REAL")
        self.assertEqual(metrics["aigc_probability_expression"], "1 - sigmoid(logit)")
        for actual, expected in zip(
            metrics["aigc_probabilities"],
            [0.9, 0.2, 0.1, 0.8],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(metrics["aigc_labels"], [1, 1, 0, 0])
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["incorrect"], 2)
        self.assertEqual(metrics["true_fake_predicted_fake"], 1)
        self.assertEqual(metrics["true_fake_predicted_real"], 1)
        self.assertEqual(metrics["true_real_predicted_real"], 1)
        self.assertEqual(metrics["true_real_predicted_fake"], 1)
        self.assertAlmostEqual(metrics["auc_roc_aigc"], 0.75)

    def test_metrics_derive_reversed_training_mapping_without_assuming_indices(self) -> None:
        metrics = calculate_bytedance_metrics(
            {"REAL": 0, "FAKE": 1},
            {"FAKE": 0, "REAL": 1},
            raw_probabilities=[0.9, 0.1],
            validation_labels=[0, 1],
        )

        self.assertEqual(metrics["raw_probability_class"], "FAKE")
        self.assertEqual(metrics["aigc_probability_expression"], "sigmoid(logit)")
        self.assertEqual(metrics["correct"], 2)
        self.assertAlmostEqual(metrics["auc_roc_aigc"], 1.0)

    def test_existing_evaluator_does_not_change_model_parameters_or_buffers(self) -> None:
        model = ConstantLogitModel()
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}

        evaluate(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"))

        after = model.state_dict()
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_workflow_warns_on_count_mismatch_and_remains_inference_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cifake_root = create_cifake_fixture(root / "cifake")
            validation_root = create_bytedance_fixture(root / "validation")
            model = ConstantLogitModel()
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            output = io.StringIO()

            with (
                patch(
                    "src.bytedance_validation.download_dataset",
                    return_value=cifake_root,
                ),
                patch("src.bytedance_validation.load_model", return_value=model),
                patch(
                    "src.bytedance_validation.DataLoader",
                    wraps=DataLoader,
                ) as loader_mock,
                redirect_stdout(output),
            ):
                result = run_bytedance_validation(
                    checkpoint_path=root / "checkpoint.pt",
                    validation_dir=validation_root,
                    data_dir=root / "unused",
                    device=torch.device("cpu"),
                    batch_size=2,
                    image_size=(8, 8),
                    num_workers=0,
                )

            rendered = output.getvalue()
            self.assertIn("WARNING: ByteDance validation image count", rendered)
            self.assertIn("Found FAKE=1, REAL=1, total=2", rendered)
            self.assertIn("Raw model probability represents: REAL", rendered)
            self.assertIn("1 - sigmoid(logit)", rendered)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["class_counts"], {"FAKE": 1, "REAL": 1})
            self.assertIs(loader_mock.call_args.kwargs["shuffle"], False)
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[name], value), name)

    def test_cli_defaults_and_checkpoint_overrides(self) -> None:
        args = build_parser().parse_args(["validate-bytedance"])
        self.assertEqual(args.validation_dir, Path("validation"))
        self.assertEqual(
            resolve_checkpoint_path(args.command, args.checkpoint),
            DEFAULT_STAGED_CHECKPOINT,
        )

        baseline = Path("checkpoints/best_model.pt")
        args = build_parser().parse_args(
            ["validate-bytedance", "--checkpoint", str(baseline)]
        )
        self.assertEqual(resolve_checkpoint_path(args.command, args.checkpoint), baseline)

    def test_plain_state_dict_checkpoint_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "plain.pt"
            torch.save(ConstantLogitModel().state_dict(), checkpoint_path)

            with patch("src.model.build_model", return_value=ConstantLogitModel()):
                model = load_model(checkpoint_path, torch.device("cpu"))

            self.assertFalse(model.training)
            self.assertEqual(model.logit.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
