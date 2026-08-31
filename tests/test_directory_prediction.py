from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from PIL import Image

import main as app
from src.predict import discover_input_images, predict_folder


def write_image(path: Path, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color=color).save(path)


class ConstantLogitModel(nn.Module):
    def __init__(self, logit: float, raw_input: bool = False) -> None:
        super().__init__()
        self.logit = float(logit)
        self.expects_unnormalized_input = raw_input
        self.observed_inputs: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.observed_inputs.append(images.detach().cpu())
        return torch.full(
            (images.shape[0], 1),
            self.logit,
            device=images.device,
        )


class DirectoryDiscoveryTests(unittest.TestCase):
    def test_recursively_discovers_supported_images_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            write_image(root / "B" / "second.PNG")
            write_image(root / "a" / "first.jpg")
            (root / "a" / "notes.txt").write_text("ignored")

            discovered = discover_input_images(root)

            self.assertEqual(
                [path.name for path in discovered],
                ["first.jpg", "second.PNG"],
            )

    def test_missing_or_empty_directory_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                discover_input_images(root / "missing")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "no supported images"):
                discover_input_images(empty)


class DirectoryScoringTests(unittest.TestCase):
    def test_outputs_only_image_path_and_continuous_aigc_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input"
            write_image(images / "one.jpg")
            write_image(images / "nested" / "two.png")
            checkpoint = root / "model.pt"
            checkpoint.touch()
            output = root / "predictions.json"
            model = ConstantLogitModel(logit=2.0, raw_input=True)

            with (
                patch("src.predict.load_model", return_value=model) as loader,
                redirect_stdout(io.StringIO()),
            ):
                results = predict_folder(
                    input_dir=images,
                    checkpoint_path=checkpoint,
                    output_path=output,
                    device=torch.device("cpu"),
                    batch_size=2,
                    image_size=(8, 8),
                    num_workers=0,
                )

            expected = 1.0 - torch.sigmoid(torch.tensor(2.0)).item()
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(results), 2)
            self.assertEqual(
                [result["image_path"] for result in results],
                ["nested/two.png", "one.jpg"],
            )
            for result in results:
                self.assertEqual(set(result), {"image_path", "pred"})
                self.assertTrue(math.isclose(result["pred"], expected, abs_tol=1e-7))
                self.assertGreaterEqual(result["pred"], 0.0)
                self.assertLessEqual(result["pred"], 1.0)
            self.assertEqual(json.loads(output.read_text()), results)

    def test_model_aware_preprocessing_uses_raw_input_for_hybrids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input"
            write_image(images / "black.jpg")
            checkpoint = root / "model.pt"
            checkpoint.touch()

            raw_model = ConstantLogitModel(0.0, raw_input=True)
            with (
                patch("src.predict.load_model", return_value=raw_model),
                redirect_stdout(io.StringIO()),
            ):
                predict_folder(
                    input_dir=images,
                    checkpoint_path=checkpoint,
                    output_path=root / "raw.json",
                    device=torch.device("cpu"),
                    batch_size=1,
                    image_size=(8, 8),
                    num_workers=0,
                )
            raw = raw_model.observed_inputs[0]
            self.assertGreaterEqual(float(raw.min()), 0.0)
            self.assertLessEqual(float(raw.max()), 1.0)

            normalized_model = ConstantLogitModel(0.0, raw_input=False)
            with (
                patch(
                    "src.predict.load_model",
                    return_value=normalized_model,
                ),
                redirect_stdout(io.StringIO()),
            ):
                predict_folder(
                    input_dir=images,
                    checkpoint_path=checkpoint,
                    output_path=root / "normalized.json",
                    device=torch.device("cpu"),
                    batch_size=1,
                    image_size=(8, 8),
                    num_workers=0,
                )
            self.assertLess(float(normalized_model.observed_inputs[0].min()), 0.0)

    def test_unreadable_image_does_not_produce_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input"
            images.mkdir()
            (images / "broken.jpg").write_bytes(b"not an image")
            checkpoint = root / "model.pt"
            checkpoint.touch()
            output = root / "predictions.json"

            with patch(
                "src.predict.load_model",
                return_value=ConstantLogitModel(0.0),
            ):
                with self.assertRaisesRegex(ValueError, "Could not read image"):
                    predict_folder(
                        input_dir=images,
                        checkpoint_path=checkpoint,
                        output_path=output,
                        device=torch.device("cpu"),
                        batch_size=1,
                        image_size=(8, 8),
                        num_workers=0,
                    )
            self.assertFalse(output.exists())

    def test_one_logit_per_image_is_required(self) -> None:
        class InvalidModel(ConstantLogitModel):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.zeros(images.shape[0], 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input"
            write_image(images / "one.jpg")
            checkpoint = root / "model.pt"
            checkpoint.touch()
            with patch(
                "src.predict.load_model",
                return_value=InvalidModel(0.0),
            ):
                with self.assertRaisesRegex(ValueError, "one logit per image"):
                    predict_folder(
                        input_dir=images,
                        checkpoint_path=checkpoint,
                        output_path=root / "predictions.json",
                        device=torch.device("cpu"),
                        batch_size=1,
                        image_size=(8, 8),
                        num_workers=0,
                    )


class DirectoryPredictionCliTests(unittest.TestCase):
    def test_cli_requires_directory_and_checkpoint_and_has_no_threshold(self) -> None:
        parser = app.build_parser()
        args = parser.parse_args(
            ["predict", "--input-dir", "images", "--checkpoint", "model.pt"]
        )
        self.assertEqual(args.output, Path("predictions.json"))
        with (
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parser.parse_args(
                [
                    "predict", "--input-dir",
                    "images",
                    "--checkpoint",
                    "model.pt",
                    "--probability-threshold",
                    "0.5",
                ]
            )

    def test_cli_dispatches_to_directory_scorer(self) -> None:
        scorer = Mock()
        with (
            patch.object(__import__("sys"), "argv", [
                "main.py", "predict", "--input-dir", "images", "--checkpoint",
                "model.pt", "--output", "scores.json", "--batch-size", "8",
                "--num-workers", "0",
            ]),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.predict_folder", scorer),
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertEqual(scorer.call_args.kwargs["input_dir"], Path("images"))
        self.assertEqual(scorer.call_args.kwargs["checkpoint_path"], Path("model.pt"))
        self.assertEqual(scorer.call_args.kwargs["output_path"], Path("scores.json"))
        self.assertEqual(scorer.call_args.kwargs["batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
