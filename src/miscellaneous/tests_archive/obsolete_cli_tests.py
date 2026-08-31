"""Archived tests for superseded public CLI commands."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import main as app


class MultiSourceCliTests(unittest.TestCase):
    def test_checkpoint_defaults_and_override(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-multisource", None),
            app.DEFAULT_MULTISOURCE_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-staged", None),
            app.DEFAULT_STAGED_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train", None),
            app.DEFAULT_CHECKPOINT,
        )
        custom_checkpoint = Path("checkpoints/custom_multisource.pt")
        self.assertEqual(
            app.resolve_checkpoint_path("train-multisource", custom_checkpoint),
            custom_checkpoint,
        )

    def test_multisource_dispatch_uses_one_checkpoint_for_train_load_and_bytedance(self) -> None:
        device = torch.device("cpu")
        cifake_root = Path("resolved/cifake")
        model = Mock(name="model")
        train_loader = Mock(name="train_loader")
        validation_loader = Mock(name="validation_loader")
        output = io.StringIO()

        with (
            patch.object(sys, "argv", ["main.py", "train-multisource"]),
            patch("main.get_device", return_value=device),
            patch("main.download_dataset", return_value=cifake_root) as download_mock,
            patch(
                "main.get_multisource_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loaders_mock,
            patch("main.get_data_loaders") as cifake_loaders_mock,
            patch("main.build_model", return_value=model),
            patch("main.train_staged_model") as train_mock,
            patch("main.load_model") as load_mock,
            patch("main.run_bytedance_validation") as bytedance_mock,
            patch("main.run_robustness_benchmark") as robustness_mock,
            redirect_stdout(output),
        ):
            app.main()

        checkpoint = app.DEFAULT_MULTISOURCE_CHECKPOINT
        download_mock.assert_called_once_with(Path("data/raw"))
        loaders_mock.assert_called_once_with(
            cifake_root,
            Path("data/raw") / "WildFake",
            batch_size=32,
            image_size=(224, 224),
            num_workers=2,
            train_fraction=1.0,
        )
        train_mock.assert_called_once_with(
            model,
            train_loader,
            validation_loader,
            device,
            stage2_epochs=5,
            checkpoint_path=checkpoint,
        )
        load_mock.assert_called_once_with(checkpoint, device)
        self.assertEqual(
            bytedance_mock.call_args.kwargs["checkpoint_path"],
            checkpoint,
        )
        self.assertNotEqual(
            bytedance_mock.call_args.kwargs["checkpoint_path"],
            app.DEFAULT_STAGED_CHECKPOINT,
        )
        cifake_loaders_mock.assert_not_called()
        robustness_mock.assert_not_called()

    def test_explicit_checkpoint_flows_through_entire_multisource_dispatch(self) -> None:
        custom_checkpoint = Path("checkpoints/experiment_02.pt")
        device = torch.device("cpu")

        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "train-multisource",
                    "--checkpoint",
                    str(custom_checkpoint),
                    "--train-fraction",
                    "0.5",
                ],
            ),
            patch("main.get_device", return_value=device),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_multisource_data_loaders",
                return_value=(Mock(), Mock()),
            ) as loaders_mock,
            patch("main.build_model", return_value=Mock()),
            patch("main.train_staged_model") as train_mock,
            patch("main.load_model") as load_mock,
            patch("main.run_bytedance_validation") as bytedance_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertEqual(train_mock.call_args.kwargs["checkpoint_path"], custom_checkpoint)
        self.assertEqual(loaders_mock.call_args.kwargs["train_fraction"], 0.5)
        load_mock.assert_called_once_with(custom_checkpoint, device)
        self.assertEqual(
            bytedance_mock.call_args.kwargs["checkpoint_path"],
            custom_checkpoint,
        )

    def test_prepare_data_dispatch_is_isolated_from_training_and_cifake(self) -> None:
        custom_root = Path("datasets/WildFake")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "prepare-data",
                    "--wildfake-dir",
                    str(custom_root),
                    "--train-ratio",
                    "0.75",
                    "--seed",
                    "17",
                ],
            ),
            patch("main.prepare_wildfake_data") as prepare_mock,
            patch("main.get_device") as device_mock,
            patch("main.download_dataset") as download_mock,
            patch("main.build_model") as model_mock,
        ):
            app.main()

        prepare_mock.assert_called_once_with(custom_root, train_ratio=0.75, seed=17)
        device_mock.assert_not_called()
        download_mock.assert_not_called()
        model_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
