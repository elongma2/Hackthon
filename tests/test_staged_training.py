from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import load_model
from src.train import (
    configure_head_only,
    configure_partial_unfreezing,
    train_one_epoch,
    train_staged_model,
)


class TinyEfficientNet(nn.Module):
    """Small EfficientNet-shaped model for training-control tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(4, 4),
                    nn.BatchNorm1d(4),
                    nn.ReLU(),
                )
                for _ in range(9)
            ]
        )
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(4, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def make_loader() -> DataLoader:
    inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


class StagedFreezingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.device = torch.device("cpu")

    def assert_state_equal(
        self,
        before: dict[str, torch.Tensor],
        after: dict[str, torch.Tensor],
    ) -> None:
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_head_only_training_keeps_entire_backbone_fixed(self) -> None:
        model = TinyEfficientNet()
        frozen_blocks = configure_head_only(model)
        feature_state = clone_state(model.features)
        classifier_before = clone_state(model.classifier)
        optimizer = torch.optim.AdamW(
            [{"params": model.classifier.parameters(), "lr": 1e-3, "name": "classifier"}],
            weight_decay=1e-2,
        )

        train_one_epoch(
            model,
            make_loader(),
            nn.BCEWithLogitsLoss(),
            optimizer,
            self.device,
            frozen_modules=frozen_blocks,
        )

        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        self.assert_state_equal(feature_state, model.features.state_dict())
        self.assertTrue(
            any(
                not torch.equal(classifier_before[name], value)
                for name, value in model.classifier.state_dict().items()
            )
        )
        self.assertTrue(all(not block.training for block in model.features))

    def test_partial_unfreezing_updates_only_features_six_through_eight(self) -> None:
        model = TinyEfficientNet()
        frozen_blocks, trainable_blocks = configure_partial_unfreezing(model, 3)
        frozen_state = clone_state(nn.Sequential(*frozen_blocks))
        trainable_state = clone_state(nn.Sequential(*trainable_blocks))
        backbone_parameters = [
            parameter for block in trainable_blocks for parameter in block.parameters()
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": 1e-4, "name": "classifier"},
                {"params": backbone_parameters, "lr": 1e-5, "name": "backbone"},
            ],
            weight_decay=1e-2,
        )

        train_one_epoch(
            model,
            make_loader(),
            nn.BCEWithLogitsLoss(),
            optimizer,
            self.device,
            frozen_modules=frozen_blocks,
        )

        for index, block in enumerate(model.features):
            expected_trainable = index >= 6
            self.assertTrue(
                all(parameter.requires_grad is expected_trainable for parameter in block.parameters())
            )
            self.assertEqual(block.training, expected_trainable)
        self.assert_state_equal(frozen_state, nn.Sequential(*frozen_blocks).state_dict())
        self.assertTrue(
            any(
                not torch.equal(trainable_state[name], value)
                for name, value in nn.Sequential(*trainable_blocks).state_dict().items()
            )
        )
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["classifier", "backbone"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-4, 1e-5])

    def test_staged_training_uses_cosine_lrs_and_best_auc_checkpoint(self) -> None:
        model = TinyEfficientNet()
        auc_values = iter([0.50, 0.60, 0.55, 0.65, 0.64, 0.70, 0.69])

        def fake_train_one_epoch(
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            probability_threshold: float = 0.5,
            frozen_modules: tuple[nn.Module, ...] = (),
        ) -> tuple[float, float]:
            del model, dataloader, criterion, device, probability_threshold, frozen_modules
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return 0.25, 0.75

        def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            return {"loss": 0.20, "accuracy": 0.80, "auc_roc": next(auc_values)}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "staged.pt"
            with (
                patch("src.train.train_one_epoch", side_effect=fake_train_one_epoch),
                patch("src.train.evaluate", side_effect=fake_evaluate),
                patch("src.train.torch.save", wraps=torch.save) as save_mock,
            ):
                result = train_staged_model(
                    model,
                    make_loader(),
                    make_loader(),
                    self.device,
                    stage2_epochs=5,
                    checkpoint_path=checkpoint_path,
                )

            history = result["history"]
            self.assertEqual(len(history), 7)
            self.assertEqual([entry["stage"] for entry in history[:2]], ["head-only"] * 2)
            self.assertEqual(
                [entry["stage"] for entry in history[2:]],
                ["partial-unfreezing"] * 5,
            )
            self.assertEqual(save_mock.call_count, 4)
            self.assertEqual(result["best_auc_roc"], 0.70)

            expected_classifier_lrs = [
                1e-4,
                9.045084971874738e-5,
                6.545084971874738e-5,
                3.4549150281252636e-5,
                9.549150281252633e-6,
            ]
            for entry, expected_lr in zip(history[2:], expected_classifier_lrs):
                learning_rates = entry["learning_rates"]
                self.assertAlmostEqual(learning_rates["classifier"], expected_lr)
                self.assertAlmostEqual(learning_rates["backbone"], expected_lr / 10)

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 6)
            self.assertEqual(checkpoint["stage"], "partial-unfreezing")
            self.assertEqual(checkpoint["validation_auc_roc"], 0.70)
            self.assertIn("model_state_dict", checkpoint)

            with patch("src.model.build_model", return_value=TinyEfficientNet()):
                loaded_model = load_model(checkpoint_path, self.device)
            self.assertFalse(loaded_model.training)


if __name__ == "__main__":
    unittest.main()
