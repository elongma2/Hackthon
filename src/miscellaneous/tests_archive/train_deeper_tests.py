from __future__ import annotations

import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import main as app
from src.multisource_dataset import FAKE_LABEL, REAL_LABEL, ImageSource
from src.train import train_one_epoch
from src.miscellaneous.train_deeper import (
    DEFAULT_CLASSIFIER_LR,
    DEFAULT_LATE_BLOCKS_LR,
    DEFAULT_MIDDLE_BLOCKS_LR,
    DEFAULT_WEIGHT_DECAY,
    _is_better_candidate,
    _write_checkpoint_exclusively,
    calculate_progressive_validation_metrics,
    clone_model_state_to_cpu,
    configure_progressive_stage,
    create_progressive_optimizer,
    train_progressive_deeper,
    transition_to_stage2,
    transition_to_stage3,
    verify_heldout_validation_contract,
)


class TinyEfficientNet(nn.Module):
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
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _initialize_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameters: list[nn.Parameter],
) -> None:
    optimizer.zero_grad()
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def _copy_parameter_state(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, value in optimizer.state[parameter].items():
        copied[key] = value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    return copied


def _assert_parameter_state_equal(
    test: unittest.TestCase,
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    expected: dict[str, object],
) -> None:
    actual = optimizer.state[parameter]
    test.assertEqual(set(actual), set(expected))
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, torch.Tensor):
            test.assertTrue(torch.equal(actual_value, expected_value))
        else:
            test.assertEqual(actual_value, expected_value)


def _source_loaders() -> tuple[SimpleNamespace, SimpleNamespace]:
    train_sources = (
        ImageSource("CIFAKE train FAKE", Path("cifake-fake"), FAKE_LABEL),
        ImageSource("WildFake ADM train", Path("adm"), FAKE_LABEL),
        ImageSource("CIFAKE train REAL", Path("cifake-real"), REAL_LABEL),
    )
    validation_sources = (
        ImageSource("WildFake DDPM test", Path("ddpm"), FAKE_LABEL),
        ImageSource("CIFAKE test REAL", Path("cifake-test-real"), REAL_LABEL),
    )
    train_dataset = SimpleNamespace(
        sources=train_sources,
        samples=[
            (Path("fake.jpg"), FAKE_LABEL, "CIFAKE train FAKE"),
            (Path("adm.jpg"), FAKE_LABEL, "WildFake ADM train"),
            (Path("real.jpg"), REAL_LABEL, "CIFAKE train REAL"),
        ],
    )
    validation_dataset = SimpleNamespace(
        sources=validation_sources,
        samples=[
            (Path("ddpm.jpg"), FAKE_LABEL, "WildFake DDPM test"),
            (Path("real-test.jpg"), REAL_LABEL, "CIFAKE test REAL"),
        ],
    )
    return SimpleNamespace(dataset=train_dataset), SimpleNamespace(dataset=validation_dataset)


class ProgressiveStageTests(unittest.TestCase):
    def test_exact_trainability_for_all_stages(self) -> None:
        model = TinyEfficientNet()
        expected = {
            "stage1": set(),
            "stage2": {6, 7, 8},
            "stage3": {4, 5, 6, 7, 8},
        }
        for stage, expected_trainable in expected.items():
            configure_progressive_stage(model, stage)
            for index, block in enumerate(model.features):
                self.assertEqual(
                    any(parameter.requires_grad for parameter in block.parameters()),
                    index in expected_trainable,
                )
            self.assertTrue(
                all(parameter.requires_grad for parameter in model.classifier.parameters())
            )

    def test_adamw_state_survives_group_additions_and_schedulers_restart(self) -> None:
        model = TinyEfficientNet()
        configure_progressive_stage(model, "stage1")
        optimizer = create_progressive_optimizer(model)
        classifier_parameter = next(model.classifier.parameters())
        _initialize_optimizer_state(optimizer, list(model.classifier.parameters()))
        classifier_state = _copy_parameter_state(optimizer, classifier_parameter)

        _, _, stage2_scheduler = transition_to_stage2(
            model,
            optimizer,
            stage_epochs=2,
        )
        _assert_parameter_state_equal(self, optimizer, classifier_parameter, classifier_state)
        late_parameter = next(model.features[6].parameters())
        self.assertNotIn(late_parameter, optimizer.state)
        self.assertEqual(stage2_scheduler.T_max, 2)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [DEFAULT_CLASSIFIER_LR, DEFAULT_LATE_BLOCKS_LR],
        )

        active_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        _initialize_optimizer_state(optimizer, active_parameters)
        classifier_state = _copy_parameter_state(optimizer, classifier_parameter)
        late_state = _copy_parameter_state(optimizer, late_parameter)
        stage2_scheduler.step()

        _, _, stage3_scheduler = transition_to_stage3(
            model,
            optimizer,
            stage_epochs=3,
        )
        _assert_parameter_state_equal(self, optimizer, classifier_parameter, classifier_state)
        _assert_parameter_state_equal(self, optimizer, late_parameter, late_state)
        middle_parameter = next(model.features[4].parameters())
        self.assertNotIn(middle_parameter, optimizer.state)
        self.assertIsNot(stage2_scheduler, stage3_scheduler)
        self.assertEqual(stage3_scheduler.T_max, 3)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [
                DEFAULT_CLASSIFIER_LR,
                DEFAULT_LATE_BLOCKS_LR,
                DEFAULT_MIDDLE_BLOCKS_LR,
            ],
        )
        self.assertEqual(
            [group["initial_lr"] for group in optimizer.param_groups],
            [
                DEFAULT_CLASSIFIER_LR,
                DEFAULT_LATE_BLOCKS_LR,
                DEFAULT_MIDDLE_BLOCKS_LR,
            ],
        )

    def test_adamw_configuration_matches_existing_trainer(self) -> None:
        model = TinyEfficientNet()
        configure_progressive_stage(model, "stage1")
        optimizer = create_progressive_optimizer(model)
        reference = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=DEFAULT_CLASSIFIER_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        for key in (
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
        ):
            self.assertEqual(optimizer.defaults.get(key), reference.defaults.get(key))

    def test_frozen_batchnorm_statistics_do_not_change(self) -> None:
        torch.manual_seed(7)
        model = TinyEfficientNet()
        frozen, trainable = configure_progressive_stage(model, "stage2")
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": 1e-4},
                {
                    "params": [
                        parameter
                        for block in trainable
                        for parameter in block.parameters()
                    ],
                    "lr": 1e-5,
                },
            ],
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.tensor([0, 1] * 4)),
            batch_size=4,
        )
        frozen_bn = model.features[0][1]
        trainable_bn = model.features[6][1]
        frozen_mean = frozen_bn.running_mean.clone()
        frozen_var = frozen_bn.running_var.clone()
        frozen_batches = frozen_bn.num_batches_tracked.clone()
        trainable_batches = trainable_bn.num_batches_tracked.clone()

        train_one_epoch(
            model,
            loader,
            nn.BCEWithLogitsLoss(),
            optimizer,
            torch.device("cpu"),
            frozen_modules=frozen,
            trainable_modules=(*trainable, model.classifier),
        )

        self.assertFalse(frozen_bn.training)
        self.assertTrue(trainable_bn.training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(torch.equal(frozen_bn.running_mean, frozen_mean))
        self.assertTrue(torch.equal(frozen_bn.running_var, frozen_var))
        self.assertTrue(torch.equal(frozen_bn.num_batches_tracked, frozen_batches))
        self.assertGreater(trainable_bn.num_batches_tracked.item(), trainable_batches.item())


class ProgressiveValidationAndCheckpointTests(unittest.TestCase):
    def test_holdout_contract_and_user_facing_validation_names(self) -> None:
        train_loader, validation_loader = _source_loaders()
        metadata = verify_heldout_validation_contract(
            train_loader,
            validation_loader,
            "DDPM",
        )
        self.assertNotIn("WildFake DDPM train", metadata["train_fake_sources"])
        self.assertIn(
            "WildFake DDPM held-out validation",
            metadata["heldout_validation_sources"],
        )
        self.assertNotIn("WildFake DDPM test", metadata["heldout_validation_sources"])

    def test_holdout_contract_rejects_training_leakage(self) -> None:
        train_loader, validation_loader = _source_loaders()
        train_loader.dataset.samples.append(
            (Path("leak.jpg"), FAKE_LABEL, "WildFake DDPM train")
        )
        with self.assertRaisesRegex(AssertionError, "contributed training samples"):
            verify_heldout_validation_contract(train_loader, validation_loader, "DDPM")

    def test_aigc_metrics_and_auc_tie_breaker(self) -> None:
        validation = {
            "auc_roc": 0.75,
            "probabilities": [0.1, 0.8, 0.9, 0.4],
            "labels": [0, 0, 1, 1],
            "source_names": [
                "WildFake DDPM test",
                "WildFake DDPM test",
                "CIFAKE test REAL",
                "WildFake COCO test",
            ],
        }
        metrics = calculate_progressive_validation_metrics(validation, "DDPM")
        self.assertAlmostEqual(metrics["heldout_generator_auc_roc"], 0.75)
        self.assertAlmostEqual(metrics["heldout_fake_recall"], 0.5)
        self.assertAlmostEqual(metrics["real_recall"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertTrue(_is_better_candidate(0.8, 0.7, (0.8, 0.6)))
        self.assertFalse(_is_better_candidate(0.8, 0.5, (0.8, 0.6)))
        self.assertFalse(_is_better_candidate(0.79, 1.0, (0.8, 0.0)))

    def test_best_state_is_an_independent_cpu_copy(self) -> None:
        model = TinyEfficientNet()
        snapshot = clone_model_state_to_cpu(model)
        original = snapshot["classifier.1.weight"].clone()
        with torch.no_grad():
            model.classifier[1].weight.add_(10.0)
        self.assertTrue(torch.equal(snapshot["classifier.1.weight"], original))
        self.assertEqual(snapshot["classifier.1.weight"].device.type, "cpu")
        self.assertNotEqual(
            snapshot["classifier.1.weight"].data_ptr(),
            model.classifier[1].weight.data_ptr(),
        )

    def test_exclusive_checkpoint_writer_refuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "new.pt"
            _write_checkpoint_exclusively(checkpoint, {"value": 1})
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _write_checkpoint_exclusively(checkpoint, {"value": 2})
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_training_writes_true_best_state_once_at_completion(self) -> None:
        model = TinyEfficientNet()
        initial_weight = model.classifier[1].weight.detach().clone()
        train_loader, validation_loader = _source_loaders()
        metric_values = iter(
            [
                (0.8, 0.6),
                (0.8, 0.7),
                (0.7, 0.9),
            ]
        )

        def fake_train(*args: object, **kwargs: object) -> tuple[float, float]:
            with torch.no_grad():
                model.classifier[1].weight.add_(1.0)
            args[3].step()
            return 0.2, 0.8

        def fake_metrics(*args: object, **kwargs: object) -> dict[str, object]:
            auc, balanced = next(metric_values)
            return {
                "heldout_generator_auc_roc": auc,
                "heldout_generator_recall": balanced,
                "heldout_fake_recall": balanced,
                "real_recall": balanced,
                "balanced_accuracy": balanced,
                "pooled_validation_auc_roc": auc,
                "source_recalls": {},
                "source_labels": {},
                "macro_source_recall": balanced,
                "overall_accuracy": balanced,
                "overall_auc_roc": auc,
                "heldout_generator": "DDPM",
                "heldout_generator_source": "WildFake DDPM test",
            }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "progressive.pt"
            with (
                patch("src.train_deeper.train_one_epoch", side_effect=fake_train),
                patch(
                    "src.train_deeper.evaluate",
                    return_value={
                        "loss": 0.2,
                        "accuracy": 0.8,
                        "auc_roc": 0.8,
                        "probabilities": [0.1, 0.9],
                        "labels": [0, 1],
                        "source_names": [
                            "WildFake DDPM test",
                            "CIFAKE test REAL",
                        ],
                    },
                ),
                patch(
                    "src.train_deeper.calculate_progressive_validation_metrics",
                    side_effect=fake_metrics,
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = train_progressive_deeper(
                    model,
                    train_loader,
                    validation_loader,
                    torch.device("cpu"),
                    heldout_generator="DDPM",
                    checkpoint_path=checkpoint,
                    run_name="ddpm_progressive",
                    samples_per_epoch=10,
                    seed=42,
                    stage1_epochs=1,
                    stage2_epochs=1,
                    stage3_epochs=1,
                )

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["training_variant"], "progressive_deeper")
            self.assertEqual(saved["heldout_validation_role"], "held-out-generator validation")
            self.assertTrue(
                torch.allclose(
                    saved["model_state_dict"]["classifier.1.weight"],
                    initial_weight + 2.0,
                )
            )
            self.assertTrue(torch.allclose(model.classifier[1].weight, initial_weight + 3.0))
            self.assertEqual(result["checkpoint_path"], checkpoint.resolve())


class ProgressiveCliTests(unittest.TestCase):
    def test_checkpoint_name_uses_only_sanitized_run_name(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-efficientnet-deeper",
                None,
                run_name="DDPM Progressive",
            ),
            Path("checkpoints/efficientnet_deeper_ddpm_progressive_best.pt"),
        )

    def test_dispatch_uses_new_workflow_and_two_stage2_epochs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "new-progressive.pt"
            train_loader, validation_loader = _source_loaders()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "main.py",
                        "train-efficientnet-deeper",
                        "--holdout-fake-source",
                        "ddpm",
                        "--run-name",
                        "ddpm_progressive",
                        "--checkpoint",
                        str(checkpoint),
                    ],
                ),
                patch("main.resolve_wildfake_holdout", return_value="DDPM"),
                patch("main.get_device", return_value=torch.device("cpu")),
                patch("main.download_dataset", return_value=Path("cifake")),
                patch(
                    "main.get_source_balanced_data_loaders",
                    return_value=(train_loader, validation_loader),
                ) as loader_mock,
                patch("main.build_model", return_value=TinyEfficientNet()) as build_mock,
                patch("main.train_progressive_deeper") as train_mock,
                redirect_stdout(io.StringIO()),
            ):
                app.main()

            build_mock.assert_called_once_with(torch.device("cpu"), pretrained=True)
            self.assertEqual(loader_mock.call_args.kwargs["holdout"], "DDPM")
            self.assertEqual(train_mock.call_args.kwargs["stage1_epochs"], 2)
            self.assertEqual(train_mock.call_args.kwargs["stage2_epochs"], 2)
            self.assertEqual(train_mock.call_args.kwargs["stage3_epochs"], 3)
            self.assertEqual(
                train_mock.call_args.kwargs["checkpoint_path"],
                checkpoint,
            )

    def test_existing_output_fails_before_device_dataset_or_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "existing.pt"
            checkpoint.write_bytes(b"old checkpoint")
            stderr = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "main.py",
                        "train-efficientnet-deeper",
                        "--holdout-fake-source",
                        "ddpm",
                        "--run-name",
                        "ddpm_progressive",
                        "--checkpoint",
                        str(checkpoint),
                    ],
                ),
                patch("main.resolve_wildfake_holdout", return_value="DDPM"),
                patch("main.get_device") as device_mock,
                patch("main.download_dataset") as dataset_mock,
                patch("main.train_progressive_deeper") as train_mock,
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit):
                    app.main()

            self.assertIn("Refusing to overwrite", stderr.getvalue())
            self.assertEqual(checkpoint.read_bytes(), b"old checkpoint")
            device_mock.assert_not_called()
            dataset_mock.assert_not_called()
            train_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
