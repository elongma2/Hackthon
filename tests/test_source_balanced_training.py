from __future__ import annotations

import io
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import main as app
from src.multisource_dataset import (
    FAKE_LABEL,
    REAL_LABEL,
    ImageSource,
    MultiSourceImageDataset,
)
from src.source_balanced import (
    SourceBalancedBatchSampler,
    build_heldout_sources,
    calculate_source_metrics,
    get_source_balanced_data_loaders,
)
from src.train import train_staged_model


def write_images(root: Path, count: int) -> None:
    for index in range(count):
        path = root / "nested" / f"image_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index % 255, 20, 30)).save(path)


def create_six_source_dataset(root: Path) -> MultiSourceImageDataset:
    definitions = (
        ("CIFAKE train FAKE", FAKE_LABEL, 1),
        ("WildFake ADM train", FAKE_LABEL, 2),
        ("WildFake DDPM train", FAKE_LABEL, 9),
        ("CIFAKE train REAL", REAL_LABEL, 1),
        ("WildFake COCO train", REAL_LABEL, 3),
        ("WildFake LAION-5B train", REAL_LABEL, 7),
    )
    sources = []
    for name, label, count in definitions:
        source_root = root / name.replace(" ", "_")
        write_images(source_root, count)
        sources.append(ImageSource(name, source_root, label))
    return MultiSourceImageDataset(sources)


def create_multisource_fixture(root: Path) -> tuple[Path, Path]:
    cifake = root / "cifake"
    wildfake = root / "WildFake"
    paths = (
        cifake / "train" / "FAKE",
        cifake / "train" / "REAL",
        cifake / "test" / "FAKE",
        cifake / "test" / "REAL",
        wildfake / "FAKE" / "ADM" / "TRAIN",
        wildfake / "FAKE" / "ADM" / "TEST",
        wildfake / "FAKE" / "DDPM" / "TRAIN",
        wildfake / "FAKE" / "DDPM" / "TEST",
        wildfake / "REAL" / "cocofolder" / "TRAIN",
        wildfake / "REAL" / "cocofolder" / "TEST",
        wildfake / "REAL" / "laion5b" / "TRAIN",
        wildfake / "REAL" / "laion5b" / "TEST",
    )
    for path in paths:
        write_images(path, 1)
    return cifake, wildfake


class TinyEfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(9)]
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(4, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class SourceBalancedSamplerTests(unittest.TestCase):
    def test_balances_classes_and_all_sources_while_resampling_small_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = create_six_source_dataset(Path(directory))
            sampler = SourceBalancedBatchSampler(
                dataset,
                batch_size=32,
                samples_per_epoch=96,
                seed=42,
            )
            sampled = [index for batch in sampler for index in batch]
            labels = Counter(dataset.samples[index][1] for index in sampled)
            sources = Counter(dataset.samples[index][2] for index in sampled)

            self.assertEqual(labels, {FAKE_LABEL: 48, REAL_LABEL: 48})
            self.assertTrue(all(count == 16 for count in sources.values()))
            one_image_source_indices = [
                index
                for index, sample in enumerate(dataset.samples)
                if sample[2] == "CIFAKE train FAKE"
            ]
            self.assertEqual(len(one_image_source_indices), 1)
            self.assertEqual(sampled.count(one_image_source_indices[0]), 16)

    def test_is_deterministic_between_runs_and_changes_by_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = create_six_source_dataset(Path(directory))
            first = SourceBalancedBatchSampler(dataset, 12, 48, 42)
            second = SourceBalancedBatchSampler(dataset, 12, 48, 42)

            first_epoch = list(first)
            second_epoch = list(first)
            self.assertEqual(first_epoch, list(second))
            self.assertEqual(second_epoch, list(second))
            self.assertNotEqual(first_epoch, second_epoch)

    def test_samples_per_epoch_controls_exact_output_and_final_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = create_six_source_dataset(Path(directory))
            sampler = SourceBalancedBatchSampler(dataset, 32, 35, 42)
            batches = list(sampler)

            self.assertEqual(len(sampler), 2)
            self.assertEqual([len(batch) for batch in batches], [32, 3])
            self.assertEqual(sum(map(len, batches)), 35)

    def test_balances_an_arbitrary_number_of_sources_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definitions = [
                ("fake-a", FAKE_LABEL),
                ("fake-b", FAKE_LABEL),
                ("fake-c", FAKE_LABEL),
                ("fake-d", FAKE_LABEL),
                ("real-a", REAL_LABEL),
                ("real-b", REAL_LABEL),
            ]
            sources: list[ImageSource] = []
            for name, label in definitions:
                source_root = root / name
                write_images(source_root, 2)
                sources.append(ImageSource(name, source_root, label))
            dataset = MultiSourceImageDataset(sources)
            sampler = SourceBalancedBatchSampler(dataset, 24, 96, 42)
            sampled = [index for batch in sampler for index in batch]
            labels = Counter(dataset.samples[index][1] for index in sampled)
            source_counts = Counter(dataset.samples[index][2] for index in sampled)

            self.assertEqual(labels, {FAKE_LABEL: 48, REAL_LABEL: 48})
            self.assertTrue(all(source_counts[name] == 12 for name, _ in definitions[:4]))
            self.assertTrue(all(source_counts[name] == 24 for name, _ in definitions[4:]))


class HeldOutDatasetTests(unittest.TestCase):
    def test_ddpm_and_adm_holdouts_never_enter_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake, wildfake = create_multisource_fixture(Path(directory))
            for holdout, retained in (("DDPM", "ADM"), ("ADM", "DDPM")):
                training, validation = build_heldout_sources(cifake, wildfake, holdout)
                training_names = {source.name for source in training}
                validation_names = {source.name for source in validation}

                self.assertNotIn(f"WildFake {holdout} train", training_names)
                self.assertIn(f"WildFake {retained} train", training_names)
                self.assertEqual(
                    validation_names,
                    {
                        f"WildFake {holdout} test",
                        "CIFAKE test REAL",
                        "WildFake COCO test",
                        "WildFake LAION-5B test",
                    },
                )
                self.assertTrue(all("validation" not in str(source.root) for source in training))

    def test_loader_uses_balanced_batches_and_sequential_complete_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake, wildfake = create_multisource_fixture(Path(directory))
            with redirect_stdout(io.StringIO()):
                training, validation = get_source_balanced_data_loaders(
                    cifake,
                    wildfake,
                    "DDPM",
                    batch_size=4,
                    samples_per_epoch=10,
                    seed=42,
                    image_size=(8, 8),
                    num_workers=0,
                )

            self.assertIsInstance(training.batch_sampler, SourceBalancedBatchSampler)
            self.assertEqual(sum(len(batch) for batch in training.batch_sampler), 10)
            self.assertIsInstance(validation.sampler, SequentialSampler)
            self.assertEqual(len(validation.dataset), 4)
            self.assertTrue(validation.dataset.return_source)

    def test_arbitrary_fake_holdout_is_excluded_and_uses_all_real_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake, wildfake = create_multisource_fixture(Path(directory))
            for split in ("TRAIN", "TEST"):
                write_images(wildfake / "FAKE" / "Midjourney" / split, 1)
                write_images(wildfake / "REAL" / "Flickr" / split, 1)

            training, validation = build_heldout_sources(
                cifake,
                wildfake,
                "midJOURNEY",
            )
            training_names = {source.name for source in training}
            validation_names = {source.name for source in validation}

            self.assertNotIn("WildFake Midjourney train", training_names)
            self.assertIn("WildFake ADM train", training_names)
            self.assertIn("WildFake DDPM train", training_names)
            self.assertEqual(
                validation_names,
                {
                    "WildFake Midjourney test",
                    "CIFAKE test REAL",
                    "WildFake COCO test",
                    "WildFake Flickr test",
                    "WildFake LAION-5B test",
                },
            )

    def test_no_holdout_balances_every_discovered_training_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake, wildfake = create_multisource_fixture(Path(directory))
            for split in ("TRAIN", "TEST"):
                write_images(wildfake / "FAKE" / "Midjourney" / split, 1)
                write_images(wildfake / "REAL" / "Flickr" / split, 1)

            with redirect_stdout(io.StringIO()):
                training, validation = get_source_balanced_data_loaders(
                    cifake,
                    wildfake,
                    batch_size=8,
                    samples_per_epoch=16,
                    seed=42,
                    image_size=(8, 8),
                    num_workers=0,
                )

            training_names = {source.name for source in training.dataset.sources}
            validation_names = {source.name for source in validation.dataset.sources}
            self.assertEqual(
                training_names,
                {
                    "CIFAKE train FAKE",
                    "WildFake ADM train",
                    "WildFake DDPM train",
                    "WildFake Midjourney train",
                    "CIFAKE train REAL",
                    "WildFake COCO train",
                    "WildFake Flickr train",
                    "WildFake LAION-5B train",
                },
            )
            self.assertEqual(len(validation_names), 8)
            self.assertIn("WildFake Midjourney test", validation_names)
            self.assertIn("WildFake Flickr test", validation_names)
            self.assertIsInstance(training.batch_sampler, SourceBalancedBatchSampler)
            self.assertFalse(validation.dataset.return_source)


class SourceMetricTests(unittest.TestCase):
    def test_calculates_per_source_and_fake_positive_holdout_metrics(self) -> None:
        metrics = calculate_source_metrics(
            raw_probabilities=[0.1, 0.8, 0.9, 0.4, 0.8],
            labels=[0, 0, 1, 1, 1],
            source_names=[
                "WildFake DDPM test",
                "WildFake DDPM test",
                "CIFAKE test REAL",
                "WildFake COCO test",
                "WildFake LAION-5B test",
            ],
            holdout="DDPM",
        )

        self.assertEqual(metrics["heldout_generator_recall"], 0.5)
        self.assertEqual(metrics["source_recalls"]["CIFAKE test REAL"], 1.0)
        self.assertEqual(metrics["source_recalls"]["WildFake COCO test"], 0.0)
        self.assertEqual(metrics["source_recalls"]["WildFake LAION-5B test"], 1.0)
        self.assertAlmostEqual(metrics["macro_source_recall"], 0.625)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.6)
        self.assertAlmostEqual(metrics["overall_auc_roc"], 0.75)
        self.assertAlmostEqual(metrics["heldout_generator_auc_roc"], 0.75)

    def test_checkpoint_selection_uses_heldout_auc_instead_of_pooled_auc(self) -> None:
        heldout_values = iter([0.20, 0.80, 0.50])
        pooled_values = iter([0.95, 0.70, 0.60])

        def fake_train(*args: object, **kwargs: object) -> tuple[float, float]:
            optimizer = args[3]
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return 0.2, 0.8

        def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
            return {
                "loss": 0.2,
                "accuracy": 0.8,
                "auc_roc": next(pooled_values),
                "probabilities": [0.1, 0.9],
                "labels": [0, 1],
                "source_names": ["WildFake DDPM test", "CIFAKE test REAL"],
            }

        def fake_source_metrics(*args: object, **kwargs: object) -> dict[str, object]:
            value = next(heldout_values)
            return {
                "source_recalls": {"WildFake DDPM test": 0.5, "CIFAKE test REAL": 1.0},
                "macro_source_recall": 0.75,
                "overall_accuracy": 0.75,
                "overall_auc_roc": 0.5,
                "heldout_generator": "DDPM",
                "heldout_generator_source": "WildFake DDPM test",
                "heldout_generator_recall": 0.5,
                "heldout_generator_auc_roc": value,
            }

        loader = DataLoader(
            TensorDataset(torch.randn(2, 4), torch.tensor([0, 1])),
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "balanced.pt"
            with (
                patch("src.train.train_one_epoch", side_effect=fake_train),
                patch("src.train.evaluate", side_effect=fake_evaluate),
                patch("src.train.calculate_source_metrics", side_effect=fake_source_metrics),
                patch("src.train.print_source_metrics"),
            ):
                result = train_staged_model(
                    TinyEfficientNet(),
                    loader,
                    loader,
                    torch.device("cpu"),
                    stage2_epochs=1,
                    checkpoint_path=checkpoint,
                    heldout_generator="DDPM",
                )

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["selection_metric"], "heldout_generator_auc_roc")
            self.assertEqual(saved["best_heldout_generator_auc_roc"], 0.8)
            self.assertEqual(result["best_heldout_generator_auc_roc"], 0.8)


class SourceBalancedCliTests(unittest.TestCase):
    def test_checkpoint_defaults_and_case_insensitive_holdout(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-source-balanced", None),
            app.DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-source-balanced", None, "ddpm"),
            Path("checkpoints/efficientnet_balanced_holdout_ddpm_best.pt"),
        )
        custom = Path("checkpoints/custom.pt")
        self.assertEqual(
            app.resolve_checkpoint_path("train-source-balanced", custom, "ADM"),
            custom,
        )
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-source-balanced",
                None,
                "Midjourney v6",
            ),
            Path("checkpoints/efficientnet_balanced_holdout_midjourney_v6_best.pt"),
        )

    def test_dispatch_never_calls_bytedance_or_robustness(self) -> None:
        model = Mock(name="model")
        train_loader = Mock(name="train_loader")
        validation_loader = Mock(name="validation_loader")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "train-source-balanced",
                    "--holdout",
                    "ddpm",
                    "--samples-per-epoch",
                    "64",
                ],
            ),
            patch("main.resolve_wildfake_holdout", return_value="DDPM") as resolve_mock,
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_model", return_value=model),
            patch("main.train_staged_model") as train_mock,
            patch("main.run_bytedance_validation") as bytedance_mock,
            patch("main.run_robustness_benchmark") as robustness_mock,
            patch("main.load_model") as load_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertEqual(loader_mock.call_args.kwargs["holdout"], "DDPM")
        resolve_mock.assert_called_once_with(Path("data/raw") / "WildFake", "ddpm")
        self.assertEqual(loader_mock.call_args.kwargs["samples_per_epoch"], 64)
        self.assertEqual(loader_mock.call_args.kwargs["seed"], 42)
        self.assertEqual(train_mock.call_args.kwargs["heldout_generator"], "DDPM")
        self.assertEqual(
            train_mock.call_args.kwargs["checkpoint_path"],
            Path("checkpoints/efficientnet_balanced_holdout_ddpm_best.pt"),
        )
        bytedance_mock.assert_not_called()
        robustness_mock.assert_not_called()
        load_mock.assert_not_called()

    def test_no_holdout_dispatches_all_source_balanced_training(self) -> None:
        with (
            patch.object(sys, "argv", ["main.py", "train-source-balanced"]),
            patch("main.resolve_wildfake_holdout") as resolve_mock,
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(Mock(), Mock()),
            ) as loader_mock,
            patch("main.build_model", return_value=Mock()),
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        resolve_mock.assert_not_called()
        self.assertIsNone(loader_mock.call_args.kwargs["holdout"])
        self.assertIsNone(train_mock.call_args.kwargs["heldout_generator"])
        self.assertEqual(
            train_mock.call_args.kwargs["checkpoint_path"],
            app.DEFAULT_ALL_SOURCE_BALANCED_CHECKPOINT,
        )


if __name__ == "__main__":
    unittest.main()
