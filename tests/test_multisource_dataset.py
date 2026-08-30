from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.multisource_dataset import (
    FAKE_LABEL,
    REAL_LABEL,
    ImageSource,
    MultiSourceImageDataset,
    build_multisource_sources,
    discover_wildfake_sources,
    get_multisource_data_loaders,
    resolve_wildfake_holdout,
)


def write_image(
    path: Path,
    mode: str = "RGB",
    color: tuple[int, ...] = (10, 20, 30),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (8, 8), color=color).save(path)


def create_multisource_fixture(root: Path) -> tuple[Path, Path]:
    cifake_root = root / "cifake"
    wildfake_root = root / "WildFake"
    write_image(cifake_root / "train" / "FAKE" / "fake.jpg")
    write_image(cifake_root / "train" / "REAL" / "real.jpeg")
    write_image(cifake_root / "test" / "FAKE" / "fake.png")
    write_image(cifake_root / "test" / "REAL" / "real.webp")

    write_image(
        wildfake_root / "FAKE" / "ADM" / "TRAIN" / "nested" / "adm.png",
        mode="RGBA",
        color=(10, 20, 30, 40),
    )
    write_image(wildfake_root / "FAKE" / "DDPM" / "TRAIN" / "ddpm.png")
    write_image(wildfake_root / "REAL" / "cocofolder" / "TRAIN" / "coco.jpg")
    write_image(wildfake_root / "REAL" / "laion5b" / "TRAIN" / "laion.jpg")
    write_image(wildfake_root / "FAKE" / "ADM" / "TEST" / "adm.png")
    write_image(wildfake_root / "FAKE" / "DDPM" / "TEST" / "ddpm.png")
    write_image(wildfake_root / "REAL" / "cocofolder" / "TEST" / "coco.jpg")
    write_image(wildfake_root / "REAL" / "laion5b" / "TEST" / "laion.jpg")
    (wildfake_root / "FAKE" / "ADM" / "TRAIN" / "ignored.txt").write_text(
        "not an image",
        encoding="utf-8",
    )
    return cifake_root, wildfake_root


class MultiSourceDatasetTests(unittest.TestCase):
    def test_recursive_discovery_explicit_labels_and_rgb_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_root = root / "alphabetically-real-looking"
            real_root = root / "alphabetically-fake-looking"
            write_image(
                fake_root / "nested" / "fake.png",
                mode="RGBA",
                color=(10, 20, 30, 40),
            )
            write_image(real_root / "real.webp")
            (fake_root / "ignored.txt").write_text("ignore", encoding="utf-8")

            def rgb_tensor(image: Image.Image) -> torch.Tensor:
                self.assertEqual(image.mode, "RGB")
                return torch.zeros(3, image.height, image.width)

            dataset = MultiSourceImageDataset(
                [
                    ImageSource("explicit fake", fake_root, FAKE_LABEL),
                    ImageSource("explicit real", real_root, REAL_LABEL),
                ],
                transform=rgb_tensor,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.source_counts, {"explicit fake": 1, "explicit real": 1})
            self.assertEqual(dataset.class_counts[FAKE_LABEL], 1)
            self.assertEqual(dataset.class_counts[REAL_LABEL], 1)
            self.assertEqual([sample[1] for sample in dataset.samples], [FAKE_LABEL, REAL_LABEL])
            tensor, label = dataset[0]
            self.assertEqual(tensor.shape, (3, 8, 8))
            self.assertEqual(label, FAKE_LABEL)
            self.assertIn("nested", str(dataset.samples[0][0]))

    def test_missing_and_empty_sources_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                MultiSourceImageDataset(
                    [ImageSource("missing", root / "missing", FAKE_LABEL)]
                )

            empty_root = root / "empty"
            empty_root.mkdir()
            with self.assertRaises(ValueError):
                MultiSourceImageDataset(
                    [ImageSource("empty", empty_root, REAL_LABEL)]
                )

            write_image(empty_root / "one.jpg")
            with self.assertRaises(ValueError):
                MultiSourceImageDataset(
                    [ImageSource("invalid fraction", empty_root, REAL_LABEL)],
                    source_fraction=0.0,
                )

    def test_fraction_is_applied_per_source_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first"
            second_root = root / "second"
            for index in range(10):
                write_image(first_root / f"first_{index}.jpg")
            for index in range(7):
                write_image(second_root / f"second_{index}.png")
            sources = [
                ImageSource("first source", first_root, FAKE_LABEL),
                ImageSource("second source", second_root, REAL_LABEL),
            ]

            first_run = MultiSourceImageDataset(
                sources,
                source_fraction=0.5,
                sampling_seed=42,
            )
            second_run = MultiSourceImageDataset(
                sources,
                source_fraction=0.5,
                sampling_seed=42,
            )

            self.assertEqual(
                first_run.source_original_counts,
                {"first source": 10, "second source": 7},
            )
            self.assertEqual(
                first_run.source_counts,
                {"first source": 5, "second source": 3},
            )
            self.assertEqual(
                [sample[0] for sample in first_run.samples],
                [sample[0] for sample in second_run.samples],
            )
            self.assertEqual(first_run.class_counts[FAKE_LABEL], 5)
            self.assertEqual(first_run.class_counts[REAL_LABEL], 3)

    def test_loader_sources_counts_and_shuffle_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake_root, wildfake_root = create_multisource_fixture(Path(directory))
            output = io.StringIO()
            with redirect_stdout(output):
                training_loader, validation_loader = get_multisource_data_loaders(
                    cifake_root,
                    wildfake_root,
                    batch_size=6,
                    image_size=(8, 8),
                    num_workers=0,
                )

            self.assertIsInstance(training_loader.sampler, RandomSampler)
            self.assertIsInstance(validation_loader.sampler, SequentialSampler)
            self.assertEqual(len(training_loader.dataset), 6)
            self.assertEqual(len(validation_loader.dataset), 6)
            self.assertEqual(training_loader.dataset.class_counts[FAKE_LABEL], 3)
            self.assertEqual(training_loader.dataset.class_counts[REAL_LABEL], 3)
            self.assertEqual(validation_loader.dataset.class_counts[FAKE_LABEL], 3)
            self.assertEqual(validation_loader.dataset.class_counts[REAL_LABEL], 3)

            shuffled_indices = list(iter(training_loader.sampler))
            first_batch_sources = {
                training_loader.dataset.samples[index][2]
                for index in shuffled_indices[: training_loader.batch_size]
            }
            self.assertGreater(len(first_batch_sources), 1)
            self.assertIn("Training totals: FAKE=3, REAL=3, total=6", output.getvalue())
            self.assertIn(
                "Internal validation totals: FAKE=3, REAL=3, total=6",
                output.getvalue(),
            )

    def test_training_fraction_does_not_reduce_internal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake_root, wildfake_root = create_multisource_fixture(Path(directory))
            training_sources, _ = build_multisource_sources(cifake_root, wildfake_root)
            for source in training_sources:
                write_image(source.root / "second.jpg")

            output = io.StringIO()
            with redirect_stdout(output):
                training_loader, validation_loader = get_multisource_data_loaders(
                    cifake_root,
                    wildfake_root,
                    batch_size=2,
                    image_size=(8, 8),
                    num_workers=0,
                    train_fraction=0.5,
                )

            self.assertEqual(len(training_loader.dataset), 6)
            self.assertEqual(len(validation_loader.dataset), 6)
            self.assertTrue(
                all(count == 2 for count in training_loader.dataset.source_original_counts.values())
            )
            self.assertTrue(
                all(count == 1 for count in training_loader.dataset.source_counts.values())
            )
            self.assertTrue(
                all(count == 1 for count in validation_loader.dataset.source_counts.values())
            )
            self.assertIn("2 -> 1 images", output.getvalue())

    def test_source_configuration_never_includes_external_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake_root, wildfake_root = create_multisource_fixture(Path(directory))
            training_sources, validation_sources = build_multisource_sources(
                cifake_root,
                wildfake_root,
            )

            configured_roots = [
                str(source.root.resolve()).lower()
                for source in training_sources + validation_sources
            ]
            self.assertTrue(all("validation" not in root for root in configured_roots))
            self.assertTrue(all(source.label in (FAKE_LABEL, REAL_LABEL) for source in training_sources))

    def test_dynamic_sources_enter_multisource_training_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cifake_root, wildfake_root = create_multisource_fixture(Path(directory))
            for source_name in ("Midjourney", "StableDiffusion"):
                write_image(wildfake_root / "FAKE" / source_name / "TRAIN" / "train.jpg")
                write_image(wildfake_root / "FAKE" / source_name / "TEST" / "test.jpg")
            write_image(wildfake_root / "REAL" / "Flickr" / "TRAIN" / "train.jpg")
            write_image(wildfake_root / "REAL" / "Flickr" / "TEST" / "test.jpg")
            write_image(wildfake_root / "FAKE" / "Broken" / "TRAIN" / "only.jpg")

            output = io.StringIO()
            with redirect_stdout(output):
                discovered = discover_wildfake_sources(wildfake_root)
                training, validation = build_multisource_sources(cifake_root, wildfake_root)

            discovered_names = [source.name for source in discovered]
            self.assertEqual(
                discovered_names,
                ["ADM", "DDPM", "Midjourney", "StableDiffusion", "cocofolder", "Flickr", "laion5b"],
            )
            training_names = {source.name for source in training}
            validation_names = {source.name for source in validation}
            self.assertIn("WildFake Midjourney train", training_names)
            self.assertIn("WildFake StableDiffusion train", training_names)
            self.assertIn("WildFake Flickr train", training_names)
            self.assertIn("WildFake Midjourney test", validation_names)
            self.assertNotIn("WildFake Broken train", training_names)
            self.assertIn("Skipping WildFake FAKE source 'Broken'", output.getvalue())

    def test_arbitrary_holdout_resolution_is_case_insensitive_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, wildfake_root = create_multisource_fixture(Path(directory))
            write_image(wildfake_root / "FAKE" / "Midjourney" / "TRAIN" / "train.jpg")
            write_image(wildfake_root / "FAKE" / "Midjourney" / "TEST" / "test.jpg")
            write_image(wildfake_root / "FAKE" / "Incomplete" / "TRAIN" / "only.jpg")

            self.assertEqual(
                resolve_wildfake_holdout(wildfake_root, "midJOURNEY"),
                "Midjourney",
            )
            with self.assertRaisesRegex(ValueError, "needs nonempty TRAIN and TEST"):
                resolve_wildfake_holdout(wildfake_root, "Incomplete")
            with self.assertRaisesRegex(ValueError, "REAL source"):
                resolve_wildfake_holdout(wildfake_root, "laion5b")
            with self.assertRaisesRegex(ValueError, "Available FAKE sources"):
                resolve_wildfake_holdout(wildfake_root, "DoesNotExist")


if __name__ == "__main__":
    unittest.main()
