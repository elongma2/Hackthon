from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import main as app
from src.bytedance_validation import run_bytedance_validation
from src.hybrid_model import (
    FREQUENCY_FEATURE_DIM,
    HYBRID_MODEL_TYPE,
    SPATIAL_FEATURE_DIM,
    FFTPreprocessor,
    FrequencyBranch,
    HybridAIGCDetector,
    load_spatial_checkpoint,
)
from src.model import expects_unnormalized_input, load_model
from src.multisource_dataset import FAKE_LABEL, REAL_LABEL, ImageSource, MultiSourceImageDataset
from src.predict import predict_image
from src.robustness import _test_transforms
from src.source_balanced import SourceBalancedBatchSampler
from src.train import _save_staged_checkpoint, configure_head_only, configure_partial_unfreezing
from src.transforms import build_eval_transforms


def write_image(path: Path, color: tuple[int, int, int] = (64, 96, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def efficientnet_state_for(model: HybridAIGCDetector) -> dict[str, torch.Tensor]:
    state = {
        f"features.{key}": value.detach().clone()
        for key, value in model.features.state_dict().items()
    }
    state["classifier.1.weight"] = torch.zeros(1, SPATIAL_FEATURE_DIM)
    state["classifier.1.bias"] = torch.zeros(1)
    return state


class HybridArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HybridAIGCDetector(pretrained_spatial=False).eval()

    def test_feature_and_single_logit_shapes(self) -> None:
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            spatial = self.model.extract_spatial_features(images)
            frequency = self.model.classifier.frequency_branch(images)
            logits = self.model(images)
            one_logit = self.model(images[:1])
        self.assertEqual(spatial.shape, (2, SPATIAL_FEATURE_DIM))
        self.assertEqual(frequency.shape, (2, FREQUENCY_FEATURE_DIM))
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(one_logit.shape, (1, 1))
        labels = torch.tensor([0.0, 1.0]).unsqueeze(1)
        self.assertTrue(torch.isfinite(nn.BCEWithLogitsLoss()(logits, labels)))

    def test_fft_is_finite_float32_for_constant_and_near_constant_inputs(self) -> None:
        preprocessor = FFTPreprocessor()
        for images in (
            torch.zeros(2, 3, 32, 32),
            torch.full((2, 3, 32, 32), 0.5),
            torch.full((2, 3, 32, 32), 0.5) + torch.randn(2, 3, 32, 32) * 1e-7,
        ):
            spectrum = preprocessor(images)
            self.assertEqual(spectrum.dtype, torch.float32)
            self.assertTrue(torch.isfinite(spectrum).all())

    def test_fft_disables_cpu_autocast(self) -> None:
        preprocessor = FFTPreprocessor()
        branch = FrequencyBranch().eval()
        images = torch.rand(2, 3, 32, 32, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            spectrum = preprocessor(images)
            features = branch(images)
        self.assertEqual(spectrum.dtype, torch.float32)
        self.assertTrue(torch.isfinite(spectrum).all())
        self.assertEqual(features.shape, (2, FREQUENCY_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_fft_disables_cuda_float16_autocast(self) -> None:
        preprocessor = FFTPreprocessor().cuda()
        branch = FrequencyBranch().cuda().eval()
        images = torch.rand(2, 3, 32, 32, device="cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            spectrum = preprocessor(images)
            features = branch(images)
        self.assertEqual(spectrum.dtype, torch.float32)
        self.assertTrue(torch.isfinite(spectrum).all())
        self.assertEqual(features.shape, (2, FREQUENCY_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())

    def test_shared_trainer_freezes_spatial_and_trains_complete_custom_head(self) -> None:
        model = HybridAIGCDetector(pretrained_spatial=False)
        frozen = configure_head_only(model)
        self.assertEqual(len(frozen), 9)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("classifier.") for name in trainable_names))
        self.assertTrue(any("frequency_branch" in name for name in trainable_names))
        self.assertTrue(any("fusion" in name for name in trainable_names))

        frozen_blocks, trainable_blocks = configure_partial_unfreezing(model, 3)
        self.assertEqual(len(frozen_blocks), 6)
        self.assertEqual(len(trainable_blocks), 3)
        self.assertTrue(all(not p.requires_grad for block in frozen_blocks for p in block.parameters()))
        self.assertTrue(all(p.requires_grad for block in trainable_blocks for p in block.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_source_balanced_sampler_feeds_raw_hybrid_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources: list[ImageSource] = []
            for label, prefix, count in (
                (FAKE_LABEL, "fake", 4),
                (REAL_LABEL, "real", 3),
            ):
                for index in range(count):
                    source_root = root / f"{prefix}-{index}"
                    write_image(source_root / "nested" / "image.png")
                    sources.append(ImageSource(f"{prefix}-{index}", source_root, label))
            dataset = MultiSourceImageDataset(
                sources,
                transform=build_eval_transforms((32, 32), normalize=False),
            )
            sampler = SourceBalancedBatchSampler(
                dataset,
                batch_size=4,
                samples_per_epoch=4,
                seed=42,
            )
            images, labels = next(iter(DataLoader(dataset, batch_sampler=sampler)))
            self.assertEqual(images.shape, (4, 3, 32, 32))
            self.assertGreaterEqual(float(images.min()), 0.0)
            self.assertLessEqual(float(images.max()), 1.0)
            self.assertEqual(set(labels.tolist()), {FAKE_LABEL, REAL_LABEL})
            with torch.no_grad():
                self.assertEqual(self.model(images).shape, (4, 1))


class HybridCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = HybridAIGCDetector(pretrained_spatial=False)
        self.state = efficientnet_state_for(self.model)

    def test_spatial_loading_derives_exact_runtime_keys_and_reports_count(self) -> None:
        output = io.StringIO()
        with (
            patch("src.hybrid_model.torch.load", return_value={"model_state_dict": self.state}),
            redirect_stdout(output),
        ):
            loaded = load_spatial_checkpoint(
                self.model,
                Path("efficientnet.pt"),
                torch.device("cpu"),
            )
        self.assertEqual(loaded, len(self.model.features.state_dict()))
        self.assertIn(f"Loaded {loaded:,} spatial state entries", output.getvalue())
        self.assertIn("classifier.1.weight", output.getvalue())

    def test_spatial_loading_rejects_missing_unexpected_and_shape_mismatches(self) -> None:
        feature_keys = [key for key in self.state if key.startswith("features.")]
        first_key = feature_keys[0]
        cases: list[tuple[str, dict[str, torch.Tensor], str]] = []

        missing = dict(self.state)
        missing.pop(first_key)
        cases.append(("missing", missing, "do not match exactly"))

        unexpected = dict(self.state)
        unexpected["features.not_a_real_layer"] = torch.zeros(1)
        cases.append(("unexpected", unexpected, "do not match exactly"))

        mismatched = dict(self.state)
        mismatched[first_key] = torch.zeros(1)
        cases.append(("shape", mismatched, "tensor shapes do not match"))

        unsupported = dict(self.state)
        unsupported["module.features.invalid"] = torch.zeros(1)
        cases.append(("prefix", unsupported, "unsupported state prefixes"))

        for name, state, message in cases:
            with self.subTest(name=name):
                before = {
                    key: value.detach().clone()
                    for key, value in self.model.features.state_dict().items()
                }
                with patch("src.hybrid_model.torch.load", return_value=state):
                    with self.assertRaisesRegex(ValueError, message):
                        load_spatial_checkpoint(
                            self.model,
                            Path("invalid.pt"),
                            torch.device("cpu"),
                        )
                for key, value in self.model.features.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)

    def test_hybrid_checkpoint_metadata_and_model_aware_loader(self) -> None:
        metrics = {
            "stage": "head-only",
            "stage_epoch": 1,
            "epoch": 1,
            "validation_loss": 0.5,
            "validation_accuracy": 0.6,
            "validation_auc_roc": 0.7,
            "learning_rates": {"classifier": 1e-4},
        }
        captured: dict[str, object] = {}

        def capture_save(payload: dict[str, object], _: Path) -> None:
            captured.update(payload)

        with patch("src.train.torch.save", side_effect=capture_save):
            _save_staged_checkpoint(
                self.model,
                metrics,
                Path("unused.pt"),
                "validation_auc_roc",
                0.7,
                {"model_type": HYBRID_MODEL_TYPE},
            )
        self.assertEqual(captured["model_type"], HYBRID_MODEL_TYPE)
        with patch("src.model.torch.load", return_value=captured):
            loaded = load_model(Path("hybrid.pt"), torch.device("cpu"))
        self.assertIsInstance(loaded, HybridAIGCDetector)
        self.assertFalse(loaded.training)
        self.assertTrue(expects_unnormalized_input(loaded))


class HybridInferenceAndCliTests(unittest.TestCase):
    def test_prediction_and_robustness_leave_hybrid_inputs_unnormalized(self) -> None:
        class CapturingModel(nn.Module):
            expects_unnormalized_input = True

            def __init__(self) -> None:
                super().__init__()
                self.last_input: torch.Tensor | None = None

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                self.last_input = inputs.detach().clone()
                return torch.zeros(inputs.shape[0], 1)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            write_image(image_path)
            model = CapturingModel()
            result = predict_image(model, image_path, torch.device("cpu"), image_size=(32, 32))
            self.assertEqual(result["label"], "REAL")
            assert model.last_input is not None
            self.assertGreaterEqual(float(model.last_input.min()), 0.0)
            self.assertLessEqual(float(model.last_input.max()), 1.0)
            transformed = _test_transforms((32, 32), normalize_inputs=False)["clean"](
                Image.open(image_path).convert("RGB")
            )
            self.assertGreaterEqual(float(transformed.min()), 0.0)
            self.assertLessEqual(float(transformed.max()), 1.0)

    def test_bytedance_validation_leaves_hybrid_inputs_unnormalized(self) -> None:
        class CapturingModel(nn.Module):
            expects_unnormalized_input = True

            def __init__(self) -> None:
                super().__init__()
                self.last_input: torch.Tensor | None = None

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                self.last_input = inputs.detach().clone()
                return torch.zeros(inputs.shape[0], 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cifake = root / "cifake"
            validation = root / "validation"
            for split in ("train", "test"):
                write_image(cifake / split / "FAKE" / "fake.png")
                write_image(cifake / split / "REAL" / "real.png")
            write_image(validation / "FAKE" / "fake.png")
            write_image(validation / "REAL" / "real.png")
            model = CapturingModel()

            with (
                patch("src.bytedance_validation.download_dataset", return_value=cifake),
                patch("src.bytedance_validation.load_model", return_value=model),
                redirect_stdout(io.StringIO()),
            ):
                result = run_bytedance_validation(
                    checkpoint_path=root / "hybrid.pt",
                    validation_dir=validation,
                    data_dir=root,
                    device=torch.device("cpu"),
                    batch_size=2,
                    image_size=(32, 32),
                    num_workers=0,
                )

            self.assertEqual(result["total"], 2)
            self.assertIsNotNone(model.last_input)
            assert model.last_input is not None
            self.assertGreaterEqual(float(model.last_input.min()), 0.0)
            self.assertLessEqual(float(model.last_input.max()), 1.0)

    def test_hybrid_checkpoint_defaults(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid", None),
            app.DEFAULT_ALL_SOURCE_HYBRID_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid", None, "Midjourney v6"),
            Path("checkpoints/hybrid_balanced_holdout_midjourney_v6_best.pt"),
        )

    def test_train_hybrid_dispatch_reuses_source_balanced_loader_and_shared_trainer(self) -> None:
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE train FAKE")])
        )
        validation_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE test REAL")])
        )
        with (
            patch.object(
                __import__("sys"),
                "argv",
                [
                    "main.py",
                    "train-hybrid",
                    "--spatial-checkpoint",
                    "checkpoints/spatial.pt",
                    "--samples-per-epoch",
                    "64",
                ],
            ),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_hybrid_model", return_value=Mock()) as build_mock,
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertIs(loader_mock.call_args.kwargs["normalize_inputs"], False)
        build_mock.assert_called_once_with(
            torch.device("cpu"),
            spatial_checkpoint=Path("checkpoints/spatial.pt"),
        )
        self.assertEqual(train_mock.call_args.kwargs["stage1_classifier_learning_rate"], 1e-4)
        self.assertEqual(train_mock.call_args.kwargs["stage2_backbone_learning_rate"], 1e-5)
        self.assertEqual(
            train_mock.call_args.kwargs["checkpoint_metadata"]["model_type"],
            HYBRID_MODEL_TYPE,
        )

    def test_train_hybrid_dispatch_preserves_arbitrary_holdout_semantics(self) -> None:
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE train FAKE")])
        )
        validation_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="WildFake Midjourney test")])
        )
        with (
            patch.object(
                __import__("sys"),
                "argv",
                ["main.py", "train-hybrid", "--holdout", "midjourney"],
            ),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.resolve_wildfake_holdout", return_value="Midjourney"),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_hybrid_model", return_value=Mock()),
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertEqual(loader_mock.call_args.kwargs["holdout"], "Midjourney")
        self.assertEqual(
            train_mock.call_args.kwargs["heldout_generator"],
            "Midjourney",
        )
        self.assertEqual(
            train_mock.call_args.kwargs["checkpoint_path"],
            Path("checkpoints/hybrid_balanced_holdout_midjourney_best.pt"),
        )


if __name__ == "__main__":
    unittest.main()
