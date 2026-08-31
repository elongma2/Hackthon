from __future__ import annotations

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
from src.evaluate import evaluate
from src.hybrid_v31_model import (
    DEFAULT_RADIAL_BINS,
    HYBRID_V31_MODEL_TYPE,
    V31_MAGNITUDE_FEATURE_DIM,
    V31_PHASE_FEATURE_DIM,
    V31_SPATIAL_FEATURE_DIM,
    HybridV31AIGCDetector,
    LearnedFrequencyResidualClassifier,
    RadialProfileExtractor,
    configure_hybrid_v31_stage,
    v31_epoch_metadata,
    v31_validation_forward,
)
from src.model import expects_unnormalized_input, load_model
from src.train import train_staged_model


def v31_checkpoint(model: HybridV31AIGCDetector) -> dict[str, object]:
    weights = model.classifier.normalized_frequency_weights().detach().cpu()
    return {
        "model_type": HYBRID_V31_MODEL_TYPE,
        "model_state_dict": model.state_dict(),
        "spatial_feature_dim": V31_SPATIAL_FEATURE_DIM,
        "magnitude_feature_dim": V31_MAGNITUDE_FEATURE_DIM,
        "phase_feature_dim": V31_PHASE_FEATURE_DIM,
        "frequency_hidden_dim": 64,
        "frequency_dropout": 0.5,
        "radial_bins": model.radial_bins,
        "radial_hidden_dim": 64,
        "radial_dropout": 0.3,
        "frequency_scale": model.frequency_scale,
        "frequency_branch_dropout": model.frequency_branch_dropout,
        "frequency_mask_prob": model.frequency_mask_probability,
        "frequency_fusion_type": "learned_softmax",
        "fft_normalization": "ortho",
        "phase_representation": "sin_cos",
        "spatial_classifier_loaded": True,
        "spatial_classifier_source": "EfficientNet classifier.1",
        "learned_frequency_weights": {
            "magnitude": float(weights[0]),
            "phase": float(weights[1]),
            "radial": float(weights[2]),
        },
    }


class RadialProfileTests(unittest.TestCase):
    def test_shapes_and_finite_values_for_varied_inputs(self) -> None:
        extractor = RadialProfileExtractor(32)
        inputs = (
            torch.rand(2, 1, 33, 47),
            torch.zeros(1, 1, 8, 8),
            torch.full((1, 1, 9, 7), 0.5) + torch.rand(1, 1, 9, 7) * 1e-7,
            torch.ones(1, 1, 2, 2),
        )
        for spectrum in inputs:
            with self.subTest(shape=tuple(spectrum.shape)):
                output = extractor(spectrum)
                self.assertEqual(output.shape, (spectrum.shape[0], 32))
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())

    def test_bins_are_ordered_from_low_to_high_frequency(self) -> None:
        size = 33
        rows = torch.arange(size, dtype=torch.float32)
        columns = torch.arange(size, dtype=torch.float32)
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        radius = torch.sqrt(
            (row_grid - float(size // 2)).square()
            + (column_grid - float(size // 2)).square()
        )
        profile = RadialProfileExtractor(8)(radius.view(1, 1, size, size))[0]
        self.assertTrue(torch.all(profile[1:] >= profile[:-1]))

    def test_cache_key_separates_all_required_dimensions(self) -> None:
        extractor32 = RadialProfileExtractor(32)
        extractor16 = RadialProfileExtractor(16)
        cpu_key = extractor32.cache_key(224, 224, torch.device("cpu"))
        resolution_key = extractor32.cache_key(128, 224, torch.device("cpu"))
        bins_key = extractor16.cache_key(224, 224, torch.device("cpu"))
        cuda0_key = extractor32.cache_key(224, 224, torch.device("cuda:0"))
        cuda1_key = extractor32.cache_key(224, 224, torch.device("cuda:1"))
        self.assertEqual(len({cpu_key, resolution_key, bins_key, cuda0_key, cuda1_key}), 5)

        spectrum = torch.ones(1, 1, 8, 8)
        extractor32(spectrum)
        first_geometry = next(iter(extractor32._bin_cache.values()))
        extractor32(spectrum)
        second_geometry = next(iter(extractor32._bin_cache.values()))
        self.assertEqual(len(extractor32._bin_cache), 1)
        self.assertIs(first_geometry, second_geometry)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_radial_output_is_float32_and_finite(self) -> None:
        output = RadialProfileExtractor(32).cuda()(
            torch.rand(2, 1, 32, 32, device="cuda", dtype=torch.float16)
        )
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())


class HybridV31ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HybridV31AIGCDetector(pretrained_spatial=False).eval()

    def test_component_shapes_and_single_fft(self) -> None:
        images = torch.rand(2, 3, 64, 64)
        original_fft2 = torch.fft.fft2
        with patch("src.hybrid_v3_model.torch.fft.fft2", wraps=original_fft2) as fft2:
            with torch.no_grad():
                output, diagnostics = self.model.forward_with_branch_logits(images)
        self.assertEqual(fft2.call_count, 1)
        components = tuple(diagnostics.values())
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(components[0].shape, (2, 1))
        self.assertEqual(components[1].shape, (2, 1))
        self.assertEqual(components[2].shape, (2, 1))
        self.assertEqual(components[3].shape, (2, 1))
        with torch.no_grad():
            ordinary_output = self.model(images)
            self.assertEqual(self.model(images[:1]).shape, (1, 1))
            spatial = self.model.extract_spatial_features(images)
            magnitude, phase = self.model.classifier.fft_preprocessor(images)
            radial = self.model.classifier.radial_extractor(magnitude)
        self.assertEqual(spatial.shape, (2, 1280))
        self.assertEqual(phase.shape, (2, 2, 64, 64))
        self.assertEqual(radial.shape, (2, DEFAULT_RADIAL_BINS))
        self.assertTrue(torch.equal(output, ordinary_output))

    def test_radial_values_are_extracted_before_magnitude_masking(self) -> None:
        classifier = LearnedFrequencyResidualClassifier(
            spatial_dim=4,
            mask_probability=1.0,
            branch_dropout=0.0,
        ).train()
        magnitude = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
        phase = torch.zeros(1, 2, 8, 8)
        events: list[str] = []
        observed: dict[str, torch.Tensor] = {}

        def fake_fft(_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return magnitude, phase

        original_radial = classifier.radial_extractor.forward

        def capture_radial(value: torch.Tensor) -> torch.Tensor:
            events.append("radial")
            observed["radial_input"] = value.detach().clone()
            return original_radial(value)

        def mask_to_zero(value: torch.Tensor) -> torch.Tensor:
            events.append("mask")
            observed["mask_input"] = value.detach().clone()
            return torch.zeros_like(value)

        def capture_cnn(
            _: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            observed["cnn_input"] = inputs[0].detach().clone()

        handle = classifier.magnitude_branch.cnn.register_forward_pre_hook(capture_cnn)
        try:
            with (
                patch.object(classifier.fft_preprocessor, "forward", side_effect=fake_fft),
                patch.object(classifier.radial_extractor, "forward", side_effect=capture_radial),
                patch.object(
                    classifier.magnitude_branch,
                    "apply_frequency_mask",
                    side_effect=mask_to_zero,
                ),
            ):
                classifier.forward_components(torch.zeros(1, 4), torch.rand(1, 3, 8, 8))
        finally:
            handle.remove()

        self.assertEqual(events, ["radial", "mask"])
        self.assertTrue(torch.equal(observed["radial_input"], magnitude))
        self.assertTrue(torch.equal(observed["mask_input"], magnitude))
        self.assertTrue(torch.equal(observed["cnn_input"], torch.zeros_like(magnitude)))

    def test_learned_weights_and_exact_residual_arithmetic(self) -> None:
        classifier = LearnedFrequencyResidualClassifier(
            spatial_dim=2,
            frequency_scale=0.25,
            branch_dropout=0.0,
        ).eval()
        initial = classifier.normalized_frequency_weights()
        self.assertTrue(torch.allclose(initial, torch.full((3,), 1.0 / 3.0)))
        self.assertAlmostEqual(float(initial.detach().sum()), 1.0, places=6)
        self.assertTrue((initial >= 0).all())

        desired = torch.tensor([0.2, 0.3, 0.5])
        with torch.no_grad():
            classifier.raw_frequency_weights.copy_(desired.log())
        spatial = torch.tensor([[1.0]])
        magnitude = torch.tensor([[2.0]])
        phase = torch.tensor([[4.0]])
        radial = torch.tensor([[8.0]])
        output = classifier.combine_logits(spatial, magnitude, phase, radial)
        expected = spatial + 0.25 * (0.2 * magnitude + 0.3 * phase + 0.5 * radial)
        self.assertTrue(torch.allclose(output, expected))

        zero_scale = LearnedFrequencyResidualClassifier(
            spatial_dim=2,
            frequency_scale=0.0,
            branch_dropout=0.0,
        ).eval()
        self.assertTrue(
            torch.equal(
                zero_scale.combine_logits(spatial, magnitude, phase, radial),
                spatial,
            )
        )

    def test_gradient_reaches_all_raw_fusion_parameters(self) -> None:
        classifier = LearnedFrequencyResidualClassifier(
            spatial_dim=2,
            branch_dropout=0.0,
        )
        output = classifier.combine_logits(
            torch.zeros(1, 1),
            torch.tensor([[1.0]]),
            torch.tensor([[2.0]]),
            torch.tensor([[4.0]]),
        )
        output.sum().backward()
        self.assertIsNotNone(classifier.raw_frequency_weights.grad)
        self.assertTrue((classifier.raw_frequency_weights.grad != 0).all())

    def test_combined_branch_dropout_matches_v3_semantics(self) -> None:
        classifier = LearnedFrequencyResidualClassifier(
            spatial_dim=2,
            branch_dropout=1.0,
        ).train()
        spatial = torch.tensor([[1.0]])
        branches = (torch.tensor([[2.0]]), torch.tensor([[4.0]]), torch.tensor([[8.0]]))
        self.assertTrue(torch.equal(classifier.combine_logits(spatial, *branches), spatial))
        classifier.eval()
        self.assertFalse(torch.equal(classifier.combine_logits(spatial, *branches), spatial))


class HybridV31DiagnosticsAndTrainerTests(unittest.TestCase):
    def test_evaluator_uses_one_forward_for_predictions_and_diagnostics(self) -> None:
        class CountingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def component_forward(
                self,
                images: torch.Tensor,
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                self.calls += 1
                spatial = images[:, :1, 0, 0]
                magnitude = spatial + 1.0
                phase = spatial + 2.0
                radial = spatial + 3.0
                return spatial, {
                    "spatial_logit": spatial,
                    "magnitude_logit": magnitude,
                    "phase_logit": phase,
                    "radial_logit": radial,
                }

        images = torch.zeros(3, 3, 2, 2)
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = CountingModel()
        result = evaluate(
            model,
            loader,
            nn.BCEWithLogitsLoss(),
            torch.device("cpu"),
            component_forward=lambda current, batch: current.component_forward(batch),
        )
        self.assertEqual(model.calls, len(loader))
        self.assertEqual(result["probabilities"], [0.5, 0.5, 0.5])
        diagnostics = result["mean_absolute_branch_logits"]
        self.assertEqual(diagnostics["spatial_logit"], 0.0)
        self.assertEqual(diagnostics["magnitude_logit"], 1.0)
        self.assertEqual(diagnostics["phase_logit"], 2.0)
        self.assertEqual(diagnostics["radial_logit"], 3.0)

    def test_stage_groups_include_radial_and_learned_fusion(self) -> None:
        model = HybridV31AIGCDetector(pretrained_spatial=False)
        model.spatial_classifier_loaded = True
        frozen, groups = configure_hybrid_v31_stage(model, "stage1")
        self.assertEqual(len(frozen), 10)
        self.assertEqual(
            [group["name"] for group in groups],
            ["magnitude", "phase", "radial", "frequency_fusion"],
        )
        self.assertEqual([group["lr"] for group in groups], [5e-5] * 4)
        self.assertTrue(model.classifier.raw_frequency_weights.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))

        frozen, groups = configure_hybrid_v31_stage(model, "stage2")
        self.assertEqual(len(frozen), 6)
        self.assertEqual(
            [group["name"] for group in groups],
            [
                "magnitude",
                "phase",
                "radial",
                "frequency_fusion",
                "spatial_classifier",
                "backbone",
            ],
        )
        for index, block in enumerate(model.features):
            self.assertTrue(all(p.requires_grad is (index >= 6) for p in block.parameters()))

    def test_trainer_reports_once_per_epoch_and_saves_dynamic_diagnostics(self) -> None:
        model = HybridV31AIGCDetector(pretrained_spatial=False)
        model.spatial_classifier_loaded = True
        saved: list[dict[str, object]] = []

        def fake_train(*args: object, **kwargs: object) -> tuple[float, float]:
            optimizer = args[3]
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return 0.5, 0.5

        auc_values = iter((0.6, 0.7))

        def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
            self.assertIs(kwargs["component_forward"], v31_validation_forward)
            return {
                "loss": 0.5,
                "accuracy": 0.5,
                "auc_roc": next(auc_values),
                "mean_absolute_branch_logits": {
                    "spatial_logit": 1.0,
                    "magnitude_logit": 2.0,
                    "phase_logit": 3.0,
                    "radial_logit": 4.0,
                },
            }

        output = io.StringIO()
        with (
            patch("src.train.train_one_epoch", side_effect=fake_train),
            patch("src.train.evaluate", side_effect=fake_evaluate),
            patch("src.train.torch.save", side_effect=lambda payload, _: saved.append(payload)),
            redirect_stdout(output),
        ):
            train_staged_model(
                model,
                Mock(),
                Mock(),
                torch.device("cpu"),
                stage1_epochs=1,
                stage2_epochs=1,
                checkpoint_path=Path("unused.pt"),
                stage_configurator=configure_hybrid_v31_stage,
                validation_component_forward=v31_validation_forward,
                epoch_metadata_provider=v31_epoch_metadata,
            )
        self.assertEqual(output.getvalue().count("Frequency fusion weights:"), 2)
        self.assertEqual(output.getvalue().count("Validation mean absolute branch logits:"), 2)
        self.assertIn("learned_frequency_weights", saved[-1])
        self.assertEqual(
            saved[-1]["validation_mean_absolute_branch_logits"]["radial_logit"],
            4.0,
        )


class HybridV31CheckpointAndCliTests(unittest.TestCase):
    def test_model_loader_reconstructs_v31_strictly(self) -> None:
        model = HybridV31AIGCDetector(
            pretrained_spatial=False,
            radial_bins=16,
            frequency_scale=0.1,
        )
        with torch.no_grad():
            model.classifier.raw_frequency_weights.copy_(torch.tensor([1.0, 0.0, -1.0]))
        with patch("src.model.torch.load", return_value=v31_checkpoint(model)):
            loaded = load_model(Path("v31.pt"), torch.device("cpu"))
        self.assertIsInstance(loaded, HybridV31AIGCDetector)
        self.assertEqual(loaded.radial_bins, 16)
        self.assertTrue(
            torch.equal(
                loaded.classifier.raw_frequency_weights,
                model.classifier.raw_frequency_weights,
            )
        )
        self.assertTrue(expects_unnormalized_input(loaded))
        self.assertFalse(loaded.training)

    def test_checkpoint_paths_and_cli_validation(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v31", None),
            app.DEFAULT_ALL_SOURCE_HYBRID_V31_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v31", None, "DDPM"),
            Path("checkpoints/hybrid_v31_balanced_holdout_ddpm_best.pt"),
        )
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v31",
                None,
                run_name="Radial 32",
            ),
            Path("checkpoints/hybrid_v31_radial_32_all_sources_best.pt"),
        )
        for extra in (
            ["--radial-bins", "3"],
            ["--v2-checkpoint", "v2.pt"],
        ):
            with self.subTest(extra=extra):
                with (
                    patch.object(sys, "argv", ["main.py", "train-hybrid-v31", *extra]),
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        app.main()

    def test_automatic_v31_checkpoint_overwrite_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.pt"
            existing.touch()
            with (
                patch.object(sys, "argv", ["main.py", "train-hybrid-v31"]),
                patch("main.resolve_checkpoint_path", return_value=existing),
                patch("main.get_device") as device_mock,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    app.main()
            device_mock.assert_not_called()

    def test_v31_dispatch_reuses_balanced_loader_and_shared_trainer(self) -> None:
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE train FAKE")])
        )
        validation_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE test REAL")])
        )
        model = Mock(
            spatial_classifier_loaded=True,
            spatial_classifier_source="EfficientNet classifier.1",
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "train-hybrid-v31",
                    "--checkpoint",
                    "checkpoints/explicit.pt",
                    "--spatial-checkpoint",
                    "checkpoints/spatial.pt",
                    "--radial-bins",
                    "16",
                    "--run-name",
                    "trial",
                ],
            ),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_hybrid_v31_model", return_value=model) as build_mock,
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertFalse(loader_mock.call_args.kwargs["normalize_inputs"])
        build_mock.assert_called_once_with(
            torch.device("cpu"),
            spatial_checkpoint=Path("checkpoints/spatial.pt"),
            frequency_scale=0.25,
            frequency_branch_dropout=0.2,
            frequency_mask_probability=0.0,
            radial_bins=16,
        )
        self.assertIs(
            train_mock.call_args.kwargs["stage_configurator"],
            configure_hybrid_v31_stage,
        )
        self.assertIs(
            train_mock.call_args.kwargs["validation_component_forward"],
            v31_validation_forward,
        )
        self.assertIs(
            train_mock.call_args.kwargs["epoch_metadata_provider"],
            v31_epoch_metadata,
        )
        metadata = train_mock.call_args.kwargs["checkpoint_metadata"]
        self.assertEqual(metadata["model_type"], HYBRID_V31_MODEL_TYPE)
        self.assertEqual(metadata["radial_bins"], 16)


if __name__ == "__main__":
    unittest.main()
