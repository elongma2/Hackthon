from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import main as app
from src.hybrid_v2_model import HYBRID_V2_MODEL_TYPE, HybridV2AIGCDetector
from src.hybrid_v3_model import (
    HYBRID_V3_MODEL_TYPE,
    V3_MAGNITUDE_FEATURE_DIM,
    V3_PHASE_FEATURE_DIM,
    V3_SPATIAL_FEATURE_DIM,
    HybridV3AIGCDetector,
    MagnitudePhaseResidualClassifier,
    SharedFFTPreprocessor,
    configure_hybrid_v3_stage,
    load_v2_warm_start,
    load_v3_spatial_checkpoint,
)
from src.model import expects_unnormalized_input, load_model


def v3_checkpoint(model: HybridV3AIGCDetector) -> dict[str, object]:
    return {
        "model_type": HYBRID_V3_MODEL_TYPE,
        "model_state_dict": model.state_dict(),
        "spatial_feature_dim": V3_SPATIAL_FEATURE_DIM,
        "magnitude_feature_dim": V3_MAGNITUDE_FEATURE_DIM,
        "phase_feature_dim": V3_PHASE_FEATURE_DIM,
        "frequency_hidden_dim": 64,
        "frequency_dropout": 0.5,
        "frequency_scale": model.frequency_scale,
        "supplied_magnitude_weight": model.supplied_magnitude_weight,
        "supplied_phase_weight": model.supplied_phase_weight,
        "normalized_magnitude_weight": model.magnitude_weight,
        "normalized_phase_weight": model.phase_weight,
        "frequency_branch_dropout": model.frequency_branch_dropout,
        "frequency_mask_prob": model.frequency_mask_probability,
        "fft_normalization": "ortho",
        "phase_representation": "sin_cos",
        "spatial_classifier_loaded": True,
        "spatial_classifier_source": "EfficientNet classifier.1",
        "magnitude_initialized_from_v2": False,
    }


def efficientnet_state_for(model: HybridV3AIGCDetector) -> dict[str, torch.Tensor]:
    state = {
        f"features.{key}": value.detach().clone()
        for key, value in model.features.state_dict().items()
    }
    state["classifier.1.weight"] = torch.full((1, V3_SPATIAL_FEATURE_DIM), 0.125)
    state["classifier.1.bias"] = torch.tensor([0.25])
    return state


class HybridV3ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HybridV3AIGCDetector(pretrained_spatial=False).eval()

    def test_shared_fft_outputs_and_phase_identity(self) -> None:
        preprocessor = SharedFFTPreprocessor()
        for images in (
            torch.rand(2, 3, 32, 32),
            torch.zeros(2, 3, 32, 32),
            torch.full((2, 3, 32, 32), 0.5) + torch.rand(2, 3, 32, 32) * 1e-7,
        ):
            magnitude, phase = preprocessor(images)
            self.assertEqual(magnitude.shape, (2, 1, 32, 32))
            self.assertEqual(phase.shape, (2, 2, 32, 32))
            self.assertEqual(magnitude.dtype, torch.float32)
            self.assertEqual(phase.dtype, torch.float32)
            self.assertTrue(torch.isfinite(magnitude).all())
            self.assertTrue(torch.isfinite(phase).all())
            identity = phase[:, :1].square() + phase[:, 1:].square()
            self.assertTrue(torch.allclose(identity, torch.ones_like(identity), atol=1e-6))

    def test_fft_is_computed_once_per_model_forward(self) -> None:
        model = HybridV3AIGCDetector(pretrained_spatial=False).eval()
        images = torch.rand(2, 3, 32, 32)
        original_fft2 = torch.fft.fft2
        with patch("src.hybrid_v3_model.torch.fft.fft2", wraps=original_fft2) as fft2:
            with torch.no_grad():
                output = model(images)
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(fft2.call_count, 1)

    def test_all_paths_receive_the_same_augmented_tensor(self) -> None:
        model = HybridV3AIGCDetector(pretrained_spatial=False).eval()
        images = torch.rand(2, 3, 32, 32)
        captured: dict[str, torch.Tensor] = {}

        def fake_spatial(inputs: torch.Tensor) -> torch.Tensor:
            captured["spatial"] = inputs
            return torch.zeros(inputs.shape[0], V3_SPATIAL_FEATURE_DIM)

        def capture_fft(
            _: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            captured["frequency"] = inputs[0]

        handle = model.classifier.fft_preprocessor.register_forward_pre_hook(capture_fft)
        try:
            with patch.object(model, "extract_spatial_features", side_effect=fake_spatial):
                with torch.no_grad():
                    model(images)
        finally:
            handle.remove()
        self.assertIs(captured["spatial"], images)
        self.assertIs(captured["frequency"], images)

    def test_component_shapes_include_batch_size_one(self) -> None:
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            spatial = self.model.extract_spatial_features(images)
            magnitude, phase = self.model.extract_frequency_features(images)
            spatial_logit, magnitude_logit, phase_logit = self.model.forward_components(images)
            output = self.model(images)
            one_output = self.model(images[:1])
        self.assertEqual(spatial.shape, (2, V3_SPATIAL_FEATURE_DIM))
        self.assertEqual(magnitude.shape, (2, V3_MAGNITUDE_FEATURE_DIM))
        self.assertEqual(phase.shape, (2, V3_PHASE_FEATURE_DIM))
        self.assertEqual(spatial_logit.shape, (2, 1))
        self.assertEqual(magnitude_logit.shape, (2, 1))
        self.assertEqual(phase_logit.shape, (2, 1))
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(one_output.shape, (1, 1))

    def test_residual_arithmetic_modes_and_weight_normalization(self) -> None:
        spatial = torch.tensor([[0.5], [-0.25]])
        magnitude = torch.tensor([[2.0], [4.0]])
        phase = torch.tensor([[6.0], [8.0]])
        for scale in (0.0, 0.1, 0.25, 1.0):
            classifier = MagnitudePhaseResidualClassifier(
                spatial_dim=2,
                frequency_scale=scale,
                magnitude_weight=1.0,
                phase_weight=1.0,
                branch_dropout=0.0,
            ).eval()
            expected = spatial + scale * (0.5 * magnitude + 0.5 * phase)
            self.assertTrue(
                torch.equal(classifier.combine_logits(spatial, magnitude, phase), expected)
            )

        magnitude_only = MagnitudePhaseResidualClassifier(
            spatial_dim=2,
            magnitude_weight=1.0,
            phase_weight=0.0,
            branch_dropout=0.0,
        ).eval()
        phase_only = MagnitudePhaseResidualClassifier(
            spatial_dim=2,
            magnitude_weight=0.0,
            phase_weight=1.0,
            branch_dropout=0.0,
        ).eval()
        self.assertTrue(
            torch.equal(
                magnitude_only.combine_logits(spatial, magnitude, phase),
                spatial + 0.25 * magnitude,
            )
        )
        self.assertTrue(
            torch.equal(
                phase_only.combine_logits(spatial, magnitude, phase),
                spatial + 0.25 * phase,
            )
        )
        with self.assertRaises(ValueError):
            MagnitudePhaseResidualClassifier(magnitude_weight=0.0, phase_weight=0.0)

    def test_branch_dropout_applies_after_combining_residual(self) -> None:
        spatial = torch.tensor([[0.5], [-0.25]])
        magnitude = torch.tensor([[2.0], [4.0]])
        phase = torch.tensor([[6.0], [8.0]])
        always_drop = MagnitudePhaseResidualClassifier(
            spatial_dim=2,
            branch_dropout=1.0,
        ).train()
        self.assertTrue(
            torch.equal(always_drop.combine_logits(spatial, magnitude, phase), spatial)
        )
        always_drop.eval()
        expected = spatial + 0.25 * (0.5 * magnitude + 0.5 * phase)
        self.assertTrue(
            torch.equal(always_drop.combine_logits(spatial, magnitude, phase), expected)
        )

    def test_masking_changes_only_magnitude_and_is_seeded(self) -> None:
        classifier = MagnitudePhaseResidualClassifier(mask_probability=1.0).train()
        magnitude = torch.ones(2, 1, 32, 32)
        phase = torch.ones(2, 2, 32, 32)
        torch.manual_seed(42)
        first = classifier.magnitude_branch.apply_frequency_mask(magnitude)
        torch.manual_seed(42)
        second = classifier.magnitude_branch.apply_frequency_mask(magnitude)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue((first == 0).any())
        self.assertTrue((first != 0).any())
        self.assertTrue(torch.equal(phase, torch.ones_like(phase)))
        classifier.eval()
        self.assertTrue(
            torch.equal(
                classifier.magnitude_branch.apply_frequency_mask(magnitude),
                magnitude,
            )
        )

    def test_float32_fft_inside_cpu_autocast(self) -> None:
        preprocessor = SharedFFTPreprocessor()
        images = torch.rand(2, 3, 32, 32).to(torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            magnitude, phase = preprocessor(images)
        self.assertEqual(magnitude.dtype, torch.float32)
        self.assertEqual(phase.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_float32_fft_inside_cuda_autocast(self) -> None:
        preprocessor = SharedFFTPreprocessor().cuda()
        images = torch.rand(2, 3, 32, 32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            magnitude, phase = preprocessor(images)
        self.assertEqual(magnitude.dtype, torch.float32)
        self.assertEqual(phase.dtype, torch.float32)


class HybridV3InitializationAndTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = HybridV3AIGCDetector(pretrained_spatial=False)

    def test_strict_spatial_checkpoint_loads_features_and_binary_head(self) -> None:
        state = efficientnet_state_for(self.model)
        with patch("src.hybrid_v3_model.torch.load", return_value=state):
            result = load_v3_spatial_checkpoint(
                self.model,
                Path("efficientnet.pt"),
                torch.device("cpu"),
            )
        self.assertEqual(result.feature_entry_count, len(self.model.features.state_dict()))
        self.assertTrue(result.classifier_loaded)
        self.assertTrue(self.model.spatial_classifier_loaded)
        self.assertTrue(
            torch.equal(
                self.model.classifier.spatial_classifier.weight,
                state["classifier.1.weight"],
            )
        )

    def test_exact_v2_warm_start_loads_spatial_and_magnitude_only(self) -> None:
        v2 = HybridV2AIGCDetector(pretrained_spatial=False)
        phase_before = {
            key: value.detach().clone()
            for key, value in self.model.classifier.phase_branch.state_dict().items()
        }
        checkpoint = {
            "model_type": HYBRID_V2_MODEL_TYPE,
            "model_state_dict": v2.state_dict(),
        }
        with (
            patch("src.hybrid_v3_model.torch.load", return_value=checkpoint),
            redirect_stdout(io.StringIO()),
        ):
            result = load_v2_warm_start(
                self.model,
                Path("v2.pt"),
                torch.device("cpu"),
            )
        self.assertTrue(result.classifier_loaded)
        self.assertTrue(result.magnitude_loaded)
        self.assertTrue(self.model.magnitude_initialized_from_v2)
        for key, value in self.model.classifier.magnitude_branch.cnn.state_dict().items():
            self.assertTrue(
                torch.equal(value, v2.classifier.frequency_branch.cnn.state_dict()[key])
            )
        for key, value in self.model.classifier.magnitude_head.state_dict().items():
            self.assertTrue(torch.equal(value, v2.classifier.frequency_head.state_dict()[key]))
        for key, value in self.model.classifier.phase_branch.state_dict().items():
            self.assertTrue(torch.equal(value, phase_before[key]))

    def test_malformed_v2_warm_start_fails_before_loading(self) -> None:
        v2 = HybridV2AIGCDetector(pretrained_spatial=False)
        state = dict(v2.state_dict())
        state.pop("classifier.frequency_branch.cnn.0.weight")
        before = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        checkpoint = {"model_type": HYBRID_V2_MODEL_TYPE, "model_state_dict": state}
        with patch("src.hybrid_v3_model.torch.load", return_value=checkpoint):
            with self.assertRaises(ValueError):
                load_v2_warm_start(self.model, Path("bad.pt"), torch.device("cpu"))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_stage_configuration_matches_v3_contract(self) -> None:
        self.model.spatial_classifier_loaded = True
        frozen, groups = configure_hybrid_v3_stage(self.model, "stage1")
        self.assertEqual([group["name"] for group in groups], ["magnitude", "phase"])
        self.assertEqual([group["lr"] for group in groups], [5e-5, 5e-5])
        self.assertEqual(len(frozen), 10)
        self.assertTrue(all(not p.requires_grad for p in self.model.features.parameters()))
        self.assertTrue(
            all(p.requires_grad for p in self.model.classifier.magnitude_branch.parameters())
        )
        self.assertTrue(
            all(p.requires_grad for p in self.model.classifier.phase_branch.parameters())
        )

        frozen, groups = configure_hybrid_v3_stage(self.model, "stage2")
        self.assertEqual(len(frozen), 6)
        self.assertEqual(
            [group["name"] for group in groups],
            ["magnitude", "phase", "spatial_classifier", "backbone"],
        )
        self.assertEqual([group["lr"] for group in groups], [5e-5, 5e-5, 1e-5, 1e-5])
        for index, block in enumerate(self.model.features):
            self.assertTrue(all(p.requires_grad is (index >= 6) for p in block.parameters()))


class HybridV3CheckpointAndCliTests(unittest.TestCase):
    def test_model_loader_reconstructs_v3_strictly(self) -> None:
        model = HybridV3AIGCDetector(
            pretrained_spatial=False,
            frequency_scale=0.1,
            magnitude_weight=1.0,
            phase_weight=3.0,
            frequency_branch_dropout=0.3,
            frequency_mask_probability=0.2,
        )
        with patch("src.model.torch.load", return_value=v3_checkpoint(model)):
            loaded = load_model(Path("v3.pt"), torch.device("cpu"))
        self.assertIsInstance(loaded, HybridV3AIGCDetector)
        self.assertEqual(loaded.frequency_scale, 0.1)
        self.assertEqual(loaded.magnitude_weight, 0.25)
        self.assertEqual(loaded.phase_weight, 0.75)
        self.assertTrue(expects_unnormalized_input(loaded))
        self.assertFalse(loaded.training)

    def test_malformed_v3_metadata_fails_clearly(self) -> None:
        model = HybridV3AIGCDetector(pretrained_spatial=False)
        checkpoint = v3_checkpoint(model)
        checkpoint.pop("phase_representation")
        with patch("src.model.torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "phase_representation"):
                load_model(Path("bad-v3.pt"), torch.device("cpu"))

    def test_v3_checkpoint_names_and_explicit_override(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v3", None),
            app.DEFAULT_ALL_SOURCE_HYBRID_V3_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v3", None, "DDPM"),
            Path("checkpoints/hybrid_v3_balanced_holdout_ddpm_best.pt"),
        )
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v3",
                None,
                holdout="Midjourney v6",
                run_name="Dual Spectrum",
            ),
            Path("checkpoints/hybrid_v3_dual_spectrum_holdout_midjourney_v6_best.pt"),
        )
        explicit = Path("checkpoints/explicit.pt")
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v3",
                explicit,
                run_name="ignored",
            ),
            explicit,
        )

    def test_v3_cli_validation_rejects_invalid_weights_and_checkpoint_pair(self) -> None:
        cases = (
            ["--magnitude-weight", "0", "--phase-weight", "0"],
            ["--magnitude-weight", "-1", "--phase-weight", "1"],
            ["--spatial-checkpoint", "spatial.pt", "--v2-checkpoint", "v2.pt"],
        )
        for extra in cases:
            with self.subTest(extra=extra):
                with (
                    patch.object(sys, "argv", ["main.py", "train-hybrid-v3", *extra]),
                    redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        app.main()

    def test_v3_dispatch_reuses_balanced_loader_and_shared_trainer(self) -> None:
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE train FAKE")])
        )
        validation_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE test REAL")])
        )
        model = Mock(
            spatial_classifier_loaded=True,
            spatial_classifier_source="EfficientNet classifier.1",
            magnitude_initialized_from_v2=False,
            magnitude_weight=0.5,
            phase_weight=0.5,
        )
        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "train-hybrid-v3",
                    "--checkpoint",
                    "checkpoints/explicit.pt",
                    "--spatial-checkpoint",
                    "checkpoints/spatial.pt",
                    "--magnitude-weight",
                    "1",
                    "--phase-weight",
                    "1",
                    "--run-name",
                    "dual",
                ],
            ),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_hybrid_v3_model", return_value=model) as build_mock,
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertFalse(loader_mock.call_args.kwargs["normalize_inputs"])
        build_mock.assert_called_once_with(
            torch.device("cpu"),
            spatial_checkpoint=Path("checkpoints/spatial.pt"),
            v2_checkpoint=None,
            frequency_scale=0.25,
            magnitude_weight=1.0,
            phase_weight=1.0,
            frequency_branch_dropout=0.2,
            frequency_mask_probability=0.0,
        )
        self.assertIs(
            train_mock.call_args.kwargs["stage_configurator"],
            configure_hybrid_v3_stage,
        )
        metadata = train_mock.call_args.kwargs["checkpoint_metadata"]
        self.assertEqual(metadata["model_type"], HYBRID_V3_MODEL_TYPE)
        self.assertEqual(metadata["normalized_magnitude_weight"], 0.5)
        self.assertEqual(metadata["normalized_phase_weight"], 0.5)
        self.assertEqual(metadata["run_name"], "dual")


if __name__ == "__main__":
    unittest.main()
