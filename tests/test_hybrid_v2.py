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

import main as app
from src.hybrid_v2_model import (
    HYBRID_V2_MODEL_TYPE,
    V2_FREQUENCY_FEATURE_DIM,
    V2_FREQUENCY_HIDDEN_DIM,
    V2_SPATIAL_FEATURE_DIM,
    FrequencyBranchV2,
    HybridV2AIGCDetector,
    ResidualHybridClassifier,
    configure_hybrid_v2_stage,
    load_v2_spatial_checkpoint,
)
from src.model import expects_unnormalized_input, load_model
from src.train import train_staged_model


def efficientnet_state_for(model: HybridV2AIGCDetector) -> dict[str, torch.Tensor]:
    state = {
        f"features.{key}": value.detach().clone()
        for key, value in model.features.state_dict().items()
    }
    state["classifier.1.weight"] = torch.full((1, V2_SPATIAL_FEATURE_DIM), 0.125)
    state["classifier.1.bias"] = torch.tensor([0.25])
    return state


def v2_checkpoint(model: HybridV2AIGCDetector) -> dict[str, object]:
    return {
        "model_type": HYBRID_V2_MODEL_TYPE,
        "model_state_dict": model.state_dict(),
        "spatial_feature_dim": V2_SPATIAL_FEATURE_DIM,
        "frequency_feature_dim": V2_FREQUENCY_FEATURE_DIM,
        "frequency_hidden_dim": V2_FREQUENCY_HIDDEN_DIM,
        "frequency_dropout": 0.5,
        "frequency_scale": model.frequency_scale,
        "frequency_branch_dropout": model.frequency_branch_dropout,
        "frequency_mask_prob": model.frequency_mask_probability,
        "spatial_classifier_loaded": True,
        "spatial_classifier_source": "EfficientNet classifier.1",
    }


class HybridV2ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = HybridV2AIGCDetector(pretrained_spatial=False).eval()

    def test_component_and_output_shapes(self) -> None:
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            spatial_features = self.model.extract_spatial_features(images)
            frequency_features = self.model.extract_frequency_features(images)
            spatial_logit, frequency_logit = self.model.forward_components(images)
            output = self.model(images)
            one_output = self.model(images[:1])
        self.assertEqual(spatial_features.shape, (2, V2_SPATIAL_FEATURE_DIM))
        self.assertEqual(frequency_features.shape, (2, V2_FREQUENCY_FEATURE_DIM))
        self.assertEqual(spatial_logit.shape, (2, 1))
        self.assertEqual(frequency_logit.shape, (2, 1))
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(one_output.shape, (1, 1))

    def test_residual_fusion_matches_exact_formula_for_supported_scales(self) -> None:
        spatial = torch.tensor([[0.5], [-0.25]])
        frequency = torch.tensor([[2.0], [4.0]])
        for scale in (0.0, 0.1, 0.25, 1.0):
            with self.subTest(scale=scale):
                classifier = ResidualHybridClassifier(
                    spatial_dim=2,
                    frequency_scale=scale,
                    branch_dropout=0.0,
                ).eval()
                combined = classifier.combine_logits(spatial, frequency)
                self.assertTrue(torch.equal(combined, spatial + scale * frequency))
        classifier = ResidualHybridClassifier(
            spatial_dim=2,
            frequency_scale=0.0,
        ).eval()
        self.assertTrue(
            torch.equal(
                classifier.combine_logits(spatial, frequency),
                classifier.combine_logits(spatial, frequency * 1000),
            )
        )

    def test_branch_dropout_training_and_evaluation_edges(self) -> None:
        spatial = torch.tensor([[0.5], [-0.25]])
        frequency = torch.tensor([[2.0], [4.0]])

        never_drop = ResidualHybridClassifier(
            spatial_dim=2,
            frequency_scale=0.25,
            branch_dropout=0.0,
        ).train()
        self.assertTrue(
            torch.equal(
                never_drop.combine_logits(spatial, frequency),
                spatial + 0.25 * frequency,
            )
        )

        always_drop = ResidualHybridClassifier(
            spatial_dim=2,
            frequency_scale=0.25,
            branch_dropout=1.0,
        ).train()
        self.assertTrue(
            torch.equal(always_drop.combine_logits(spatial, frequency), spatial)
        )
        always_drop.eval()
        self.assertTrue(
            torch.equal(
                always_drop.combine_logits(spatial, frequency),
                spatial + 0.25 * frequency,
            )
        )

    def test_frequency_mask_is_seeded_training_only_and_never_masks_everything(self) -> None:
        branch = FrequencyBranchV2(mask_probability=1.0).train()
        spectrum = torch.ones(2, 1, 32, 32)
        torch.manual_seed(42)
        first = branch.apply_frequency_mask(spectrum)
        torch.manual_seed(42)
        second = branch.apply_frequency_mask(spectrum)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue((first == 0).any())
        self.assertTrue((first != 0).any())
        self.assertTrue(torch.equal(spectrum, torch.ones_like(spectrum)))
        branch.eval()
        self.assertTrue(torch.equal(branch.apply_frequency_mask(spectrum), spectrum))

    def test_both_branches_receive_the_same_raw_augmented_tensor(self) -> None:
        model = HybridV2AIGCDetector(pretrained_spatial=False).eval()
        images = torch.rand(2, 3, 32, 32)
        captured: dict[str, torch.Tensor] = {}

        def fake_spatial(inputs: torch.Tensor) -> torch.Tensor:
            captured["spatial"] = inputs
            return torch.zeros(inputs.shape[0], V2_SPATIAL_FEATURE_DIM)

        def capture_frequency(
            _: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            captured["frequency"] = inputs[0]

        handle = model.classifier.frequency_branch.register_forward_pre_hook(
            capture_frequency
        )
        try:
            with patch.object(model, "extract_spatial_features", side_effect=fake_spatial):
                with torch.no_grad():
                    model(images)
        finally:
            handle.remove()
        self.assertIs(captured["spatial"], images)
        self.assertIs(captured["frequency"], images)


class HybridV2InitializationAndTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = HybridV2AIGCDetector(pretrained_spatial=False)

    def test_strict_features_and_compatible_binary_classifier_are_loaded(self) -> None:
        state = efficientnet_state_for(self.model)
        output = io.StringIO()
        with (
            patch(
                "src.hybrid_v2_model.torch.load",
                return_value={"model_state_dict": state},
            ),
            redirect_stdout(output),
        ):
            result = load_v2_spatial_checkpoint(
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
        self.assertIn("Loaded spatial classifier", output.getvalue())

    def test_v1_classifier_falls_back_to_reported_random_spatial_head(self) -> None:
        state = efficientnet_state_for(self.model)
        state.pop("classifier.1.weight")
        state.pop("classifier.1.bias")
        state["classifier.frequency_branch.cnn.0.weight"] = torch.zeros(32, 1, 3, 3)
        before = self.model.classifier.spatial_classifier.weight.detach().clone()
        output = io.StringIO()
        with (
            patch("src.hybrid_v2_model.torch.load", return_value=state),
            redirect_stdout(output),
        ):
            result = load_v2_spatial_checkpoint(
                self.model,
                Path("hybrid_v1.pt"),
                torch.device("cpu"),
            )
        self.assertFalse(result.classifier_loaded)
        self.assertTrue(
            torch.equal(before, self.model.classifier.spatial_classifier.weight)
        )
        self.assertIn("ignored classifier entries", output.getvalue())

    def test_partial_classifier_and_inexact_features_fail_before_loading(self) -> None:
        valid = efficientnet_state_for(self.model)
        cases: list[dict[str, torch.Tensor]] = []
        partial = dict(valid)
        partial.pop("classifier.1.bias")
        cases.append(partial)
        missing = dict(valid)
        missing.pop(next(key for key in missing if key.startswith("features.")))
        cases.append(missing)

        for state in cases:
            before = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
            with self.subTest(keys=len(state)):
                with patch("src.hybrid_v2_model.torch.load", return_value=state):
                    with self.assertRaises(ValueError):
                        load_v2_spatial_checkpoint(
                            self.model,
                            Path("invalid.pt"),
                            torch.device("cpu"),
                        )
                for key, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)

    def test_stage_configuration_respects_loaded_and_random_spatial_heads(self) -> None:
        self.model.spatial_classifier_loaded = True
        frozen, groups = configure_hybrid_v2_stage(self.model, "stage1")
        self.assertEqual([group["name"] for group in groups], ["frequency"])
        self.assertEqual(len(frozen), 10)
        self.assertTrue(all(not p.requires_grad for p in self.model.features.parameters()))
        self.assertTrue(
            all(
                not p.requires_grad
                for p in self.model.classifier.spatial_classifier.parameters()
            )
        )
        self.assertTrue(
            all(
                p.requires_grad
                for p in self.model.classifier.frequency_branch.parameters()
            )
        )

        self.model.spatial_classifier_loaded = False
        _, groups = configure_hybrid_v2_stage(self.model, "stage1")
        self.assertEqual(
            [group["name"] for group in groups],
            ["frequency", "spatial_classifier"],
        )
        self.assertEqual([group["lr"] for group in groups], [5e-5, 1e-5])

        frozen, groups = configure_hybrid_v2_stage(self.model, "stage2")
        self.assertEqual(len(frozen), 6)
        self.assertEqual(
            [group["name"] for group in groups],
            ["frequency", "spatial_classifier", "backbone"],
        )
        self.assertEqual([group["lr"] for group in groups], [5e-5, 1e-5, 1e-5])
        for index, block in enumerate(self.model.features):
            self.assertTrue(
                all(p.requires_grad is (index >= 6) for p in block.parameters())
            )

    def test_shared_trainer_uses_v2_stage_groups_without_a_duplicate_loop(self) -> None:
        self.model.spatial_classifier_loaded = True
        observed: list[tuple[list[str], list[float], int]] = []

        def fake_train(*args: object, **kwargs: object) -> tuple[float, float]:
            optimizer = args[3]
            frozen_modules = kwargs["frozen_modules"]
            observed.append(
                (
                    [str(group["name"]) for group in optimizer.param_groups],
                    [float(group["lr"]) for group in optimizer.param_groups],
                    len(frozen_modules),
                )
            )
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return 0.5, 0.5

        auc_values = iter((0.6, 0.7))

        def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
            return {"loss": 0.5, "accuracy": 0.5, "auc_roc": next(auc_values)}

        with (
            patch("src.train.train_one_epoch", side_effect=fake_train),
            patch("src.train.evaluate", side_effect=fake_evaluate),
            patch("src.train.torch.save"),
            redirect_stdout(io.StringIO()),
        ):
            train_staged_model(
                self.model,
                Mock(),
                Mock(),
                torch.device("cpu"),
                stage1_epochs=1,
                stage2_epochs=1,
                checkpoint_path=Path("unused.pt"),
                stage_configurator=configure_hybrid_v2_stage,
            )

        self.assertEqual(observed[0], (["frequency"], [5e-5], 10))
        self.assertEqual(
            observed[1],
            (["frequency", "spatial_classifier", "backbone"], [5e-5, 1e-5, 1e-5], 6),
        )


class HybridV2CheckpointAndCliTests(unittest.TestCase):
    def test_model_loader_reconstructs_v2_from_metadata_strictly(self) -> None:
        model = HybridV2AIGCDetector(
            pretrained_spatial=False,
            frequency_scale=0.1,
            frequency_branch_dropout=0.3,
            frequency_mask_probability=0.2,
        )
        with patch("src.model.torch.load", return_value=v2_checkpoint(model)):
            loaded = load_model(Path("v2.pt"), torch.device("cpu"))
        self.assertIsInstance(loaded, HybridV2AIGCDetector)
        self.assertEqual(loaded.frequency_scale, 0.1)
        self.assertEqual(loaded.frequency_branch_dropout, 0.3)
        self.assertEqual(loaded.frequency_mask_probability, 0.2)
        self.assertTrue(expects_unnormalized_input(loaded))
        self.assertFalse(loaded.training)

    def test_v2_checkpoint_names_and_explicit_override(self) -> None:
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v2", None),
            app.DEFAULT_ALL_SOURCE_HYBRID_V2_CHECKPOINT,
        )
        self.assertEqual(
            app.resolve_checkpoint_path("train-hybrid-v2", None, "DDPM"),
            Path("checkpoints/hybrid_v2_balanced_holdout_ddpm_best.pt"),
        )
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v2",
                None,
                run_name="Alpha 0.25",
            ),
            Path("checkpoints/hybrid_v2_alpha_0_25_all_sources_best.pt"),
        )
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v2",
                None,
                holdout="Midjourney v6",
                run_name="Alpha025",
            ),
            Path(
                "checkpoints/hybrid_v2_alpha025_holdout_midjourney_v6_best.pt"
            ),
        )
        explicit = Path("checkpoints/explicit.pt")
        self.assertEqual(
            app.resolve_checkpoint_path(
                "train-hybrid-v2",
                explicit,
                run_name="ignored-for-path",
            ),
            explicit,
        )

    def test_automatic_existing_checkpoint_is_refused_before_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.pt"
            existing.touch()
            with (
                patch.object(sys, "argv", ["main.py", "train-hybrid-v2"]),
                patch(
                    "main.resolve_checkpoint_path",
                    return_value=existing,
                ),
                patch("main.get_device") as device_mock,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    app.main()
            device_mock.assert_not_called()

    def test_v2_dispatch_reuses_balanced_loader_and_shared_trainer(self) -> None:
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
                    "train-hybrid-v2",
                    "--checkpoint",
                    "checkpoints/explicit.pt",
                    "--spatial-checkpoint",
                    "checkpoints/spatial.pt",
                    "--frequency-scale",
                    "0.1",
                    "--frequency-branch-dropout",
                    "0.3",
                    "--frequency-mask-prob",
                    "0.2",
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
            patch("main.build_hybrid_v2_model", return_value=model) as build_mock,
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertFalse(loader_mock.call_args.kwargs["normalize_inputs"])
        build_mock.assert_called_once_with(
            torch.device("cpu"),
            spatial_checkpoint=Path("checkpoints/spatial.pt"),
            frequency_scale=0.1,
            frequency_branch_dropout=0.3,
            frequency_mask_probability=0.2,
        )
        self.assertIs(
            train_mock.call_args.kwargs["stage_configurator"],
            configure_hybrid_v2_stage,
        )
        metadata = train_mock.call_args.kwargs["checkpoint_metadata"]
        self.assertEqual(metadata["model_type"], HYBRID_V2_MODEL_TYPE)
        self.assertEqual(metadata["frequency_scale"], 0.1)
        self.assertEqual(metadata["run_name"], "trial")

    def test_v2_dispatch_reuses_existing_arbitrary_holdout_path(self) -> None:
        train_loader = SimpleNamespace(
            dataset=SimpleNamespace(sources=[SimpleNamespace(name="CIFAKE train FAKE")])
        )
        validation_loader = SimpleNamespace(
            dataset=SimpleNamespace(
                sources=[SimpleNamespace(name="WildFake Midjourney test")]
            )
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
                    "train-hybrid-v2",
                    "--holdout",
                    "midjourney",
                    "--checkpoint",
                    "checkpoints/explicit.pt",
                ],
            ),
            patch("main.resolve_wildfake_holdout", return_value="Midjourney"),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.download_dataset", return_value=Path("cifake")),
            patch(
                "main.get_source_balanced_data_loaders",
                return_value=(train_loader, validation_loader),
            ) as loader_mock,
            patch("main.build_hybrid_v2_model", return_value=model),
            patch("main.train_staged_model") as train_mock,
            redirect_stdout(io.StringIO()),
        ):
            app.main()

        self.assertEqual(loader_mock.call_args.kwargs["holdout"], "Midjourney")
        self.assertEqual(
            train_mock.call_args.kwargs["heldout_generator"],
            "Midjourney",
        )


if __name__ == "__main__":
    unittest.main()
