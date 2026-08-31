from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

import main as app
import src.robustness_matrix as robustness_matrix_module
from src.distort_dataset2 import generate, required_variants
from src.distort_folder import distort_folder
from src.robustness_matrix import (
    DISTORTION_CONDITION_IDS,
    ROBUSTNESS_CONDITION_IDS,
    audit_prepared_robustness_data,
    calculate_robustness_metrics,
    map_manifest_condition,
    run_robustness_matrix,
)


def write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def create_clean(root: Path) -> Path:
    write_image(root / "FAKE" / "nested" / "same.jpg", (20, 10), (220, 30, 20))
    write_image(root / "REAL" / "other" / "same.jpg", (20, 10), (20, 200, 40))
    return root


def create_folder_schema(root: Path) -> tuple[Path, Path]:
    clean = create_clean(root / "validation")
    distorted = root / "validation_distorted"
    distort_folder(clean, distorted, seed=42)
    return clean, distorted


def create_labelled_schema(root: Path) -> tuple[Path, Path]:
    input_root = root / "source"
    clean = create_clean(input_root / "validation")
    distorted = root / "prepared"
    generate(input_root, distorted, ["validation"], 42, False, False)
    return clean, distorted


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ConstantRealLogit(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.value.expand(images.shape[0], 1)


class RobustnessManifestTests(unittest.TestCase):
    def test_all_official_variants_map_from_manifest_semantics(self) -> None:
        image = Image.new("RGB", (12, 8), color=(100, 120, 140))
        mapped = {
            map_manifest_condition(variant.transform, variant.parameters)
            for variant in required_variants(image, 42, "FAKE/example.jpg")
        }
        self.assertEqual(mapped, set(DISTORTION_CONDITION_IDS))
        self.assertEqual(len(mapped), 19)

    def test_both_existing_manifest_schemas_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_folder, distorted_folder = create_folder_schema(root / "folder")
            clean_labelled, distorted_labelled = create_labelled_schema(root / "labelled")

            folder_audit = audit_prepared_robustness_data(
                clean_folder, distorted_folder, only="jpeg_30"
            )
            labelled_audit = audit_prepared_robustness_data(
                clean_labelled, distorted_labelled, only="crop_80"
            )

            self.assertEqual(folder_audit.schema, "distort_folder")
            self.assertEqual(labelled_audit.schema, "distort_dataset2")
            self.assertEqual(folder_audit.counts["jpeg_30"], {"total": 2, "FAKE": 1, "REAL": 1})
            self.assertEqual(labelled_audit.counts["crop_80"], {"total": 2, "FAKE": 1, "REAL": 1})
            self.assertFalse(
                [warning for warning in labelled_audit.warnings if "Dimension warning" in warning]
            )

    def test_stable_source_ids_survive_copied_absolute_paths_and_duplicate_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean, distorted = create_folder_schema(Path(directory))
            manifest = distorted / "distortions.csv"
            fieldnames, rows = read_manifest(manifest)
            for row in rows:
                source = row["source"].replace("\\", "/")
                class_marker = "/FAKE/" if "/FAKE/" in source else "/REAL/"
                suffix = source.split(class_marker, 1)[1]
                class_name = class_marker.strip("/")
                row["source"] = f"Z:/old-machine/archive/{class_name}/{suffix}"
                output_name = Path(row["output"]).name
                row["output"] = f"Z:/old-machine/output/{class_name}/{Path(suffix).parent.as_posix()}/{output_name}"
            write_manifest(manifest, fieldnames, rows)

            audit = audit_prepared_robustness_data(clean, distorted, only="jpeg_30")
            source_ids = {record.source_id for record in audit.condition_records["jpeg_30"]}
            self.assertEqual(source_ids, {"FAKE/nested/same.jpg", "REAL/other/same.jpg"})

    def test_duplicate_rows_missing_sources_and_label_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean, distorted = create_labelled_schema(root)
            manifest = distorted / "distortions.csv"
            fieldnames, original = read_manifest(manifest)

            duplicate = list(original)
            duplicate.append(dict(original[0]))
            write_manifest(manifest, fieldnames, duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate manifest rows"):
                audit_prepared_robustness_data(clean, distorted, only="jpeg_30")

            missing = [
                row
                for row in original
                if not (
                    json.loads(row["parameters"]).get("quality") == 30
                    and row["class_label"] == "REAL"
                )
            ]
            write_manifest(manifest, fieldnames, missing)
            with self.assertRaisesRegex(ValueError, "Source-set mismatch"):
                audit_prepared_robustness_data(clean, distorted, only="jpeg_30")

            mismatched = [dict(row) for row in original]
            mismatched[0]["class_label"] = "REAL" if mismatched[0]["class_label"] == "FAKE" else "FAKE"
            write_manifest(manifest, fieldnames, mismatched)
            with self.assertRaisesRegex(ValueError, "class label"):
                audit_prepared_robustness_data(clean, distorted, only="jpeg_30")

    def test_unreadable_selected_file_fails_and_valid_manifest_name_mismatch_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean, distorted = create_folder_schema(root)
            manifest = distorted / "distortions.csv"
            fieldnames, rows = read_manifest(manifest)
            selected = next(
                row for row in rows if json.loads(row["parameters"]).get("quality") == 30
            )
            old_path = Path(selected["output"])
            renamed = old_path.with_name("teammate_custom_name.jpg")
            old_path.rename(renamed)
            selected["output"] = str(renamed)
            write_manifest(manifest, fieldnames, rows)

            audit = audit_prepared_robustness_data(clean, distorted, only="jpeg_30")
            self.assertTrue(any("Naming warning" in warning for warning in audit.warnings))

            renamed.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Unreadable image"):
                audit_prepared_robustness_data(clean, distorted, only="jpeg_30")

    def test_full_matrix_requires_all_conditions_but_only_requires_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean, distorted = create_folder_schema(Path(directory))
            manifest = distorted / "distortions.csv"
            fieldnames, rows = read_manifest(manifest)
            jpeg30 = [row for row in rows if json.loads(row["parameters"]).get("quality") == 30]
            write_manifest(manifest, fieldnames, jpeg30)

            audit = audit_prepared_robustness_data(clean, distorted, only="jpeg_30")
            self.assertIn("blur_2", audit.missing_conditions)
            with self.assertRaisesRegex(ValueError, "Required prepared conditions are missing"):
                audit_prepared_robustness_data(clean, distorted)

    def test_dimension_mismatch_warns_using_current_generator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean, distorted = create_folder_schema(Path(directory))
            manifest = distorted / "distortions.csv"
            _, rows = read_manifest(manifest)
            row = next(
                row for row in rows if json.loads(row["parameters"]).get("downscale") == 0.5
            )
            path = Path(row["output"])
            Image.new("RGB", (7, 7), color=(1, 2, 3)).save(path)

            audit = audit_prepared_robustness_data(clean, distorted, only="resize_05")
            self.assertTrue(any("Dimension warning" in warning for warning in audit.warnings))

    def test_missing_manifest_and_unsupported_semantics_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = create_clean(root / "validation")
            distorted = root / "validation_distorted"
            distorted.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "manifest not found"):
                audit_prepared_robustness_data(clean, distorted, only="clean")

            manifest = distorted / "distortions.csv"
            fieldnames = ["source", "output", "transform", "parameters"]
            write_manifest(
                manifest,
                fieldnames,
                [{
                    "source": str(clean / "FAKE" / "nested" / "same.jpg"),
                    "output": "unused.png",
                    "transform": "mystery_filter",
                    "parameters": "{}",
                }],
            )
            with self.assertRaisesRegex(ValueError, "unsupported robustness transform"):
                audit_prepared_robustness_data(clean, distorted, only="clean")


class RobustnessMetricAndWorkflowTests(unittest.TestCase):
    def test_metrics_use_aigc_positive_direction_and_fixed_real_threshold(self) -> None:
        metrics = calculate_robustness_metrics(
            labels=[0, 0, 1, 1],
            p_real=[0.1, 0.8, 0.9, 0.2],
            probability_threshold=0.5,
        )
        self.assertAlmostEqual(metrics["auc_aigc"], 0.75)
        self.assertEqual(
            metrics["confusion_matrix"],
            {"fake_to_fake": 1, "fake_to_real": 1, "real_to_real": 1, "real_to_fake": 1},
        )
        self.assertAlmostEqual(metrics["fake_recall"], 0.5)
        self.assertAlmostEqual(metrics["real_recall"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_only_runs_clean_and_selected_loads_once_and_writes_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean, distorted = create_folder_schema(root)
            results_root = root / "results"
            model = ConstantRealLogit()
            with (
                patch("src.robustness_matrix.load_model", return_value=model) as load_mock,
                patch(
                    "src.robustness_matrix.evaluate",
                    wraps=robustness_matrix_module.evaluate,
                ) as evaluate_mock,
            ):
                summary = run_robustness_matrix(
                    checkpoint_path=root / "model.pt",
                    validation_dir=clean,
                    distorted_dir=distorted,
                    device=torch.device("cpu"),
                    probability_threshold=0.63,
                    batch_size=2,
                    image_size=(8, 8),
                    num_workers=0,
                    run_name="Final V31",
                    only="jpeg_30",
                    results_root=results_root,
                )

            load_mock.assert_called_once()
            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["probability_threshold"] == 0.63
                    for call in evaluate_mock.call_args_list
                )
            )
            self.assertEqual(set(summary["conditions"]), {"clean", "jpeg_30"})
            self.assertEqual(summary["probability_threshold"], 0.63)
            output = results_root / "final_v31"
            expected_files = {
                "robustness_summary.json",
                "robustness_summary.csv",
                "robustness_summary.md",
                "robustness_predictions.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            predictions = json.loads((output / "robustness_predictions.json").read_text())
            self.assertEqual(set(predictions), {"clean", "jpeg_30"})
            self.assertEqual(predictions["clean"][0]["p_aigc"], 0.5)
            self.assertIn("source_id", predictions["jpeg_30"][0])
            with patch("src.robustness_matrix.load_model", return_value=model):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    run_robustness_matrix(
                        root / "model.pt", clean, distorted, torch.device("cpu"),
                        run_name="Final V31", only="clean", results_root=results_root,
                    )

    def test_cli_parsing_and_dispatch_use_default_sibling_without_existing_command_changes(self) -> None:
        parsed = app.build_parser().parse_args(
            [
                "robustness-matrix",
                "--checkpoint", "checkpoints/model.pt",
                "--validation-dir", "validation",
                "--probability-threshold", "0.63",
                "--run-name", "final_v31",
                "--only", "crop_80",
            ]
        )
        self.assertEqual(parsed.only, "crop_80")
        self.assertEqual(parsed.probability_threshold, 0.63)
        self.assertIn("robustness", app.build_parser()._actions[1].choices)
        self.assertIn("validate-bytedance", app.build_parser()._actions[1].choices)

        argv = [
            "main.py", "robustness-matrix", "--checkpoint", "checkpoints/model.pt",
            "--validation-dir", "validation", "--run-name", "final_v31", "--only", "clean",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch("main.get_device", return_value=torch.device("cpu")),
            patch("main.run_robustness_matrix") as run_mock,
        ):
            app.main()
        self.assertEqual(run_mock.call_args.kwargs["distorted_dir"], Path("validation_distorted"))
        self.assertEqual(run_mock.call_args.kwargs["only"], "clean")


if __name__ == "__main__":
    unittest.main()
