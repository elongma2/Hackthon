from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.data_preparation import (
    _validate_move_plan,
    prepare_wildfake_data,
)


def write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic test image")
    return path


def relative_images(source: Path, split: str) -> set[str]:
    split_root = source / split
    if not split_root.exists():
        return set()
    return {
        path.relative_to(split_root).as_posix()
        for path in split_root.rglob("*")
        if path.is_file()
    }


class DataPreparationTests(unittest.TestCase):
    def test_direct_nested_extensions_ratio_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "WildFake"
            source = root / "FAKE" / "FutureGenerator"
            extensions = [".jpg", ".JPEG", ".png", ".WEBP"]
            for index in range(10):
                extension = extensions[index % len(extensions)]
                parent = source if index == 0 else source / "wrapper" / "deep"
                write_image(parent / f"image_{index}{extension}")
            ignored = source / "wrapper" / "deep" / "notes.txt"
            ignored.write_text("keep me", encoding="utf-8")
            (root / "REAL").mkdir(parents=True)

            results = prepare_wildfake_data(root)

            self.assertEqual(results[0].label, "FAKE")
            self.assertEqual(results[0].source, "FutureGenerator")
            self.assertEqual(results[0].train_count, 9)
            self.assertEqual(results[0].test_count, 1)
            self.assertEqual(
                len(relative_images(source, "TRAIN") | relative_images(source, "TEST")),
                10,
            )
            self.assertTrue(ignored.exists())
            self.assertTrue(
                any(path.startswith("wrapper/deep/") for path in relative_images(source, "TRAIN"))
            )

    def test_custom_ratio_is_deterministic_and_preserves_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            roots = [Path(first) / "WildFake", Path(second) / "WildFake"]
            for root in roots:
                source = root / "REAL" / "Flickr"
                for index in range(10):
                    write_image(source / "a" / f"image_{index}.jpg")
                write_image(source / "b" / "duplicate.png")
                write_image(source / "c" / "duplicate.png")
                (root / "FAKE").mkdir(parents=True)
                prepare_wildfake_data(root, train_ratio=0.5, seed=42)

            first_source = roots[0] / "REAL" / "Flickr"
            second_source = roots[1] / "REAL" / "Flickr"
            self.assertEqual(relative_images(first_source, "TRAIN"), relative_images(second_source, "TRAIN"))
            self.assertEqual(relative_images(first_source, "TEST"), relative_images(second_source, "TEST"))
            self.assertEqual(len(relative_images(first_source, "TRAIN")), 6)
            all_relative = relative_images(first_source, "TRAIN") | relative_images(first_source, "TEST")
            self.assertIn("b/duplicate.png", all_relative)
            self.assertIn("c/duplicate.png", all_relative)

    def test_existing_top_level_split_is_preserved_and_second_run_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "WildFake"
            source = root / "FAKE" / "ADM"
            train = write_image(source / "TRAIN" / "official_train.jpg")
            test = write_image(source / "TEST" / "official_test.jpg")
            (root / "REAL").mkdir(parents=True)

            first = prepare_wildfake_data(root)
            second = prepare_wildfake_data(root)

            self.assertEqual(first[0].status, "skipped")
            self.assertEqual(second[0].status, "skipped")
            self.assertTrue(train.exists())
            self.assertTrue(test.exists())
            self.assertEqual(relative_images(source, "TRAIN"), {"official_train.jpg"})

    def test_clear_nested_official_split_is_promoted_without_reshuffling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "WildFake"
            source = root / "FAKE" / "DDPM"
            write_image(source / "download" / "DDPM_dataset" / "TRAIN" / "cats" / "a.jpg")
            write_image(source / "download" / "DDPM_dataset" / "TEST" / "dogs" / "b.png")
            (root / "REAL").mkdir(parents=True)

            result = prepare_wildfake_data(root)[0]

            self.assertEqual(result.status, "promoted")
            self.assertTrue(source.joinpath("TRAIN", "download", "DDPM_dataset", "cats", "a.jpg").exists())
            self.assertTrue(source.joinpath("TEST", "download", "DDPM_dataset", "dogs", "b.png").exists())
            self.assertFalse(source.joinpath("TEST", "download", "DDPM_dataset", "cats", "a.jpg").exists())

    def test_partial_ambiguous_and_stray_split_structures_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "WildFake"
            partial = root / "FAKE" / "Partial"
            partial_image = write_image(partial / "TRAIN" / "a.jpg")
            stray = root / "REAL" / "Stray"
            write_image(stray / "TRAIN" / "a.jpg")
            write_image(stray / "TEST" / "b.jpg")
            stray_image = write_image(stray / "outside.jpg")
            ambiguous = root / "REAL" / "Ambiguous"
            for wrapper in ("one", "two"):
                write_image(ambiguous / wrapper / "TRAIN" / "a.jpg")
                write_image(ambiguous / wrapper / "TEST" / "b.jpg")

            results = prepare_wildfake_data(root)

            self.assertTrue(all(result.status == "warning" for result in results))
            self.assertTrue(partial_image.exists())
            self.assertTrue(stray_image.exists())
            self.assertFalse((partial / "TEST").exists())

    def test_missing_empty_one_image_and_cifake_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            missing = raw / "WildFake"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(prepare_wildfake_data(missing), [])
            self.assertIn("WildFake dataset directory not found", output.getvalue())

            empty = missing / "FAKE" / "Empty"
            empty.mkdir(parents=True)
            one = write_image(missing / "REAL" / "One" / "only.jpg")
            cifake = write_image(raw / "cifake" / "train" / "FAKE" / "sentinel.jpg")
            results = prepare_wildfake_data(missing)

            self.assertTrue(all(result.status == "warning" for result in results))
            self.assertTrue(one.exists())
            self.assertTrue(cifake.exists())
            self.assertFalse((one.parent / "TRAIN").exists())

    def test_invalid_ratio_and_move_plan_collisions_fail_before_moving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                prepare_wildfake_data(root, train_ratio=1.0)

            first = write_image(root / "first.jpg")
            second = write_image(root / "second.jpg")
            destination = root / "TRAIN" / "same.jpg"
            with self.assertRaises(FileExistsError):
                _validate_move_plan([(first, destination), (second, destination)])
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())


if __name__ == "__main__":
    unittest.main()
