from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset import DATASET_HANDLE, download_dataset


def create_splits(root: Path) -> Path:
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    return root


class DatasetDownloadTests(unittest.TestCase):
    def test_reuses_nested_split_when_data_dir_is_named_cifake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "cifake"
            split_root = create_splits(data_dir / "download" / "cifake")

            with patch("src.dataset.kagglehub.dataset_download") as download_mock:
                result = download_dataset(data_dir)

            self.assertEqual(result, split_root)
            download_mock.assert_not_called()

    def test_wildfake_train_test_is_never_reused_as_cifake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            (raw / "WildFake" / "FAKE" / "ADM" / "TRAIN").mkdir(parents=True)
            (raw / "WildFake" / "FAKE" / "ADM" / "TEST").mkdir(parents=True)
            downloaded = raw / "cifake" / "download" / "cifake"
            create_splits(downloaded)

            with patch("src.dataset.kagglehub.dataset_download", return_value=str(downloaded)):
                resolved = download_dataset(raw)

            self.assertEqual(resolved, downloaded)
            self.assertNotIn("WildFake", str(resolved))

    def test_reuses_existing_dataset_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "raw"
            split_root = create_splits(data_dir / "existing" / "cifake")

            with patch("src.dataset.kagglehub.dataset_download") as download_mock:
                result = download_dataset(data_dir)

            self.assertEqual(result, split_root)
            download_mock.assert_not_called()

    def test_downloads_directly_into_an_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "raw"

            def fake_download(handle: str, *, output_dir: str) -> str:
                self.assertEqual(handle, DATASET_HANDLE)
                return str(create_splits(Path(output_dir)))

            with patch(
                "src.dataset.kagglehub.dataset_download",
                side_effect=fake_download,
            ) as download_mock:
                result = download_dataset(data_dir)

            self.assertEqual(result, data_dir.resolve())
            self.assertEqual(download_mock.call_args.kwargs["output_dir"], str(data_dir.resolve()))

    def test_uses_dedicated_child_when_requested_directory_is_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "raw"
            (data_dir / "processed").mkdir(parents=True)
            expected_download_dir = data_dir.resolve() / "cifake"

            def fake_download(handle: str, *, output_dir: str) -> str:
                self.assertEqual(handle, DATASET_HANDLE)
                return str(create_splits(Path(output_dir)))

            with patch(
                "src.dataset.kagglehub.dataset_download",
                side_effect=fake_download,
            ) as download_mock:
                result = download_dataset(data_dir)

            self.assertEqual(result, expected_download_dir)
            self.assertEqual(
                download_mock.call_args.kwargs["output_dir"],
                str(expected_download_dir),
            )


if __name__ == "__main__":
    unittest.main()
