from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.dataset import SensitivityManifestDataset, resolve_data_path
from vlm_rag.sensitivity.manifest import ManifestBuildError, build_sensitivity_manifests
from vlm_rag.sensitivity.schema import load_manifest


class SensitivityManifestTests(unittest.TestCase):
    def test_builds_portable_leak_free_manifests(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            self._make_dataset(data_root)
            result = build_sensitivity_manifests(
                data_root,
                seed=7,
                verify_images=False,
                validate_ocr_json=True,
            )

            self.assertEqual(result.summary["total"]["pages"], 6)
            self.assertEqual(result.summary["total"]["positive"], 2)
            self.assertEqual(result.summary["total"]["negative"], 4)
            self.assertTrue((result.output_dir / "all.jsonl").is_file())
            self.assertTrue((result.output_dir / "summary.json").is_file())

            records = load_manifest(result.output_dir / "all.jsonl")
            self.assertEqual(len(records), 6)
            self.assertEqual({record.page_id for record in records}, {f"d{i}_1" for i in range(6)})
            self.assertTrue(all("\\" not in record.image_path for record in records))
            self.assertTrue(all(not Path(record.image_path).is_absolute() for record in records))

            split_docs = {
                split: {record.doc_id for record in records if record.split == split}
                for split in ("train", "val", "test")
            }
            self.assertFalse(split_docs["train"] & split_docs["val"])
            self.assertFalse(split_docs["train"] & split_docs["test"])
            self.assertFalse(split_docs["val"] & split_docs["test"])

            dataset = SensitivityManifestDataset(
                result.output_dir / "all.jsonl",
                data_root,
                load_images=False,
            )
            self.assertEqual(len(dataset), 6)
            self.assertTrue(dataset[0]["image_path"].is_file())
            self.assertIn(dataset[0]["label"], (0.0, 1.0))

            repeated = build_sensitivity_manifests(
                data_root,
                seed=7,
                verify_images=False,
                validate_ocr_json=True,
            )
            self.assertEqual(
                {record.page_id: record.split for record in result.records},
                {record.page_id: record.split for record in repeated.records},
            )

    def test_positive_must_be_a_full_dataset_member(self) -> None:
        with TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            self._make_dataset(data_root)
            self._touch(data_root / "desensitized" / "docvqa_images" / "outside_1.png")
            self._write_json(data_root / "desensitized" / "ocr" / "outside_1.json", {})
            with self.assertRaisesRegex(ManifestBuildError, "positive_images_not_in_full"):
                build_sensitivity_manifests(
                    data_root,
                    verify_images=False,
                    validate_ocr_json=True,
                )

    def test_rejects_manifest_path_traversal(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
                resolve_data_path(Path(temporary), "../secret.png")

    def _make_dataset(self, root: Path) -> None:
        for relative in (
            "docvqa_images",
            "ocr",
            "docvqa_extracted",
            "desensitized/docvqa_images",
            "desensitized/ocr",
            "desensitized/docvqa_extracted",
        ):
            (root / relative).mkdir(parents=True)

        rows_by_split: dict[str, list[dict[str, object]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        positive_rows_by_split: dict[str, list[dict[str, object]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for index in range(6):
            page_id = f"d{index}_1"
            source_split = ("train", "val", "test")[index % 3]
            self._touch(root / "docvqa_images" / f"{page_id}.png")
            self._write_json(root / "ocr" / f"{page_id}.json", {"status": "Succeeded"})
            qa_row = {
                "questionId": index,
                "question": f"q{index}",
                "image": f"documents/{page_id}.png",
                "docId": index,
                "ucsf_document_id": f"d{index}",
                "ucsf_document_page_no": "1",
                "data_split": source_split,
            }
            rows_by_split[source_split].append(qa_row)
            if index in (0, 4):
                self._touch(root / "desensitized" / "docvqa_images" / f"{page_id}.png")
                self._write_json(
                    root / "desensitized" / "ocr" / f"{page_id}.json",
                    {"status": "Succeeded"},
                )
                positive_rows_by_split[source_split].append(qa_row)

        filenames = {
            "train": "train_v1.0_withQT.json",
            "val": "val_v1.0_withQT.json",
            "test": "test_v1.0.json",
        }
        for split, filename in filenames.items():
            self._write_json(
                root / "docvqa_extracted" / filename,
                {"dataset_split": split, "data": rows_by_split[split]},
            )
            self._write_json(
                root / "desensitized" / "docvqa_extracted" / filename,
                {"dataset_split": split, "data": positive_rows_by_split[split]},
            )

    @staticmethod
    def _touch(path: Path) -> None:
        path.write_bytes(b"not-decoded-in-these-tests")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
