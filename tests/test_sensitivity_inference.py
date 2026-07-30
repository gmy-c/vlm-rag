from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.inference import (
    InferenceError,
    InferenceItem,
    PredictionResult,
    items_from_manifest,
    run_batched_inference,
)
from vlm_rag.sensitivity.metrics import (
    binary_metrics_from_probabilities,
    calibrate_thresholds,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.maximum_batch = 0
        self.calls = 0

    def __call__(self, *, images, return_tensors):
        self.maximum_batch = max(self.maximum_batch, len(images))
        self.calls += 1
        values = [image.getpixel((0, 0))[0] / 255.0 for image in images]
        return {"pixel_values": torch.tensor(values).view(-1, 1)}


class FakeClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.image_processor = FakeProcessor()

    def forward(self, pixel_values):
        return (pixel_values[:, 0] - 0.5) * 8.0 + self.anchor * 0


class SensitivityInferenceTests(unittest.TestCase):
    def test_auc_confusion_matrix_and_threshold_calibration(self) -> None:
        probabilities = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        metrics = binary_metrics_from_probabilities(
            probabilities,
            labels,
            threshold=0.5,
        )
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.false_negative_rate, 0.0)
        self.assertEqual(metrics.pr_auc, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertEqual(
            metrics.to_dict()["confusion_matrix"],
            {
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
                "true_positive": 2,
            },
        )
        calibration = calibrate_thresholds(
            probabilities,
            labels,
            target_recall=0.9,
        )
        self.assertEqual(calibration["default"]["threshold"], 0.5)
        self.assertEqual(calibration["best_f1"]["threshold"], 0.8)
        self.assertEqual(calibration["target_recall"]["threshold"], 0.8)

    def test_batch_one_and_batched_predictions_match(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, value in enumerate((20, 80, 160, 240)):
                path = root / f"page_{index}.png"
                Image.new("RGB", (3, 3), (value, value, value)).save(path)
                paths.append(path)
            items = [
                InferenceItem(path.stem, path, str(path))
                for path in paths
            ]
            model = FakeClassifier()
            single = [
                event
                for event in run_batched_inference(model, items, batch_size=1)
                if isinstance(event, PredictionResult)
            ]
            batched = [
                event
                for event in run_batched_inference(model, items, batch_size=3)
                if isinstance(event, PredictionResult)
            ]
            self.assertEqual(
                [result.page_id for result in single],
                [result.page_id for result in batched],
            )
            for left, right in zip(single, batched):
                self.assertAlmostEqual(left.probability, right.probability, places=7)

    def test_corrupt_image_is_audited_and_other_items_continue(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.png"
            invalid = root / "invalid.png"
            Image.new("RGB", (2, 2), "white").save(valid)
            invalid.write_bytes(b"not an image")
            events = list(
                run_batched_inference(
                    FakeClassifier(),
                    [
                        InferenceItem("bad", invalid, str(invalid)),
                        InferenceItem("good", valid, str(valid)),
                    ],
                    batch_size=2,
                )
            )
            self.assertIsInstance(events[0], InferenceError)
            self.assertIsInstance(events[1], PredictionResult)

    def test_total_item_count_does_not_change_inference_batch_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "page.png"
            Image.new("RGB", (2, 2), "white").save(path)
            model = FakeClassifier()
            items = [
                InferenceItem(f"page_{index}", path, str(path))
                for index in range(23)
            ]
            events = list(run_batched_inference(model, items, batch_size=4))
            self.assertEqual(len(events), 23)
            self.assertEqual(model.image_processor.maximum_batch, 4)
            self.assertEqual(model.image_processor.calls, 6)

    def test_manifest_posix_path_resolves_under_new_data_root(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "docvqa_images"
            image_dir.mkdir()
            image = image_dir / "portable_1.png"
            Image.new("RGB", (2, 2), "white").save(image)
            manifest = root / "one.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "page_id": "portable_1",
                        "doc_id": "portable",
                        "page_no": 1,
                        "image_path": "docvqa_images/portable_1.png",
                        "ocr_path": "ocr/portable_1.json",
                        "is_sensitive": 1,
                        "split": "test",
                        "source_split": "test",
                        "label_source": "desensitized_membership",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = next(items_from_manifest(manifest, root))
            self.assertEqual(item.resolved_path, image.resolve())
            self.assertEqual(item.input_path, "docvqa_images/portable_1.png")


if __name__ == "__main__":
    unittest.main()
