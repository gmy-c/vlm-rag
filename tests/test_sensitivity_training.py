from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.metrics import binary_metrics_from_logits
from vlm_rag.sensitivity.model import SensitivityModelConfig
from vlm_rag.sensitivity.training import (
    SensitivityBatchCollator,
    SensitivityTrainingConfig,
    compute_pos_weight,
)


class SensitivityTrainingTests(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        metrics = binary_metrics_from_logits(
            torch.tensor([10.0, -10.0, 10.0, -10.0]),
            torch.tensor([1, 0, 0, 1]),
            loss=0.25,
        )
        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.true_negative, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.accuracy, 0.5)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.f1, 0.5)

    def test_model_config_round_trip(self) -> None:
        original = SensitivityModelConfig(
            checkpoint_path="checkpoint",
            selected_layers=(0, 8, 16, 23, 27),
            dropout=0.1,
            unfreeze_last_n=0,
            dtype="bfloat16",
        )
        self.assertEqual(
            SensitivityModelConfig.from_dict(original.to_dict()),
            original,
        )

    def test_training_config_validation(self) -> None:
        SensitivityTrainingConfig().validate()
        with self.assertRaisesRegex(ValueError, "batch_size"):
            SensitivityTrainingConfig(batch_size=0).validate()

    def test_pos_weight_comes_from_training_records(self) -> None:
        dataset = SimpleNamespace(
            records=[
                SimpleNamespace(is_sensitive=1),
                SimpleNamespace(is_sensitive=0),
                SimpleNamespace(is_sensitive=0),
                SimpleNamespace(is_sensitive=0),
            ]
        )
        self.assertEqual(compute_pos_weight(dataset), 3.0)

    def test_collator_does_not_expose_ocr_to_model(self) -> None:
        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                self.images = images
                self.return_tensors = return_tensors
                return {"pixel_values": torch.ones(len(images), 3, 2, 2)}

        processor = FakeProcessor()
        batch = SensitivityBatchCollator(processor)(
            [
                {
                    "image": object(),
                    "label": 1.0,
                    "page_id": "page_1",
                    "ocr_path": Path("must_not_reach_model.json"),
                }
            ]
        )
        self.assertEqual(set(batch), {"pixel_values", "labels", "page_ids"})
        self.assertNotIn("ocr_path", batch)


if __name__ == "__main__":
    unittest.main()
