from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.dataset import (
    DocumentUniqueBatchSampler,
    PageGroupedBatchSampler,
    RetrievalManifestDataset,
    page_grouped_retrieval_collate,
    resolve_retrieval_path,
)
from vlm_rag.retrieval.losses import (
    HybridLossConfig,
    MultiVectorMemoryQueue,
    hybrid_retrieval_loss,
    multi_positive_symmetric_cross_entropy,
    symmetric_global_info_nce,
)
from vlm_rag.retrieval.maxsim import maxsim_score_matrix
from vlm_rag.retrieval.schema import RetrievalRecord


def record(index: int, doc_id: str, page_id: str) -> RetrievalRecord:
    return RetrievalRecord(
        query_id=f"q{index}",
        query_text=f"query {index}",
        positive_page_id=page_id,
        doc_id=doc_id,
        image_path=f"docvqa_images/{page_id}.png",
        split="train",
        source_split="train",
    )


class RetrievalMultivectorTests(unittest.TestCase):
    def test_chunked_maxsim_matches_dense_reference(self) -> None:
        generator = torch.Generator().manual_seed(17)
        queries = torch.randn(3, 7, 8, generator=generator)
        documents = torch.randn(4, 11, 8, generator=generator)
        queries = torch.nn.functional.normalize(queries, dim=-1)
        documents = torch.nn.functional.normalize(documents, dim=-1)
        expected = torch.einsum(
            "bnd,csd->bcns", queries, documents
        ).amax(dim=-1).mean(dim=-1)
        actual = maxsim_score_matrix(
            queries,
            documents,
            backend="chunked",
            query_batch_chunk=2,
            document_batch_chunk=2,
            query_token_chunk=3,
        )
        torch.testing.assert_close(actual, expected)

        summed = maxsim_score_matrix(
            queries,
            documents,
            backend="chunked",
            normalization="sum",
        )
        torch.testing.assert_close(summed, expected * queries.shape[1])

    def test_multi_positive_loss_accepts_many_queries_per_page(self) -> None:
        targets = torch.tensor([0, 0, 1, 1])
        good = torch.tensor(
            [[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]]
        )
        bad = good.flip(1)
        good_loss = multi_positive_symmetric_cross_entropy(
            good, targets, temperature=1.0
        )
        bad_loss = multi_positive_symmetric_cross_entropy(
            bad, targets, temperature=1.0
        )
        self.assertLess(float(good_loss), float(bad_loss))

    def test_hybrid_loss_supports_rectangular_grouped_batch(self) -> None:
        generator = torch.Generator().manual_seed(23)
        queries = torch.nn.functional.normalize(
            torch.randn(4, 5, 8, generator=generator), dim=-1
        ).requires_grad_()
        pages = torch.nn.functional.normalize(
            torch.randn(2, 7, 8, generator=generator), dim=-1
        ).requires_grad_()
        negatives = torch.nn.functional.normalize(
            torch.randn(4, 1, 7, 8, generator=generator), dim=-1
        ).requires_grad_()
        targets = torch.tensor([0, 0, 1, 1])
        mask = torch.tensor([[True], [True], [False], [True]])
        loss, parts = hybrid_retrieval_loss(
            queries,
            pages,
            hard_negative_tokens=negatives,
            hard_negative_mask=mask,
            positive_page_indices=targets,
            config=HybridLossConfig(
                global_weight=0.0,
                maxsim_backend="chunked",
                maxsim_normalization="mean",
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("hard_negative", parts)
        loss.backward()
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(pages.grad)
        self.assertIsNotNone(negatives.grad)

    def test_document_unique_sampler_consumes_rows_once(self) -> None:
        records = [
            record(0, "a", "a1"),
            record(1, "a", "a2"),
            record(2, "b", "b1"),
            record(3, "c", "c1"),
            record(4, "d", "d1"),
            record(5, "e", "e1"),
        ]
        sampler = DocumentUniqueBatchSampler(
            records,
            batch_size=3,
            seed=4,
            drop_last=False,
        )
        batches = list(sampler)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(records))))
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            docs = [records[index].doc_id for index in batch]
            pages = [records[index].positive_page_id for index in batch]
            self.assertEqual(len(docs), len(set(docs)))
            self.assertEqual(len(pages), len(set(pages)))

    def test_page_grouped_sampler_keeps_all_queries_and_unique_docs(self) -> None:
        records = [
            record(0, "a", "a1"),
            record(1, "a", "a1"),
            record(2, "a", "a1"),
            record(3, "b", "b1"),
            record(4, "b", "b1"),
            record(5, "c", "c1"),
            record(6, "d", "d1"),
        ]
        sampler = PageGroupedBatchSampler(
            records,
            pages_per_batch=3,
            queries_per_page=2,
            seed=9,
        )
        sampler.set_epoch(2)
        batches = list(sampler)
        flattened = [index for batch in batches for index, _ in batch]
        self.assertEqual(sorted(flattened), list(range(len(records))))
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            docs = {records[index].doc_id for index, _ in batch}
            pages = {records[index].positive_page_id for index, _ in batch}
            self.assertEqual(len(docs), len(pages))
            for page_id in pages:
                count = sum(
                    records[index].positive_page_id == page_id
                    for index, _ in batch
                )
                self.assertLessEqual(count, 2)

    def test_grouped_collate_decodes_each_page_once_and_rotates_negatives(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "docvqa_images"
            image_dir.mkdir()
            for page_id in ("p1", "p2", "n1", "n2"):
                Image.new("RGB", (4, 4), color="white").save(
                    image_dir / f"{page_id}.png"
                )
            records = [
                record(0, "a", "p1"),
                record(1, "a", "p1"),
                record(2, "b", "p2"),
            ]
            dataset = RetrievalManifestDataset.__new__(
                RetrievalManifestDataset
            )
            dataset.data_root = root.resolve()
            dataset.records = records
            dataset.decode_images = False
            dataset.page_paths = {
                page_id: f"docvqa_images/{page_id}.png"
                for page_id in ("p1", "p2", "n1", "n2")
            }
            dataset.hard_negative_map = {
                "q0": ("n1", "n2"),
                "q1": ("n1", "n2"),
                "q2": ("n1", "n2"),
            }
            dataset.hard_negatives_per_query = 1
            dataset.rotate_hard_negatives = True
            first_epoch = [dataset[(index, 1)] for index in range(3)]
            second_epoch = [dataset[(index, 2)] for index in range(3)]
            self.assertEqual(first_epoch[0]["negative_page_ids"], ["n1"])
            self.assertEqual(second_epoch[0]["negative_page_ids"], ["n2"])
            batch = page_grouped_retrieval_collate(first_epoch)
            self.assertEqual(len(batch.queries), 3)
            self.assertEqual(len(batch.positive_page_positions), 2)
            self.assertEqual(len(batch.page_images), 3)
            self.assertEqual(batch.query_positive_indices, [0, 0, 1])

    def test_symmetric_global_loss_prefers_aligned_pairs(self) -> None:
        aligned = torch.eye(4)
        good = symmetric_global_info_nce(aligned, aligned, temperature=0.1)
        bad = symmetric_global_info_nce(
            aligned,
            aligned.flip(0),
            temperature=0.1,
        )
        self.assertLess(float(good), float(bad))

    def test_memory_queue_detaches_and_bounds_size(self) -> None:
        queue = MultiVectorMemoryQueue(capacity=3)
        first = torch.randn(2, 5, 4, requires_grad=True)
        second = torch.randn(2, 5, 4, requires_grad=True)
        queue.add(first)
        queue.add(second)
        stored = queue.get()
        self.assertEqual(stored.shape, (3, 5, 4))
        self.assertFalse(stored.requires_grad)

    def test_portable_paths_reject_escape(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "docvqa_images" / "one.png"
            self.assertEqual(
                resolve_retrieval_path(
                    root,
                    r"docvqa_images\one.png",
                ),
                expected.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_retrieval_path(root, "../escape.png")


if __name__ == "__main__":
    unittest.main()
