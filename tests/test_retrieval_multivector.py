from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.dataset import (
    DocumentUniqueBatchSampler,
    resolve_retrieval_path,
)
from vlm_rag.retrieval.losses import (
    MultiVectorMemoryQueue,
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
