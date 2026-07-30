from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.generation.client import DoubaoClientConfig, DoubaoVisionClient
from vlm_rag.generation.schema import GenerationAnswer, GenerationPage
from vlm_rag.pipeline.contracts import RetrievalHit
from vlm_rag.pipeline.engine import SecureRAGEngine
from vlm_rag.pipeline.policy import (
    RedactionRecord,
    SensitivityPolicy,
    SensitivityPolicyConfig,
)
from vlm_rag.pipeline.provenance import sha256_file
from vlm_rag.retrieval.index import MultiVectorIndex
from vlm_rag.sensitivity.catalog import (
    SensitivityCatalog,
    SensitivityCatalogEntry,
    build_catalog,
)


def catalog_entry(
    page_id: str,
    *,
    sensitive: bool | None,
    status: str = "ok",
) -> SensitivityCatalogEntry:
    return SensitivityCatalogEntry(
        page_id=page_id,
        doc_id=f"doc-{page_id}",
        image_path=f"docvqa_images/{page_id}.png",
        probability=0.9 if sensitive else 0.1,
        threshold=0.5,
        is_sensitive=sensitive,
        status=status,
    )


def hit(page_id: str, rank: int = 1) -> RetrievalHit:
    return RetrievalHit(
        page_id=page_id,
        doc_id=f"doc-{page_id}",
        image_path=f"docvqa_images/{page_id}.png",
        coarse_score=0.8,
        maxsim_score=1.0 / rank,
        rank=rank,
    )


class FakeRetriever:
    def encode_queries(self, queries):
        return torch.ones(1, 2, 3), torch.ones(1, 3)


class FakeIndex:
    def __init__(self, page_ids):
        self.page_ids = page_ids
        self.by_page_id = {
            page_id: SimpleNamespace(
                page_id=page_id,
                doc_id=f"doc-{page_id}",
                image_path=f"docvqa_images/{page_id}.png",
            )
            for page_id in page_ids
        }

    def coarse_candidates(self, query_global, *, top_k):
        return self.page_ids[:top_k]

    def coarse_scores(self, query_global, page_ids):
        return {page_id: 0.9 for page_id in page_ids}

    def rerank(self, query_tokens, candidate_ids, *, backend, top_k):
        return [
            (page_id, 1.0 / rank)
            for rank, page_id in enumerate(candidate_ids[:top_k], start=1)
        ]


class RecordingGenerator:
    def __init__(self):
        self.pages = []

    def answer(self, query, pages):
        self.pages.extend(pages)
        return GenerationAnswer(
            text="safe answer",
            evidence_page_ids=tuple(page.page_id for page in pages),
            confidence=0.9,
            page_results=(),
            errors=(),
            model="mock",
            api_calls=len(pages),
            cache_hits=0,
        )


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.responses.pop(0)


class SecurePipelineTests(unittest.TestCase):
    def test_index_rejects_adapter_fingerprint_mismatch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "adapter_sha256": "built-with-adapter-a",
                        "pages": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                MultiVectorIndex(
                    root,
                    expected_adapter_sha256="runtime-adapter-b",
                )

    def test_sensitive_original_is_never_sent_when_safe_page_exists(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "docvqa_images"
            image_dir.mkdir()
            Image.new("RGB", (2, 2), "red").save(image_dir / "sensitive.png")
            Image.new("RGB", (2, 2), "white").save(image_dir / "safe.png")
            catalog = SensitivityCatalog(
                [
                    catalog_entry("sensitive", sensitive=True),
                    catalog_entry("safe", sensitive=False),
                ]
            )
            policy = SensitivityPolicy(catalog=catalog, data_root=root)
            generator = RecordingGenerator()
            engine = SecureRAGEngine(
                retriever=FakeRetriever(),
                index=FakeIndex(["sensitive", "safe"]),
                policy=policy,
                generator=generator,
                rerank_top_k=2,
            )
            result = engine.answer("question")
            self.assertEqual(result.status, "answered")
            self.assertEqual([page.page_id for page in generator.pages], ["safe"])
            self.assertEqual(
                generator.pages[0].image_path,
                str((image_dir / "safe.png").resolve()),
            )
            sensitive_decision = result.access_decisions[0]
            self.assertEqual(sensitive_decision.action, "block")
            self.assertIsNone(sensitive_decision.selected_image_path)

    def test_approved_redacted_derivative_replaces_sensitive_original(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "docvqa_images" / "sensitive.png"
            redacted = root / "redacted" / "sensitive.png"
            original.parent.mkdir()
            redacted.parent.mkdir()
            Image.new("RGB", (2, 2), "red").save(original)
            Image.new("RGB", (2, 2), "black").save(redacted)
            policy = SensitivityPolicy(
                catalog=SensitivityCatalog(
                    [catalog_entry("sensitive", sensitive=True)]
                ),
                data_root=root,
                redactions={
                    "sensitive": RedactionRecord(
                        page_id="sensitive",
                        redacted_path="redacted/sensitive.png",
                        approved=True,
                        source_sha256=sha256_file(original),
                        redacted_sha256=sha256_file(redacted),
                    )
                },
            )
            decision = policy.decide(hit("sensitive"))
            self.assertEqual(decision.action, "allow_redacted")
            self.assertEqual(
                decision.selected_image_path,
                str(redacted.resolve()),
            )

    def test_missing_catalog_fails_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "docvqa_images" / "known.png"
            image.parent.mkdir()
            Image.new("RGB", (2, 2), "white").save(image)
            policy = SensitivityPolicy(
                catalog=SensitivityCatalog(
                    [catalog_entry("known", sensitive=False)]
                ),
                data_root=root,
            )
            decision = policy.decide(hit("unknown"))
            self.assertEqual(decision.action, "block")
            self.assertIsNone(decision.selected_image_path)

    def test_doubao_client_retries_429_then_caches_valid_json(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "page.png"
            Image.new("RGB", (2, 2), "white").save(image)
            body = {
                "id": "req-ok",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "relevant": True,
                                    "answer": "42",
                                    "evidence_quote": "42 USD",
                                    "confidence": 0.9,
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
            }
            session = FakeSession(
                [
                    FakeResponse(429, {"error": {"message": "slow down"}}),
                    FakeResponse(200, body),
                ]
            )
            sleeps = []
            client = DoubaoVisionClient(
                "secret-key",
                DoubaoClientConfig(
                    cache_dir=str(root / "cache"),
                    max_attempts=3,
                ),
                session=session,
                sleep_fn=sleeps.append,
            )
            page = GenerationPage("p1", str(image), 1.0, "allow_original")
            first = client.answer("amount?", [page])
            second = client.answer("amount?", [page])
            self.assertEqual(first.text, "42")
            self.assertEqual(first.api_calls, 2)
            self.assertEqual(second.api_calls, 0)
            self.assertEqual(second.cache_hits, 1)
            self.assertEqual(len(session.calls), 2)
            self.assertTrue(sleeps)

    def test_catalog_builder_rejects_missing_prediction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "docvqa_images"
            image_dir.mkdir()
            Image.new("RGB", (2, 2), "white").save(image_dir / "p1.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "page_id": "p1",
                        "doc_id": "d1",
                        "page_no": 1,
                        "image_path": "docvqa_images/p1.png",
                        "ocr_path": "ocr/p1.json",
                        "is_sensitive": 0,
                        "split": "test",
                        "source_split": "test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text("", encoding="utf-8")
            checkpoint = root / "best.pt"
            calibration = root / "calibration.json"
            checkpoint.write_bytes(b"checkpoint")
            calibration.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_catalog(
                    manifest_path=manifest,
                    predictions_path=predictions,
                    errors_path=None,
                    data_root=root,
                    checkpoint_path=checkpoint,
                    calibration_path=calibration,
                )


if __name__ == "__main__":
    unittest.main()
