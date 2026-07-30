from __future__ import annotations

import time
import uuid

import torch

from .audit import JsonlAuditWriter
from .contracts import PipelineAnswer, RetrievalHit
from .policy import SensitivityPolicy
from ..generation import DoubaoVisionClient, GenerationPage
from ..retrieval.index import MultiVectorIndex
from ..retrieval.model import LateInteractionRetriever


class SecureRAGEngine:
    def __init__(
        self,
        *,
        retriever: LateInteractionRetriever,
        index: MultiVectorIndex,
        policy: SensitivityPolicy,
        generator: DoubaoVisionClient,
        coarse_top_k: int = 128,
        rerank_top_k: int = 20,
        maxsim_backend: str = "auto",
        audit_writer: JsonlAuditWriter | None = None,
    ) -> None:
        if coarse_top_k < rerank_top_k:
            raise ValueError("coarse_top_k must be >= rerank_top_k")
        self.retriever = retriever
        self.index = index
        self.policy = policy
        self.generator = generator
        self.coarse_top_k = coarse_top_k
        self.rerank_top_k = rerank_top_k
        self.maxsim_backend = maxsim_backend
        self.audit_writer = audit_writer

    @torch.no_grad()
    def answer(self, query: str) -> PipelineAnswer:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        query_tokens, query_global = self.retriever.encode_queries([query])
        candidate_ids = self.index.coarse_candidates(
            query_global,
            top_k=self.coarse_top_k,
        )
        if not candidate_ids:
            result = self._result(
                request_id=request_id,
                status="no_retrieval_hits",
                started=started,
            )
            self._audit(query, result)
            return result
        coarse_scores = self.index.coarse_scores(query_global, candidate_ids)
        ranking = self.index.rerank(
            query_tokens,
            candidate_ids,
            backend=self.maxsim_backend,
            top_k=self.rerank_top_k,
        )
        hits = [
            RetrievalHit(
                page_id=page_id,
                doc_id=self.index.by_page_id[page_id].doc_id,
                image_path=self.index.by_page_id[page_id].image_path,
                coarse_score=coarse_scores.get(page_id),
                maxsim_score=float(score),
                rank=rank,
            )
            for rank, (page_id, score) in enumerate(ranking, start=1)
        ]
        decisions, selected = self.policy.apply(hits)
        if not selected:
            result = self._result(
                request_id=request_id,
                status="blocked_sensitive_evidence",
                started=started,
                hits=hits,
                decisions=decisions,
            )
            self._audit(query, result)
            return result
        pages = [
            GenerationPage(
                page_id=decision.hit.page_id,
                image_path=str(decision.selected_image_path),
                retrieval_score=decision.hit.maxsim_score,
                source=decision.action,
            )
            for decision in selected
        ]
        generation = self.generator.answer(query, pages)
        if generation.text:
            status = "answered"
        elif generation.errors and not generation.page_results:
            status = "generation_failed"
        else:
            status = "unable_to_answer"
        result = self._result(
            request_id=request_id,
            status=status,
            started=started,
            answer=generation.text,
            confidence=generation.confidence,
            evidence=generation.evidence_page_ids,
            hits=hits,
            decisions=decisions,
            generation=generation.to_dict(),
            errors=[error.to_dict() for error in generation.errors],
        )
        self._audit(query, result)
        return result

    @staticmethod
    def _result(
        *,
        request_id: str,
        status: str,
        started: float,
        answer: str = "",
        confidence: float = 0.0,
        evidence: tuple[str, ...] = (),
        hits: list[RetrievalHit] | None = None,
        decisions=None,
        generation=None,
        errors=None,
    ) -> PipelineAnswer:
        return PipelineAnswer(
            request_id=request_id,
            status=status,
            answer=answer,
            confidence=confidence,
            evidence_page_ids=evidence,
            retrieval_hits=tuple(hits or ()),
            access_decisions=tuple(decisions or ()),
            generation=generation,
            errors=tuple(errors or ()),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _audit(self, query: str, result: PipelineAnswer) -> None:
        if self.audit_writer is not None:
            self.audit_writer.write(query, result)
