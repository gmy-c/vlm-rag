from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AccessAction = Literal["allow_original", "allow_redacted", "block", "error"]
PipelineStatus = Literal[
    "answered",
    "no_retrieval_hits",
    "blocked_sensitive_evidence",
    "generation_failed",
    "unable_to_answer",
]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    page_id: str
    doc_id: str
    image_path: str
    coarse_score: float | None
    maxsim_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SensitivityDecision:
    page_id: str
    probability: float | None
    threshold: float | None
    is_sensitive: bool | None
    catalog_status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PageAccessDecision:
    hit: RetrievalHit
    sensitivity: SensitivityDecision
    action: AccessAction
    selected_image_path: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit.to_dict(),
            "sensitivity": self.sensitivity.to_dict(),
            "action": self.action,
            "selected_image_path": self.selected_image_path,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PipelineAnswer:
    request_id: str
    status: PipelineStatus
    answer: str
    confidence: float
    evidence_page_ids: tuple[str, ...]
    retrieval_hits: tuple[RetrievalHit, ...]
    access_decisions: tuple[PageAccessDecision, ...]
    generation: dict[str, Any] | None
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "status": self.status,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence_page_ids": list(self.evidence_page_ids),
            "retrieval_hits": [item.to_dict() for item in self.retrieval_hits],
            "access_decisions": [
                item.to_dict() for item in self.access_decisions
            ],
            "generation": self.generation,
            "errors": list(self.errors),
            "latency_ms": self.latency_ms,
        }
