from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationPage:
    page_id: str
    image_path: str
    retrieval_score: float
    source: str


@dataclass(frozen=True, slots=True)
class PageGenerationResult:
    page_id: str
    relevant: bool
    answer: str
    evidence: str
    confidence: float
    retrieval_score: float
    source: str
    cached: bool = False
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationError:
    page_id: str
    error_type: str
    message: str
    retryable: bool
    status_code: int | None = None
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationAnswer:
    text: str
    evidence_page_ids: tuple[str, ...]
    confidence: float
    page_results: tuple[PageGenerationResult, ...]
    errors: tuple[GenerationError, ...]
    model: str
    api_calls: int
    cache_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_page_ids": list(self.evidence_page_ids),
            "confidence": self.confidence,
            "page_results": [item.to_dict() for item in self.page_results],
            "errors": [item.to_dict() for item in self.errors],
            "model": self.model,
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
        }
