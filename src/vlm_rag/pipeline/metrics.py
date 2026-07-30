from __future__ import annotations

import re
import string
from typing import Iterable

from .contracts import PipelineAnswer


def normalise_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(f"[{re.escape(string.punctuation)}]", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    predicted = normalise_text(prediction)
    return float(any(predicted == normalise_text(answer) for answer in answers))


def secure_pipeline_metrics(
    rows: list[tuple[PipelineAnswer, tuple[str, ...], str]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No pipeline results were provided")
    answered = sum(result.status == "answered" for result, _, _ in rows)
    exact = sum(
        exact_match(result.answer, answers)
        for result, answers, _ in rows
        if result.status == "answered"
    )
    retrieval_hits = sum(
        positive_page_id
        in {hit.page_id for hit in result.retrieval_hits}
        for result, _, positive_page_id in rows
    )
    sensitive_original_exposure = sum(
        decision.action == "allow_original"
        and decision.sensitivity.is_sensitive is True
        for result, _, _ in rows
        for decision in result.access_decisions
    )
    missing_catalog_exposure = sum(
        decision.action.startswith("allow")
        and decision.sensitivity.catalog_status == "missing"
        for result, _, _ in rows
        for decision in result.access_decisions
    )
    api_calls = sum(
        int((result.generation or {}).get("api_calls", 0))
        for result, _, _ in rows
    )
    count = len(rows)
    return {
        "queries": count,
        "answer_rate": answered / count,
        "exact_match_on_answered": exact / max(1, answered),
        "retrieval_hit_rate": retrieval_hits / count,
        "sensitive_original_exposure": sensitive_original_exposure,
        "missing_catalog_exposure": missing_catalog_exposure,
        "api_calls": api_calls,
        "blocked_rate": sum(
            result.status == "blocked_sensitive_evidence"
            for result, _, _ in rows
        )
        / count,
    }
