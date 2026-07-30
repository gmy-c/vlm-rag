from __future__ import annotations

from .schema import PageGenerationResult


def normalise_answer(value: str) -> str:
    return "".join(value.lower().split()).replace("%", "")


def fuse_page_results(
    results: list[PageGenerationResult],
) -> tuple[str, tuple[str, ...], float]:
    merged: dict[str, dict[str, object]] = {}
    for result in results:
        if not result.relevant or not result.answer.strip():
            continue
        key = normalise_answer(result.answer)
        score = max(0.0, result.retrieval_score) * max(
            0.0, min(1.0, result.confidence)
        )
        if key not in merged:
            merged[key] = {
                "answer": result.answer,
                "score": 0.0,
                "pages": [],
            }
        merged[key]["score"] = float(merged[key]["score"]) + score
        pages = merged[key]["pages"]
        assert isinstance(pages, list)
        pages.append(result.page_id)
    if not merged:
        return "", (), 0.0
    total = sum(float(value["score"]) for value in merged.values())
    if total <= 0:
        return "", (), 0.0
    best = max(merged.values(), key=lambda value: float(value["score"]))
    return (
        str(best["answer"]),
        tuple(str(item) for item in best["pages"]),
        min(1.0, float(best["score"]) / total),
    )
