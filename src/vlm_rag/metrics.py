from __future__ import annotations

from .data import Query
from .retriever import SearchHit


def recall_at_k(queries: list[Query], ranked_hits: dict[str, list[SearchHit]], k: int) -> float:
    # Recall@K 关注正确证据页是否被召回，不关心它在 Top-K 内的具体位置。
    hits = 0
    for query in queries:
        predicted = {hit.page.page_id for hit in ranked_hits[query.query_id][:k]}
        if predicted & set(query.positive_page_ids):
            hits += 1
    return hits / len(queries)


def mrr_at_k(queries: list[Query], ranked_hits: dict[str, list[SearchHit]], k: int) -> float:
    # MRR@K 对排序更敏感：正确页面越靠前，倒数排名分越高。
    reciprocal_ranks: list[float] = []
    for query in queries:
        positives = set(query.positive_page_ids)
        rank_score = 0.0
        for hit in ranked_hits[query.query_id][:k]:
            if hit.page.page_id in positives:
                rank_score = 1.0 / hit.rank
                break
        reciprocal_ranks.append(rank_score)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def exact_match(prediction: str, answer: str) -> bool:
    return _normalize(prediction) == _normalize(answer)


def accuracy(predictions: dict[str, str], queries: list[Query]) -> float:
    correct = 0
    for query in queries:
        if exact_match(predictions[query.query_id], query.answer):
            correct += 1
    return correct / len(queries)


def _normalize(value: str) -> str:
    return "".join(value.lower().split()).replace("%", "")
