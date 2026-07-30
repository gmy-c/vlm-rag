from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .index import MultiVectorIndex
from .model import LateInteractionRetriever
from .schema import RetrievalRecord


@torch.no_grad()
def mine_hard_negatives(
    model: LateInteractionRetriever,
    index: MultiVectorIndex,
    records: Iterable[RetrievalRecord],
    output_path: Path,
    *,
    coarse_top_k: int = 128,
    negatives_per_query: int = 4,
    backend: str = "auto",
    maximum_queries: int | None = None,
) -> dict[str, Any]:
    """Mine same-split non-positive pages with coarse-to-exact retrieval."""
    selected = list(records)
    if maximum_queries is not None:
        selected = selected[:maximum_queries]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            query_tokens, query_global = model.encode_queries(
                [record.query_text]
            )
            candidates = index.coarse_candidates(
                query_global,
                top_k=coarse_top_k,
                excluded_page_ids={record.positive_page_id},
                excluded_doc_ids={record.doc_id},
            )
            ranked = index.rerank(
                query_tokens,
                candidates,
                backend=backend,
                top_k=negatives_per_query,
            )
            value = {
                "query_id": record.query_id,
                "positive_page_id": record.positive_page_id,
                "negative_page_ids": [item[0] for item in ranked],
                "scores": [item[1] for item in ranked],
                "coarse_top_k": coarse_top_k,
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            written += 1
    return {
        "queries": written,
        "negatives_per_query": negatives_per_query,
        "coarse_top_k": coarse_top_k,
        "output": str(output_path),
    }
