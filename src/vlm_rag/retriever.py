"""Dual-tower retriever with Tensor-based batch computation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import Page
from .encoders import ColPaliDualEncoder, cosine_similarity


@dataclass(frozen=True)
class SearchHit:
    """A single retrieval hit: page, similarity score, and rank (1-indexed)."""

    page: Page
    score: float
    rank: int


class DualTowerRetriever:
    """Dual-tower retriever for page-image search.

    Offline (index):
        All page images are encoded once via ``encode_page_batch`` and stored
        as a single [N, D] tensor.

    Online (search):
        The user query is encoded once, then a single matrix-vector
        multiplication yields all cosine similarities in one shot.
    """

    def __init__(self, encoder: ColPaliDualEncoder) -> None:
        self.encoder = encoder
        self.pages: list[Page] = []
        self.page_vectors: torch.Tensor | None = None  # [N, proj_dim]
        self._page_id_to_idx: dict[str, int] = {}

    # ═══════════════════════════════════════════════════════════
    # 离线索引
    # ═══════════════════════════════════════════════════════════

    def index(self, pages: list[Page]) -> None:
        """Batch-encode all pages and store as a single tensor.

        Replaces the old per-page dict with one matrix multiplication at
        search time.
        """
        self.pages = pages
        self._page_id_to_idx = {p.page_id: i for i, p in enumerate(pages)}

        self.encoder.eval()
        with torch.no_grad():
            self.page_vectors = self.encoder.encode_page_batch(pages)
            # Shape: [N, proj_dim], each row L2-normalised

    # ═══════════════════════════════════════════════════════════
    # 在线检索
    # ═══════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Encode query once, compute all scores via matrix multiply, return Top-K.

        Complexity: O(N × D) where N = num pages, D = proj_dim.
        For production-scale N, replace with FAISSVectorIndex.
        """
        self.encoder.eval()
        with torch.no_grad():
            query_vec = self.encoder.encode_query(query)  # [proj_dim]

        # Inner product = cosine because both query_vec and page_vectors
        # are L2-normalised
        scores = (self.page_vectors @ query_vec).cpu().tolist()  # [N]

        # Sort descending, take top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return [
            SearchHit(
                page=self.pages[idx],
                score=round(score, 4),
                rank=rank,
            )
            for rank, (idx, score) in enumerate(indexed[:top_k], start=1)
        ]
