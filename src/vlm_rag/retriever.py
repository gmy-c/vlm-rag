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

    def index(
        self,
        pages: list[Page],
        *,
        batch_size: int = 2,
        storage_device: str = "cpu",
    ) -> None:
        """Encode pages in bounded batches and store one vector tensor.

        ``encode_page_batch(pages)`` previously sent the complete validation
        corpus through the ViT in one call.  A few hundred document pages are
        enough to exhaust even a large GPU, especially when hidden states are
        requested.  Chunking makes peak memory depend on ``batch_size`` rather
        than corpus size.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.pages = pages
        self._page_id_to_idx = {p.page_id: i for i, p in enumerate(pages)}

        if not pages:
            self.page_vectors = torch.empty((0, self.encoder.config.proj_dim))
            return

        self.encoder.eval()
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(pages), batch_size):
                page_batch = pages[start : start + batch_size]
                vectors = self.encoder.encode_page_batch(page_batch)
                chunks.append(vectors.detach().to(storage_device))
            self.page_vectors = torch.cat(chunks, dim=0)
            # Shape: [N, proj_dim], each row L2-normalised.

    # ═══════════════════════════════════════════════════════════
    # 在线检索
    # ═══════════════════════════════════════════════════════════

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Encode query once, compute all scores via matrix multiply, return Top-K.

        Complexity: O(N × D) where N = num pages, D = proj_dim.
        For production-scale N, replace with FAISSVectorIndex.
        """
        self.encoder.eval()
        if self.page_vectors is None:
            raise RuntimeError("Retriever index is empty; call index() first")
        with torch.no_grad():
            query_vec = self.encoder.encode_query(query)  # [proj_dim]
            query_vec = query_vec.to(self.page_vectors.device)

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
