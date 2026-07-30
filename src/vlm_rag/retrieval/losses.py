from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .maxsim import (
    MaxSimBackend,
    candidate_maxsim_scores,
    maxsim_score_matrix,
)


def symmetric_cross_entropy(
    score_matrix: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("Symmetric InfoNCE requires a square score matrix")
    labels = torch.arange(score_matrix.shape[0], device=score_matrix.device)
    logits = score_matrix / temperature
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def symmetric_global_info_nce(
    query_vectors: torch.Tensor,
    page_vectors: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    return symmetric_cross_entropy(
        query_vectors @ page_vectors.T,
        temperature=temperature,
    )


@dataclass(frozen=True, slots=True)
class HybridLossConfig:
    global_weight: float = 0.25
    late_weight: float = 0.55
    hard_negative_weight: float = 0.20
    global_temperature: float = 0.05
    late_temperature: float = 0.02
    hard_negative_margin: float = 0.05
    maxsim_backend: MaxSimBackend = "auto"
    query_batch_chunk: int = 2
    document_batch_chunk: int = 4
    query_token_chunk: int = 32


def hybrid_retrieval_loss(
    query_tokens: torch.Tensor,
    page_tokens: torch.Tensor,
    *,
    query_global: torch.Tensor | None = None,
    page_global: torch.Tensor | None = None,
    hard_negative_tokens: torch.Tensor | None = None,
    queue_tokens: torch.Tensor | None = None,
    config: HybridLossConfig = HybridLossConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = maxsim_score_matrix(
        query_tokens,
        page_tokens,
        backend=config.maxsim_backend,
        query_batch_chunk=config.query_batch_chunk,
        document_batch_chunk=config.document_batch_chunk,
        query_token_chunk=config.query_token_chunk,
    )
    late_loss = symmetric_cross_entropy(
        scores,
        temperature=config.late_temperature,
    )
    total = config.late_weight * late_loss
    parts: dict[str, torch.Tensor] = {
        "late": late_loss,
        "late_scores": scores,
    }
    if query_global is not None and page_global is not None:
        global_loss = symmetric_global_info_nce(
            query_global,
            page_global,
            temperature=config.global_temperature,
        )
        total = total + config.global_weight * global_loss
        parts["global"] = global_loss
    hard_losses: list[torch.Tensor] = []
    positive_scores = scores.diagonal()
    if hard_negative_tokens is not None:
        negative_scores = candidate_maxsim_scores(
            query_tokens,
            hard_negative_tokens,
            backend=config.maxsim_backend,
            query_token_chunk=config.query_token_chunk,
        )
        hard_losses.append(
            F.softplus(
                negative_scores
                - positive_scores.unsqueeze(1)
                + config.hard_negative_margin
            ).mean()
        )
    if queue_tokens is not None and len(queue_tokens):
        queue_scores = maxsim_score_matrix(
            query_tokens,
            queue_tokens,
            backend=config.maxsim_backend,
            query_batch_chunk=config.query_batch_chunk,
            document_batch_chunk=config.document_batch_chunk,
            query_token_chunk=config.query_token_chunk,
        )
        hardest_queue = queue_scores.max(dim=1).values
        hard_losses.append(
            F.softplus(
                hardest_queue
                - positive_scores
                + config.hard_negative_margin
            ).mean()
        )
    if hard_losses:
        hard_loss = torch.stack(hard_losses).mean()
        total = total + config.hard_negative_weight * hard_loss
        parts["hard_negative"] = hard_loss
    parts["total"] = total
    return total, parts


class MultiVectorMemoryQueue:
    def __init__(self, capacity: int = 256) -> None:
        if capacity < 0:
            raise ValueError("Queue capacity must be non-negative")
        self.capacity = capacity
        self._value: torch.Tensor | None = None

    def add(self, page_tokens: torch.Tensor) -> None:
        if self.capacity == 0:
            return
        incoming = page_tokens.detach()
        value = (
            incoming
            if self._value is None
            else torch.cat((self._value, incoming), dim=0)
        )
        self._value = value[-self.capacity :]

    def get(self) -> torch.Tensor | None:
        return self._value

    def __len__(self) -> int:
        return 0 if self._value is None else len(self._value)
