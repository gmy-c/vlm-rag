from __future__ import annotations

import importlib.util
import os
from typing import Literal

import torch


MaxSimBackend = Literal["auto", "lik", "chunked"]


def late_interaction_kernel_available() -> bool:
    return (
        importlib.util.find_spec("late_interaction_kernels") is not None
        and torch.cuda.is_available()
    )


def resolve_maxsim_backend(value: MaxSimBackend) -> Literal["lik", "chunked"]:
    if value == "lik":
        if not late_interaction_kernel_available():
            raise RuntimeError(
                "maxsim_backend='lik' requires late-interaction-kernels and CUDA. "
                "Install `colpali-engine[lik]` or use backend='chunked'."
            )
        return "lik"
    if value == "auto":
        return "lik" if late_interaction_kernel_available() else "chunked"
    if value != "chunked":
        raise ValueError(f"Unsupported MaxSim backend: {value!r}")
    return value


def maxsim_score_matrix(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    backend: MaxSimBackend = "auto",
    query_batch_chunk: int = 2,
    document_batch_chunk: int = 4,
    query_token_chunk: int = 32,
) -> torch.Tensor:
    """Return exact MaxSim scores without materialising the full BxBxLqxLd grid."""
    if queries.ndim != 3 or documents.ndim != 3:
        raise ValueError("queries/documents must be [batch, tokens, dim]")
    if queries.shape[-1] != documents.shape[-1]:
        raise ValueError("Query/document embedding dimensions differ")
    resolved = resolve_maxsim_backend(backend)
    if resolved == "lik":
        previous = os.environ.get("COLPALI_SCORES_BACKEND")
        os.environ["COLPALI_SCORES_BACKEND"] = "lik"
        try:
            from colpali_engine.utils.maxsim import maxsim_inbatch

            rows = []
            for q_start in range(0, len(queries), query_batch_chunk):
                q = queries[q_start : q_start + query_batch_chunk]
                columns = []
                for d_start in range(0, len(documents), document_batch_chunk):
                    d = documents[d_start : d_start + document_batch_chunk]
                    columns.append(maxsim_inbatch(q, d))
                rows.append(torch.cat(columns, dim=1))
            return torch.cat(rows, dim=0)
        finally:
            if previous is None:
                os.environ.pop("COLPALI_SCORES_BACKEND", None)
            else:
                os.environ["COLPALI_SCORES_BACKEND"] = previous
    return _chunked_torch_maxsim(
        queries,
        documents,
        query_batch_chunk=query_batch_chunk,
        document_batch_chunk=document_batch_chunk,
        query_token_chunk=query_token_chunk,
    )


def candidate_maxsim_scores(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    backend: MaxSimBackend = "auto",
    query_token_chunk: int = 32,
) -> torch.Tensor:
    """Score per-query candidates: query[B,Lq,D], docs[B,K,Ld,D] -> [B,K]."""
    if documents.ndim != 4 or documents.shape[0] != queries.shape[0]:
        raise ValueError("Candidate documents must be [B, K, Ld, D]")
    rows = []
    for index in range(len(queries)):
        rows.append(
            maxsim_score_matrix(
                queries[index : index + 1],
                documents[index],
                backend=backend,
                query_batch_chunk=1,
                document_batch_chunk=1,
                query_token_chunk=query_token_chunk,
            ).squeeze(0)
        )
    return torch.stack(rows)


def _chunked_torch_maxsim(
    queries: torch.Tensor,
    documents: torch.Tensor,
    *,
    query_batch_chunk: int,
    document_batch_chunk: int,
    query_token_chunk: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for q_start in range(0, len(queries), query_batch_chunk):
        q_batch = queries[q_start : q_start + query_batch_chunk]
        columns: list[torch.Tensor] = []
        for d_start in range(0, len(documents), document_batch_chunk):
            d_batch = documents[
                d_start : d_start + document_batch_chunk
            ]
            token_parts: list[torch.Tensor] = []
            for token_start in range(0, q_batch.shape[1], query_token_chunk):
                q_tokens = q_batch[
                    :, token_start : token_start + query_token_chunk
                ]
                raw = torch.einsum("bnd,csd->bcns", q_tokens, d_batch)
                token_parts.append(raw.amax(dim=-1))
            maxima = torch.cat(token_parts, dim=-1)
            valid = q_batch[:, :, 0].ne(0).to(maxima.dtype)
            scores = (maxima * valid[:, None, :]).sum(dim=-1)
            lengths = valid.sum(dim=-1).clamp_min(1)
            columns.append(scores / lengths[:, None])
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0)
