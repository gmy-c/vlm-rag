from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import torch
from tqdm.auto import tqdm

from .dataset import resolve_retrieval_path
from .maxsim import (
    MaxSimBackend,
    MaxSimNormalization,
    maxsim_score_matrix,
)
from .model import LateInteractionRetriever
from .schema import RetrievalRecord
from ..pipeline.provenance import (
    fingerprint_adapter,
    fingerprint_base_model_metadata,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class IndexedPage:
    page_id: str
    doc_id: str
    image_path: str
    shard: str
    offset: int


def unique_pages(
    records: Iterable[RetrievalRecord],
) -> list[RetrievalRecord]:
    pages: dict[str, RetrievalRecord] = {}
    for record in records:
        existing = pages.get(record.positive_page_id)
        if existing is not None and (
            existing.doc_id != record.doc_id
            or existing.image_path != record.image_path
        ):
            raise ValueError(
                f"Conflicting page mapping: {record.positive_page_id}"
            )
        pages[record.positive_page_id] = record
    return sorted(pages.values(), key=lambda item: item.positive_page_id)


@torch.no_grad()
def build_multivector_index(
    model: LateInteractionRetriever,
    records: Iterable[RetrievalRecord],
    data_root: Path,
    output_dir: Path,
    *,
    batch_size: int = 4,
    pages_per_shard: int = 128,
    manifest_path: Path | None = None,
    adapter_dir: Path | None = None,
    base_model_path: Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if batch_size < 1 or pages_per_shard < 1:
        raise ValueError("batch_size and pages_per_shard must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = unique_pages(records)
    model.eval()
    entries: list[IndexedPage] = []
    buffered_tokens: list[torch.Tensor] = []
    buffered_globals: list[torch.Tensor] = []
    buffered_records: list[RetrievalRecord] = []
    shard_number = 0

    def flush() -> None:
        nonlocal shard_number
        if not buffered_records:
            return
        shard_name = f"pages-{shard_number:05d}.pt"
        token_tensor = torch.cat(buffered_tokens, dim=0).to(
            device="cpu", dtype=torch.bfloat16
        )
        global_tensor = torch.cat(buffered_globals, dim=0).to(
            device="cpu", dtype=torch.float32
        )
        torch.save(
            {"tokens": token_tensor, "global": global_tensor},
            output_dir / shard_name,
        )
        for offset, record in enumerate(buffered_records):
            entries.append(
                IndexedPage(
                    page_id=record.positive_page_id,
                    doc_id=record.doc_id,
                    image_path=record.image_path,
                    shard=shard_name,
                    offset=offset,
                )
            )
        buffered_tokens.clear()
        buffered_globals.clear()
        buffered_records.clear()
        shard_number += 1

    starts = range(0, len(pages), batch_size)
    iterator = tqdm(
        starts,
        total=math.ceil(len(pages) / batch_size) if pages else 0,
        desc="encoding page index",
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    )
    for start in iterator:
        batch = pages[start : start + batch_size]
        images: list[Image.Image] = []
        for record in batch:
            with Image.open(
                resolve_retrieval_path(data_root, record.image_path)
            ) as image:
                images.append(image.convert("RGB"))
        tokens, global_vectors = model.encode_pages(images)
        for position, record in enumerate(batch):
            buffered_tokens.append(tokens[position : position + 1].detach())
            buffered_globals.append(
                global_vectors[position : position + 1].detach()
            )
            buffered_records.append(record)
            if len(buffered_records) >= pages_per_shard:
                flush()
        iterator.set_postfix(
            pages=min(start + len(batch), len(pages)),
            shards=shard_number,
            refresh=False,
        )
    iterator.close()
    flush()

    metadata = {
        "format_version": 2,
        "task": "retrieval_multivector_index",
        "page_count": len(entries),
        "token_dim": int(model.colpali.dim),
        "image_only_tokens": model.config.image_only_tokens,
        "model_config": {
            key: value
            for key, value in asdict(model.config).items()
            if key != "checkpoint_path"
        },
        "manifest_sha256": (
            sha256_file(manifest_path) if manifest_path is not None else None
        ),
        "adapter_sha256": (
            fingerprint_adapter(adapter_dir) if adapter_dir is not None else None
        ),
        "base_model_metadata_sha256": (
            fingerprint_base_model_metadata(base_model_path)
            if base_model_path is not None
            else None
        ),
        "pages": [
            {
                "page_id": item.page_id,
                "doc_id": item.doc_id,
                "image_path": item.image_path,
                "shard": item.shard,
                "offset": item.offset,
            }
            for item in entries
        ],
    }
    (output_dir / "index.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


class MultiVectorIndex:
    """CPU-backed index: global coarse search, exact token MaxSim reranking."""

    def __init__(
        self,
        index_dir: Path,
        *,
        expected_adapter_sha256: str | None = None,
        expected_base_model_metadata_sha256: str | None = None,
    ) -> None:
        self.index_dir = index_dir.expanduser().resolve()
        metadata = json.loads(
            (self.index_dir / "index.json").read_text(encoding="utf-8")
        )
        self.metadata = metadata
        self._validate_fingerprint(
            "adapter_sha256",
            expected_adapter_sha256,
        )
        self._validate_fingerprint(
            "base_model_metadata_sha256",
            expected_base_model_metadata_sha256,
        )
        self.pages = [IndexedPage(**item) for item in metadata["pages"]]
        self.by_page_id = {item.page_id: item for item in self.pages}
        if len(self.by_page_id) != len(self.pages):
            raise ValueError("Duplicate page_id in multi-vector index")
        global_parts = []
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}
        for shard_name in sorted({item.shard for item in self.pages}):
            payload = torch.load(
                self.index_dir / shard_name,
                map_location="cpu",
                weights_only=True,
            )
            global_parts.append(payload["global"])
        # Shards and entries are both written in sorted, contiguous order.
        self.global_vectors = torch.cat(global_parts, dim=0).float()
        if len(self.global_vectors) != len(self.pages):
            raise ValueError("Index metadata and vector count differ")
        self._position_by_page_id = {
            page.page_id: position
            for position, page in enumerate(self.pages)
        }

    def coarse_candidates(
        self,
        query_global: torch.Tensor,
        *,
        top_k: int = 128,
        excluded_page_ids: set[str] | None = None,
        excluded_doc_ids: set[str] | None = None,
    ) -> list[str]:
        scores = query_global.detach().float().cpu() @ self.global_vectors.T
        order = torch.argsort(scores.flatten(), descending=True)
        page_exclusions = excluded_page_ids or set()
        doc_exclusions = excluded_doc_ids or set()
        result: list[str] = []
        for index in order.tolist():
            page = self.pages[index]
            if (
                page.page_id in page_exclusions
                or page.doc_id in doc_exclusions
            ):
                continue
            result.append(page.page_id)
            if len(result) >= top_k:
                break
        return result

    def rerank(
        self,
        query_tokens: torch.Tensor,
        candidate_page_ids: list[str],
        *,
        backend: MaxSimBackend = "auto",
        normalization: MaxSimNormalization = "mean",
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if query_tokens.shape[0] != 1:
            raise ValueError("rerank currently accepts exactly one query")
        token_rows: list[torch.Tensor | None] = [
            None for _ in candidate_page_ids
        ]
        positions_by_shard: dict[str, list[tuple[int, int]]] = {}
        for position, page_id in enumerate(candidate_page_ids):
            page = self.by_page_id[page_id]
            positions_by_shard.setdefault(page.shard, []).append(
                (position, page.offset)
            )
        for shard, positions in positions_by_shard.items():
            payload = self._load_shard(shard)
            for position, offset in positions:
                token_rows[position] = payload["tokens"][offset].clone()
        if not token_rows:
            return []
        if any(row is None for row in token_rows):
            raise RuntimeError("Failed to load all candidate token rows")
        documents = torch.stack(
            [row for row in token_rows if row is not None]
        ).to(
            device=query_tokens.device, dtype=query_tokens.dtype
        )
        scores = maxsim_score_matrix(
            query_tokens,
            documents,
            backend=backend,
            normalization=normalization,
            query_batch_chunk=1,
            document_batch_chunk=2,
        )[0]
        ranked = sorted(
            zip(candidate_page_ids, scores.detach().float().cpu().tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k] if top_k is not None else ranked

    def coarse_scores(
        self,
        query_global: torch.Tensor,
        page_ids: list[str],
    ) -> dict[str, float]:
        if query_global.shape[0] != 1:
            raise ValueError("coarse_scores accepts exactly one query")
        positions = [self._position_by_page_id[page_id] for page_id in page_ids]
        vectors = self.global_vectors[positions]
        scores = (
            query_global.detach().float().cpu() @ vectors.T
        ).flatten().tolist()
        return {
            page_id: float(score)
            for page_id, score in zip(page_ids, scores)
        }

    def _load_shard(self, name: str) -> dict[str, torch.Tensor]:
        payload = self._shard_cache.get(name)
        if payload is None:
            payload = torch.load(
                self.index_dir / name,
                map_location="cpu",
                weights_only=True,
            )
            # Reranking must not retain the complete multi-vector corpus in
            # RAM; two shards are enough for locality without unbounded growth.
            if len(self._shard_cache) >= 2:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[name] = payload
        return payload

    def _validate_fingerprint(
        self,
        field: str,
        expected: str | None,
    ) -> None:
        if expected is None:
            return
        actual = self.metadata.get(field)
        if actual is None:
            raise ValueError(
                f"Index has no {field}; rebuild it with the current scripts"
            )
        if actual != expected:
            raise ValueError(
                f"Index/model provenance mismatch for {field}: "
                f"expected={expected}, index={actual}"
            )
