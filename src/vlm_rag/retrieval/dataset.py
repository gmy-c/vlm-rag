from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import random
from typing import Any, Iterator, Sequence

from PIL import Image
from torch.utils.data import Dataset, Sampler

from .schema import RetrievalRecord, load_retrieval_manifest


def resolve_retrieval_path(data_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe retrieval manifest path: {relative_path!r}")
    root = data_root.expanduser().resolve()
    resolved = root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path escapes data root: {relative_path!r}"
        ) from exc
    return resolved


class RetrievalManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        *,
        decode_images: bool = True,
        page_paths: dict[str, str] | None = None,
        hard_negative_map: dict[str, tuple[str, ...]] | None = None,
        hard_negatives_per_query: int | None = None,
        rotate_hard_negatives: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.data_root = data_root.expanduser().resolve()
        self.records = load_retrieval_manifest(self.manifest_path)
        self.decode_images = decode_images
        self.page_paths = page_paths or {
            record.positive_page_id: record.image_path for record in self.records
        }
        self.hard_negative_map = hard_negative_map or {}
        self.hard_negatives_per_query = hard_negatives_per_query
        self.rotate_hard_negatives = rotate_hard_negatives
        if hard_negatives_per_query is not None and hard_negatives_per_query < 0:
            raise ValueError("hard_negatives_per_query must be non-negative")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        epoch = 1
        if isinstance(index, tuple):
            index, epoch = index
        record = self.records[index]
        image_path = resolve_retrieval_path(self.data_root, record.image_path)
        result: dict[str, Any] = {
            "record": record,
            "query_id": record.query_id,
            "query_text": record.query_text,
            "positive_page_id": record.positive_page_id,
            "doc_id": record.doc_id,
            "image_path": image_path,
        }
        negative_ids = self._negative_ids(record, epoch)
        negative_paths: list[Path] = []
        existing_negative_ids: list[str] = []
        for page_id in negative_ids:
            relative = self.page_paths.get(page_id)
            if relative is None:
                raise KeyError(
                    f"Hard-negative page is absent from page catalog: {page_id}"
                )
            existing_negative_ids.append(page_id)
            negative_paths.append(
                resolve_retrieval_path(self.data_root, relative)
            )
        result["negative_page_ids"] = existing_negative_ids
        result["negative_image_paths"] = negative_paths
        if self.decode_images:
            with Image.open(image_path) as image:
                result["positive_image"] = image.convert("RGB")
            negative_images: list[Image.Image] = []
            for negative_path in negative_paths:
                with Image.open(negative_path) as image:
                    negative_images.append(image.convert("RGB"))
            result["negative_images"] = negative_images
        return result

    def _negative_ids(
        self,
        record: RetrievalRecord,
        epoch: int,
    ) -> tuple[str, ...]:
        values = self.hard_negative_map.get(
            record.query_id,
            record.hard_negative_page_ids,
        )
        limit = self.hard_negatives_per_query
        if limit is None or limit >= len(values):
            return tuple(values)
        if limit == 0 or not values:
            return ()
        start = 0
        if self.rotate_hard_negatives:
            start = ((max(1, epoch) - 1) * limit) % len(values)
        return tuple(
            values[(start + offset) % len(values)]
            for offset in range(limit)
        )


def load_hard_negative_map(path: Path) -> dict[str, tuple[str, ...]]:
    import json

    result: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            query_id = str(value["query_id"])
            if query_id in result:
                raise ValueError(
                    f"Duplicate hard-negative query at {path}:{line_number}"
                )
            result[query_id] = tuple(
                str(item) for item in value["negative_page_ids"]
            )
    return result


@dataclass(slots=True)
class RetrievalBatch:
    records: list[RetrievalRecord]
    queries: list[str]
    positive_images: list[Image.Image]
    negative_images: list[list[Image.Image]]


@dataclass(slots=True)
class PageGroupedRetrievalBatch:
    records: list[RetrievalRecord]
    queries: list[str]
    page_images: list[Image.Image]
    page_ids: list[str]
    positive_page_positions: list[int]
    query_positive_indices: list[int]
    negative_page_positions: list[list[int]]
    negative_mask: list[list[bool]]


def retrieval_collate(items: list[dict[str, Any]]) -> RetrievalBatch:
    return RetrievalBatch(
        records=[item["record"] for item in items],
        queries=[item["query_text"] for item in items],
        positive_images=[item["positive_image"] for item in items],
        negative_images=[item.get("negative_images", []) for item in items],
    )


def page_grouped_retrieval_collate(
    items: list[dict[str, Any]],
) -> PageGroupedRetrievalBatch:
    """Decode every distinct positive/negative page once per micro batch."""
    page_paths: dict[str, Path] = {}
    positive_ids: list[str] = []
    for item in items:
        page_id = str(item["positive_page_id"])
        if page_id not in page_paths:
            positive_ids.append(page_id)
            page_paths[page_id] = Path(item["image_path"])
    positive_index = {
        page_id: index for index, page_id in enumerate(positive_ids)
    }
    negative_ids_by_query: list[list[str]] = []
    for item in items:
        current: list[str] = []
        for page_id, path in zip(
            item.get("negative_page_ids", []),
            item.get("negative_image_paths", []),
        ):
            page_id = str(page_id)
            if page_id == str(item["positive_page_id"]):
                raise ValueError(
                    f"Hard negative collides with positive page: {page_id}"
                )
            page_paths.setdefault(page_id, Path(path))
            current.append(page_id)
        negative_ids_by_query.append(current)
    page_ids = list(page_paths)
    position = {page_id: index for index, page_id in enumerate(page_ids)}
    images: list[Image.Image] = []
    for page_id in page_ids:
        with Image.open(page_paths[page_id]) as image:
            images.append(image.convert("RGB"))
    width = max((len(row) for row in negative_ids_by_query), default=0)
    negative_positions: list[list[int]] = []
    negative_mask: list[list[bool]] = []
    for row in negative_ids_by_query:
        negative_positions.append(
            [position[page_id] for page_id in row] + [-1] * (width - len(row))
        )
        negative_mask.append(
            [True] * len(row) + [False] * (width - len(row))
        )
    return PageGroupedRetrievalBatch(
        records=[item["record"] for item in items],
        queries=[str(item["query_text"]) for item in items],
        page_images=images,
        page_ids=page_ids,
        positive_page_positions=[position[page_id] for page_id in positive_ids],
        query_positive_indices=[
            positive_index[str(item["positive_page_id"])] for item in items
        ],
        negative_page_positions=negative_positions,
        negative_mask=negative_mask,
    )


class PageGroupedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Pack several queries for each page while keeping documents distinct."""

    def __init__(
        self,
        records: Sequence[RetrievalRecord],
        pages_per_batch: int,
        queries_per_page: int,
        *,
        seed: int = 42,
    ) -> None:
        if pages_per_batch < 2:
            raise ValueError("pages_per_batch must be >= 2")
        if queries_per_page < 1:
            raise ValueError("queries_per_page must be >= 1")
        self.records = records
        self.pages_per_batch = pages_per_batch
        self.queries_per_page = queries_per_page
        self.seed = seed
        self.epoch = 1
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[tuple[int, int]]] = []

    def set_epoch(self, epoch: int) -> None:
        if self.epoch == epoch:
            return
        self.epoch = epoch
        self._cached_epoch = None
        self._cached_batches = []

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> list[list[tuple[int, int]]]:
        if self._cached_epoch == self.epoch:
            return self._cached_batches
        rng = random.Random(self.seed + self.epoch)
        by_page: dict[str, list[int]] = {}
        page_docs: dict[str, str] = {}
        for index, record in enumerate(self.records):
            by_page.setdefault(record.positive_page_id, []).append(index)
            existing = page_docs.setdefault(record.positive_page_id, record.doc_id)
            if existing != record.doc_id:
                raise ValueError(
                    f"Page maps to multiple documents: {record.positive_page_id}"
                )
        for indices in by_page.values():
            rng.shuffle(indices)
        batches: list[list[tuple[int, int]]] = []
        while any(by_page.values()):
            available = [page_id for page_id, rows in by_page.items() if rows]
            rng.shuffle(available)
            selected: list[str] = []
            documents: set[str] = set()
            for page_id in available:
                doc_id = page_docs[page_id]
                if doc_id in documents:
                    continue
                selected.append(page_id)
                documents.add(doc_id)
                if len(selected) == self.pages_per_batch:
                    break
            if not selected:
                raise RuntimeError("Could not construct a page-grouped batch")
            batch: list[tuple[int, int]] = []
            for page_id in selected:
                rows = by_page[page_id]
                count = min(self.queries_per_page, len(rows))
                batch.extend((rows.pop(), self.epoch) for _ in range(count))
            batches.append(batch)
        self._cached_epoch = self.epoch
        self._cached_batches = batches
        return batches


class DocumentUniqueBatchSampler(Sampler[list[int]]):
    """Yield batches without repeated positive pages or documents."""

    def __init__(
        self,
        records: Sequence[RetrievalRecord],
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 2:
            raise ValueError("Retrieval batch_size must be >= 2")
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        by_doc: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            by_doc.setdefault(record.doc_id, []).append(index)
        for indices in by_doc.values():
            rng.shuffle(indices)

        # A document can reappear in later batches, but never twice in one
        # batch. This consumes every manifest row at most once and avoids the
        # repeated-list extension used by the older sampler.
        remaining = sum(len(indices) for indices in by_doc.values())
        while remaining:
            available_docs = [
                doc_id for doc_id, indices in by_doc.items() if indices
            ]
            rng.shuffle(available_docs)
            batch: list[int] = []
            pages: set[str] = set()
            for doc_id in available_docs:
                indices = by_doc[doc_id]
                selected_position = next(
                    (
                        position
                        for position, index in enumerate(indices)
                        if self.records[index].positive_page_id not in pages
                    ),
                    None,
                )
                if selected_position is None:
                    continue
                index = indices.pop(selected_position)
                remaining -= 1
                batch.append(index)
                pages.add(self.records[index].positive_page_id)
                if len(batch) == self.batch_size:
                    break
            if len(batch) == self.batch_size:
                yield batch
            elif batch and not self.drop_last:
                yield batch
            elif not batch:
                raise RuntimeError(
                    "Could not construct a document-unique retrieval batch"
                )

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.records) // self.batch_size
        return (len(self.records) + self.batch_size - 1) // self.batch_size
