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
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.data_root = data_root.expanduser().resolve()
        self.records = load_retrieval_manifest(self.manifest_path)
        self.decode_images = decode_images
        self.page_paths = page_paths or {
            record.positive_page_id: record.image_path for record in self.records
        }
        self.hard_negative_map = hard_negative_map or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
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
        if self.decode_images:
            with Image.open(image_path) as image:
                result["positive_image"] = image.convert("RGB")
            negative_images: list[Image.Image] = []
            negative_ids = self.hard_negative_map.get(
                record.query_id,
                record.hard_negative_page_ids,
            )
            for page_id in negative_ids:
                relative = self.page_paths.get(page_id)
                if relative is None:
                    continue
                with Image.open(
                    resolve_retrieval_path(self.data_root, relative)
                ) as image:
                    negative_images.append(image.convert("RGB"))
            result["negative_images"] = negative_images
        return result


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


def retrieval_collate(items: list[dict[str, Any]]) -> RetrievalBatch:
    return RetrievalBatch(
        records=[item["record"] for item in items],
        queries=[item["query_text"] for item in items],
        positive_images=[item["positive_image"] for item in items],
        negative_images=[item.get("negative_images", []) for item in items],
    )


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
