from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any

from vlm_rag.sensitivity.schema import load_manifest as load_sensitivity_manifest

from .schema import (
    RetrievalRecord,
    write_retrieval_manifest,
)


QA_FILES = (
    "train_v1.0_withQT.json",
    "val_v1.0_withQT.json",
    "test_v1.0.json",
)


@dataclass(frozen=True, slots=True)
class RetrievalManifestBuildResult:
    output_dir: Path
    records: int
    pages: int
    documents: int
    split_counts: dict[str, int]


def build_retrieval_manifests(
    data_root: Path,
    output_dir: Path | None = None,
    *,
    sensitivity_manifest_dir: Path | None = None,
) -> RetrievalManifestBuildResult:
    data_root = data_root.expanduser().resolve()
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else data_root / "manifests" / "retrieval"
    )
    sensitivity_manifest_dir = (
        sensitivity_manifest_dir.expanduser().resolve()
        if sensitivity_manifest_dir is not None
        else data_root / "manifests" / "sensitivity"
    )

    doc_split: dict[str, str] = {}
    page_paths: dict[str, str] = {}
    page_docs: dict[str, str] = {}
    for split in ("train", "val", "test"):
        source = sensitivity_manifest_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing canonical document split manifest: {source}"
            )
        for record in load_sensitivity_manifest(source):
            existing = doc_split.setdefault(record.doc_id, split)
            if existing != split:
                raise ValueError(
                    f"Document {record.doc_id!r} occurs in {existing} and {split}"
                )
            page_paths[record.page_id] = record.image_path
            page_docs[record.page_id] = record.doc_id

    records: list[RetrievalRecord] = []
    seen_query_ids: set[str] = set()
    anomalies: list[dict[str, Any]] = []
    qa_root = data_root / "docvqa_extracted"
    for filename in QA_FILES:
        path = qa_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing DocVQA QA file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise ValueError(f"{path} has no list-valued data field")
        for entry in entries:
            query_id = f"docvqa_{entry['questionId']}"
            image_name = PurePosixPath(str(entry["image"])).name
            page_id = Path(image_name).stem
            doc_id = str(entry["ucsf_document_id"])
            source_split = str(entry.get("data_split", payload.get("dataset_split", "")))
            if query_id in seen_query_ids:
                anomalies.append(
                    {"type": "duplicate_query_id", "query_id": query_id}
                )
                continue
            seen_query_ids.add(query_id)
            if page_id not in page_paths:
                anomalies.append(
                    {
                        "type": "page_not_in_canonical_manifest",
                        "query_id": query_id,
                        "page_id": page_id,
                    }
                )
                continue
            if page_docs[page_id] != doc_id:
                anomalies.append(
                    {
                        "type": "document_mismatch",
                        "query_id": query_id,
                        "page_id": page_id,
                        "qa_doc_id": doc_id,
                        "manifest_doc_id": page_docs[page_id],
                    }
                )
                continue
            records.append(
                RetrievalRecord(
                    query_id=query_id,
                    query_text=str(entry["question"]),
                    positive_page_id=page_id,
                    doc_id=doc_id,
                    image_path=page_paths[page_id],
                    split=doc_split[doc_id],
                    source_split=source_split,
                    answers=tuple(str(item) for item in entry.get("answers", ())),
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_path = output_dir / "anomalies.json"
    anomaly_path.write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if anomalies:
        raise ValueError(
            f"Retrieval manifest build found {len(anomalies)} anomalies; "
            f"see {anomaly_path}"
        )

    records.sort(key=lambda item: item.query_id)
    write_retrieval_manifest(output_dir / "all.jsonl", records)
    for split in ("train", "val", "test"):
        write_retrieval_manifest(
            output_dir / f"{split}.jsonl",
            (record for record in records if record.split == split),
        )

    split_counts = Counter(record.split for record in records)
    source_counts = Counter(record.source_split for record in records)
    pages_by_split = {
        split: len(
            {
                record.positive_page_id
                for record in records
                if record.split == split
            }
        )
        for split in ("train", "val", "test")
    }
    docs_by_split = {
        split: len(
            {record.doc_id for record in records if record.split == split}
        )
        for split in ("train", "val", "test")
    }
    summary = {
        "schema_version": 1,
        "task": "retrieval_rag",
        "split_source": "canonical sensitivity doc_id split",
        "data_root_at_build_time": str(data_root),
        "paths_are_relative_to": "data_root",
        "total": {
            "queries": len(records),
            "pages": len({record.positive_page_id for record in records}),
            "documents": len({record.doc_id for record in records}),
            "queries_with_answers": sum(bool(record.answers) for record in records),
        },
        "splits": {
            split: {
                "queries": split_counts[split],
                "pages": pages_by_split[split],
                "documents": docs_by_split[split],
            }
            for split in ("train", "val", "test")
        },
        "source_splits": dict(sorted(source_counts.items())),
        "document_overlap": _document_overlap(records),
        "anomaly_count": 0,
    }
    if any(summary["document_overlap"].values()):
        raise RuntimeError(
            f"Document leakage in retrieval manifest: "
            f"{summary['document_overlap']}"
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return RetrievalManifestBuildResult(
        output_dir=output_dir,
        records=len(records),
        pages=summary["total"]["pages"],
        documents=summary["total"]["documents"],
        split_counts=dict(split_counts),
    )


def _document_overlap(records: list[RetrievalRecord]) -> dict[str, int]:
    docs = {
        split: {record.doc_id for record in records if record.split == split}
        for split in ("train", "val", "test")
    }
    return {
        "train_val": len(docs["train"] & docs["val"]),
        "train_test": len(docs["train"] & docs["test"]),
        "val_test": len(docs["val"] & docs["test"]),
    }
