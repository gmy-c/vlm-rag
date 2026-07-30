from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import SensitivityRecord, write_manifest
from .split import SPLIT_NAMES, assign_documents_to_splits, split_membership


QA_FILENAMES = (
    "train_v1.0_withQT.json",
    "val_v1.0_withQT.json",
    "test_v1.0.json",
)


class ManifestBuildError(RuntimeError):
    """Raised when the source dataset violates a required integrity invariant."""


@dataclass(frozen=True, slots=True)
class PageMetadata:
    page_id: str
    doc_id: str
    page_no: int
    source_split: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    output_dir: Path
    records: tuple[SensitivityRecord, ...]
    summary: dict[str, Any]


def build_sensitivity_manifests(
    data_root: Path,
    output_dir: Path | None = None,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verify_images: bool = True,
    validate_ocr_json: bool = True,
) -> BuildResult:
    """Build portable sensitivity manifests without moving source data."""

    data_root = data_root.expanduser().resolve()
    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else data_root / "manifests" / "sensitivity"
    )
    paths = _required_paths(data_root)
    _ensure_directories(paths)

    anomalies: dict[str, list[Any]] = defaultdict(list)
    full_images = _scan_unique_files(paths["images"], ".png", anomalies, "full_images")
    full_ocr = _scan_unique_files(paths["ocr"], ".json", anomalies, "full_ocr")
    positive_images = _scan_unique_files(
        paths["positive_images"], ".png", anomalies, "positive_images"
    )
    positive_ocr = _scan_unique_files(
        paths["positive_ocr"], ".json", anomalies, "positive_ocr"
    )
    page_metadata, qa_summary = _load_qa_metadata(paths["qa"], anomalies)
    positive_page_metadata, positive_qa_summary = _load_qa_metadata(
        paths["positive_qa"], anomalies
    )

    full_ids = set(full_images)
    ocr_ids = set(full_ocr)
    positive_ids = set(positive_images)
    positive_ocr_ids = set(positive_ocr)
    qa_ids = set(page_metadata)
    positive_qa_ids = set(positive_page_metadata)

    _record_set_difference(anomalies, "positive_images_not_in_full", positive_ids, full_ids)
    _record_set_difference(anomalies, "full_images_missing_ocr", full_ids, ocr_ids)
    _record_set_difference(anomalies, "ocr_without_full_image", ocr_ids, full_ids)
    _record_set_difference(anomalies, "positive_images_missing_positive_ocr", positive_ids, positive_ocr_ids)
    _record_set_difference(anomalies, "positive_ocr_without_positive_image", positive_ocr_ids, positive_ids)
    _record_set_difference(
        anomalies,
        "positive_images_missing_positive_qa_mapping",
        positive_ids,
        positive_qa_ids,
    )
    _record_set_difference(
        anomalies,
        "positive_qa_pages_without_positive_image",
        positive_qa_ids,
        positive_ids,
    )
    _record_set_difference(anomalies, "full_images_missing_qa_mapping", full_ids, qa_ids)
    _record_set_difference(anomalies, "qa_pages_missing_full_image", qa_ids, full_ids)
    for page_id in sorted(positive_qa_ids & qa_ids):
        if positive_page_metadata[page_id] != page_metadata[page_id]:
            anomalies["positive_qa_metadata_mismatch"].append(
                {
                    "page_id": page_id,
                    "full": _page_metadata_dict(page_metadata[page_id]),
                    "positive_subset": _page_metadata_dict(positive_page_metadata[page_id]),
                }
            )

    provenance_summary = _load_and_validate_provenance(
        paths["provenance"],
        full_page_count=len(full_ids),
        positive_page_count=len(positive_ids),
        full_question_count=sum(qa_summary["question_counts"].values()),
        positive_question_count=sum(positive_qa_summary["question_counts"].values()),
        anomalies=anomalies,
    )

    if verify_images:
        anomalies["invalid_full_images"].extend(_verify_images(full_images.values()))
        anomalies["invalid_positive_images"].extend(_verify_images(positive_images.values()))
    if validate_ocr_json:
        anomalies["invalid_full_ocr"].extend(_verify_json_files(full_ocr.values()))
        anomalies["invalid_positive_ocr"].extend(_verify_json_files(positive_ocr.values()))

    fatal_anomalies = {key: value for key, value in anomalies.items() if value}
    if fatal_anomalies:
        _write_anomalies(output_dir, fatal_anomalies)
        details = ", ".join(f"{key}={len(value)}" for key, value in fatal_anomalies.items())
        raise ManifestBuildError(
            f"Dataset integrity validation failed ({details}). "
            f"See {output_dir / 'anomalies.json'}"
        )

    labels = [(page_id, int(page_id in positive_ids)) for page_id in sorted(full_ids)]
    doc_assignments = assign_documents_to_splits(
        ((page_metadata[page_id].doc_id, label) for page_id, label in labels),
        ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )

    records = tuple(
        SensitivityRecord(
            page_id=page_id,
            doc_id=page_metadata[page_id].doc_id,
            page_no=page_metadata[page_id].page_no,
            image_path=full_images[page_id].relative_to(data_root).as_posix(),
            ocr_path=full_ocr[page_id].relative_to(data_root).as_posix(),
            is_sensitive=label,
            split=doc_assignments[page_metadata[page_id].doc_id],
            source_split=page_metadata[page_id].source_split,
        )
        for page_id, label in labels
    )
    _validate_records(records)

    summary = _build_summary(
        records=records,
        data_root=data_root,
        output_dir=output_dir,
        seed=seed,
        ratios=(train_ratio, val_ratio, test_ratio),
        verify_images=verify_images,
        validate_ocr_json=validate_ocr_json,
        qa_summary=qa_summary,
        positive_qa_summary=positive_qa_summary,
        provenance_summary=provenance_summary,
    )
    _write_outputs(output_dir, records, summary)
    return BuildResult(output_dir=output_dir, records=records, summary=summary)


def _required_paths(data_root: Path) -> dict[str, Path]:
    return {
        "images": data_root / "docvqa_images",
        "ocr": data_root / "ocr",
        "qa": data_root / "docvqa_extracted",
        "positive_images": data_root / "desensitized" / "docvqa_images",
        "positive_ocr": data_root / "desensitized" / "ocr",
        "positive_qa": data_root / "desensitized" / "docvqa_extracted",
        "provenance": data_root / "desensitized" / "dataset_summary.json",
    }


def _ensure_directories(paths: dict[str, Path]) -> None:
    directory_paths = {
        name: path for name, path in paths.items() if name != "provenance"
    }
    missing = [
        f"{name}: {path}" for name, path in directory_paths.items() if not path.is_dir()
    ]
    if missing:
        raise ManifestBuildError("Required dataset directories are missing:\n" + "\n".join(missing))


def _scan_unique_files(
    directory: Path,
    suffix: str,
    anomalies: dict[str, list[Any]],
    source_name: str,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() != suffix:
            continue
        page_id = path.stem
        if page_id in files:
            anomalies[f"duplicate_{source_name}_page_ids"].append(
                {"page_id": page_id, "paths": [str(files[page_id]), str(path)]}
            )
        else:
            files[page_id] = path
    return files


def _load_qa_metadata(
    qa_dir: Path,
    anomalies: dict[str, list[Any]],
) -> tuple[dict[str, PageMetadata], dict[str, Any]]:
    metadata: dict[str, PageMetadata] = {}
    question_counts: dict[str, int] = {}
    page_counts: dict[str, int] = {}
    official_doc_sets: dict[str, set[str]] = {}

    for filename in QA_FILENAMES:
        path = qa_dir / filename
        if not path.is_file():
            anomalies["missing_qa_files"].append(str(path))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            anomalies["invalid_qa_json"].append({"path": str(path), "error": str(exc)})
            continue
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            anomalies["invalid_qa_schema"].append(
                {"path": str(path), "error": "top-level 'data' must be a list"}
            )
            continue

        source_split = _source_split_from_filename(filename)
        question_counts[source_split] = len(rows)
        split_pages: set[str] = set()
        split_docs: set[str] = set()
        for row_index, row in enumerate(rows):
            try:
                image_name = Path(str(row["image"]).replace("\\", "/")).name
                page_id = Path(image_name).stem
                doc_id = str(row["ucsf_document_id"])
                page_no = int(row["ucsf_document_page_no"])
            except (KeyError, TypeError, ValueError) as exc:
                anomalies["invalid_qa_rows"].append(
                    {"path": str(path), "row": row_index, "error": str(exc)}
                )
                continue
            if not page_id or not doc_id:
                anomalies["invalid_qa_rows"].append(
                    {"path": str(path), "row": row_index, "error": "empty page_id/doc_id"}
                )
                continue
            expected_page_id = f"{doc_id}_{page_no}"
            if page_id != expected_page_id:
                anomalies["qa_filename_metadata_mismatch"].append(
                    {
                        "path": str(path),
                        "row": row_index,
                        "page_id": page_id,
                        "expected_page_id": expected_page_id,
                    }
                )
            candidate = PageMetadata(page_id, doc_id, page_no, source_split)
            previous = metadata.get(page_id)
            if previous is not None and previous != candidate:
                anomalies["conflicting_qa_page_metadata"].append(
                    {
                        "page_id": page_id,
                        "first": _page_metadata_dict(previous),
                        "second": _page_metadata_dict(candidate),
                    }
                )
            else:
                metadata[page_id] = candidate
            split_pages.add(page_id)
            split_docs.add(doc_id)
        page_counts[source_split] = len(split_pages)
        official_doc_sets[source_split] = split_docs

    official_doc_overlap = {
        f"{left}_{right}": len(official_doc_sets.get(left, set()) & official_doc_sets.get(right, set()))
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    return metadata, {
        "question_counts": question_counts,
        "unique_page_counts": page_counts,
        "official_document_overlap": official_doc_overlap,
    }


def _source_split_from_filename(filename: str) -> str:
    if filename.startswith("train"):
        return "train"
    if filename.startswith("val"):
        return "val"
    if filename.startswith("test"):
        return "test"
    raise ValueError(f"Cannot infer source split from {filename!r}")


def _load_and_validate_provenance(
    path: Path,
    *,
    full_page_count: int,
    positive_page_count: int,
    full_question_count: int,
    positive_question_count: int,
    anomalies: dict[str, list[Any]],
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_provided"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        anomalies["invalid_dataset_summary"].append(
            {"path": str(path), "error": str(exc)}
        )
        return {"status": "invalid"}
    if not isinstance(payload, dict):
        anomalies["invalid_dataset_summary"].append(
            {"path": str(path), "error": "top-level JSON value must be an object"}
        )
        return {"status": "invalid"}

    expected_counts = {
        "total_pages_original": full_page_count,
        "sensitive_pages": positive_page_count,
        "total_qa_original": full_question_count,
        "total_qa_kept": positive_question_count,
    }
    for field, expected in expected_counts.items():
        actual = payload.get(field)
        if actual != expected:
            anomalies["dataset_summary_count_mismatch"].append(
                {"field": field, "expected": expected, "actual": actual}
            )
    return {"status": "validated", **payload}


def _page_metadata_dict(metadata: PageMetadata) -> dict[str, object]:
    return {
        "page_id": metadata.page_id,
        "doc_id": metadata.doc_id,
        "page_no": metadata.page_no,
        "source_split": metadata.source_split,
    }


def _record_set_difference(
    anomalies: dict[str, list[Any]],
    key: str,
    left: set[str],
    right: set[str],
) -> None:
    anomalies[key].extend(sorted(left - right))


def _verify_images(paths: Iterable[Path]) -> list[dict[str, str]]:
    from PIL import Image

    invalid: list[dict[str, str]] = []
    for path in paths:
        try:
            if path.stat().st_size <= 0:
                raise ValueError("empty file")
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # Pillow exposes multiple decoder-specific errors.
            invalid.append({"path": str(path), "error": str(exc)})
    return invalid


def _verify_json_files(paths: Iterable[Path]) -> list[dict[str, str]]:
    invalid: list[dict[str, str]] = []
    for path in paths:
        try:
            if path.stat().st_size <= 0:
                raise ValueError("empty file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level OCR JSON value must be an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    return invalid


def _validate_records(records: tuple[SensitivityRecord, ...]) -> None:
    page_ids = [record.page_id for record in records]
    if len(page_ids) != len(set(page_ids)):
        raise ManifestBuildError("Generated records contain duplicate page_id values")

    membership = split_membership({record.doc_id: record.split for record in records})
    overlaps = {
        f"{left}_{right}": membership[left] & membership[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    non_empty = {key: value for key, value in overlaps.items() if value}
    if non_empty:
        raise ManifestBuildError(f"Document leakage detected: {non_empty}")
    if any(not membership[split] for split in SPLIT_NAMES):
        raise ManifestBuildError("At least one generated split contains no documents")
    for record in records:
        record.validate()


def _build_summary(
    *,
    records: tuple[SensitivityRecord, ...],
    data_root: Path,
    output_dir: Path,
    seed: int,
    ratios: tuple[float, float, float],
    verify_images: bool,
    validate_ocr_json: bool,
    qa_summary: dict[str, Any],
    positive_qa_summary: dict[str, Any],
    provenance_summary: dict[str, Any],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        rows = [record for record in records if record.split == split]
        positives = sum(record.is_sensitive for record in rows)
        by_split[split] = {
            "pages": len(rows),
            "positive": positives,
            "negative": len(rows) - positives,
            "positive_ratio": positives / len(rows) if rows else 0.0,
            "documents": len({record.doc_id for record in rows}),
        }

    doc_sets = {
        split: {record.doc_id for record in records if record.split == split}
        for split in SPLIT_NAMES
    }
    overlap = {
        f"{left}_{right}": len(doc_sets[left] & doc_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    positives = sum(record.is_sensitive for record in records)
    return {
        "schema_version": 1,
        "task": "page_sensitivity_classification",
        "label_definition": (
            "is_sensitive=1 iff page_id is present in "
            "desensitized/docvqa_images; otherwise 0"
        ),
        "label_source": "desensitized_membership",
        "paths_are_relative_to": "data_root",
        "data_root_at_build_time": str(data_root),
        "output_dir_at_build_time": str(output_dir),
        "seed": seed,
        "requested_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "validation": {
            "verify_images": verify_images,
            "validate_ocr_json": validate_ocr_json,
            "status": "passed",
            "anomaly_count": 0,
            "generated_document_overlap": overlap,
        },
        "total": {
            "pages": len(records),
            "positive": positives,
            "negative": len(records) - positives,
            "positive_ratio": positives / len(records) if records else 0.0,
            "documents": len({record.doc_id for record in records}),
        },
        "splits": by_split,
        "source_qa": qa_summary,
        "positive_subset_qa": positive_qa_summary,
        "positive_subset_provenance": provenance_summary,
    }


def _write_outputs(
    output_dir: Path,
    records: tuple[SensitivityRecord, ...],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "all.jsonl", records)
    for split in SPLIT_NAMES:
        write_manifest(
            output_dir / f"{split}.jsonl",
            (record for record in records if record.split == split),
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_anomalies(output_dir, {})


def _write_anomalies(output_dir: Path, anomalies: dict[str, list[Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed" if anomalies else "passed",
        "anomaly_count": sum(len(values) for values in anomalies.values()),
        "anomalies": anomalies,
    }
    (output_dir / "anomalies.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
