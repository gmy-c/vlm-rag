from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .dataset import resolve_data_path
from .schema import SensitivityRecord, load_manifest
from ..pipeline.provenance import sha256_file


@dataclass(frozen=True, slots=True)
class SensitivityCatalogEntry:
    page_id: str
    doc_id: str
    image_path: str
    probability: float | None
    threshold: float | None
    is_sensitive: bool | None
    status: str
    error_type: str | None = None
    error: str | None = None
    image_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SensitivityCatalogEntry":
        probability = value.get("probability")
        threshold = value.get("threshold")
        sensitive = value.get("is_sensitive")
        return cls(
            page_id=str(value["page_id"]),
            doc_id=str(value["doc_id"]),
            image_path=str(value["image_path"]),
            probability=float(probability) if probability is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            is_sensitive=bool(sensitive) if sensitive is not None else None,
            status=str(value["status"]),
            error_type=(
                str(value["error_type"])
                if value.get("error_type") is not None
                else None
            ),
            error=str(value["error"]) if value.get("error") is not None else None,
            image_sha256=(
                str(value["image_sha256"])
                if value.get("image_sha256") is not None
                else None
            ),
        )


class SensitivityCatalog:
    def __init__(
        self,
        entries: Iterable[SensitivityCatalogEntry],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.entries = {entry.page_id: entry for entry in entries}
        if not self.entries:
            raise ValueError("Sensitivity catalog is empty")
        self.metadata = metadata or {}

    def get(self, page_id: str) -> SensitivityCatalogEntry | None:
        return self.entries.get(page_id)

    @classmethod
    def load(cls, path: Path) -> "SensitivityCatalog":
        entries: list[SensitivityCatalogEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: catalog row is not an object"
                    )
                entries.append(SensitivityCatalogEntry.from_dict(value))
        metadata_path = path.with_name("catalog_metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        return cls(entries, metadata)


def build_catalog(
    *,
    manifest_path: Path,
    predictions_path: Path,
    errors_path: Path | None,
    data_root: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    hash_images: bool = True,
) -> tuple[list[SensitivityCatalogEntry], dict[str, Any]]:
    records = load_manifest(manifest_path)
    predictions = _load_unique_jsonl(predictions_path, "page_id")
    errors = (
        _load_unique_jsonl(errors_path, "page_id")
        if errors_path is not None and errors_path.is_file()
        else {}
    )
    overlap = predictions.keys() & errors.keys()
    if overlap:
        raise ValueError(
            f"Pages appear in both predictions and errors: {sorted(overlap)[:10]}"
        )
    expected = {record.page_id for record in records}
    observed = predictions.keys() | errors.keys()
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            "Catalog input coverage mismatch; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"missing_examples={sorted(missing)[:10]}, "
            f"unexpected_examples={sorted(unexpected)[:10]}"
        )

    entries: list[SensitivityCatalogEntry] = []
    for record in records:
        image_hash = (
            sha256_file(resolve_data_path(data_root, record.image_path))
            if hash_images
            else None
        )
        if record.page_id in errors:
            error = errors[record.page_id]
            entries.append(
                SensitivityCatalogEntry(
                    page_id=record.page_id,
                    doc_id=record.doc_id,
                    image_path=record.image_path,
                    probability=None,
                    threshold=None,
                    is_sensitive=None,
                    status="error",
                    error_type=str(error.get("error_type", "inference_error")),
                    error=str(error.get("error", "Sensitivity inference failed")),
                    image_sha256=image_hash,
                )
            )
            continue
        prediction = predictions[record.page_id]
        probability = float(prediction["probability"])
        threshold = float(prediction["threshold"])
        predicted_label = int(prediction["predicted_label"])
        if predicted_label not in {0, 1}:
            raise ValueError(
                f"{record.page_id}: predicted_label must be 0 or 1"
            )
        entries.append(
            SensitivityCatalogEntry(
                page_id=record.page_id,
                doc_id=record.doc_id,
                image_path=record.image_path,
                probability=probability,
                threshold=threshold,
                is_sensitive=bool(predicted_label),
                status="ok",
                image_sha256=image_hash,
            )
        )

    metadata = {
        "schema_version": 1,
        "task": "sensitivity_catalog",
        "pages": len(entries),
        "sensitive": sum(entry.is_sensitive is True for entry in entries),
        "non_sensitive": sum(entry.is_sensitive is False for entry in entries),
        "errors": sum(entry.status != "ok" for entry in entries),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "image_hashes": hash_images,
    }
    return entries, metadata


def write_catalog(
    path: Path,
    entries: Iterable[SensitivityCatalogEntry],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(
                json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    temporary.replace(path)
    metadata_path = path.with_name("catalog_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_unique_jsonl(
    path: Path,
    key_name: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            key = str(value[key_name])
            if key in result:
                raise ValueError(f"{path}:{line_number}: duplicate {key_name}={key}")
            result[key] = value
    return result
