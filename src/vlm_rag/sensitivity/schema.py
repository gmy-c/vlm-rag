from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


VALID_SPLITS = frozenset({"train", "val", "test"})
VALID_LABELS = frozenset({0, 1})


@dataclass(frozen=True, slots=True)
class SensitivityRecord:
    """One page and its binary desensitization label.

    Paths are stored relative to the data root with POSIX separators so that a
    manifest generated on Windows can be consumed on Linux.
    """

    page_id: str
    doc_id: str
    page_no: int
    image_path: str
    ocr_path: str
    is_sensitive: int
    split: str
    source_split: str
    label_source: str = "desensitized_membership"

    def validate(self) -> None:
        if not self.page_id:
            raise ValueError("page_id must not be empty")
        if not self.doc_id:
            raise ValueError(f"{self.page_id}: doc_id must not be empty")
        if self.page_no < 0:
            raise ValueError(f"{self.page_id}: page_no must be non-negative")
        if self.is_sensitive not in VALID_LABELS:
            raise ValueError(
                f"{self.page_id}: is_sensitive must be 0 or 1, got {self.is_sensitive!r}"
            )
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"{self.page_id}: split must be one of {sorted(VALID_SPLITS)}, "
                f"got {self.split!r}"
            )
        if not self.source_split:
            raise ValueError(f"{self.page_id}: source_split must not be empty")
        if not self.label_source:
            raise ValueError(f"{self.page_id}: label_source must not be empty")
        _validate_relative_posix_path(self.image_path, self.page_id, "image_path")
        _validate_relative_posix_path(self.ocr_path, self.page_id, "ocr_path")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SensitivityRecord":
        record = cls(
            page_id=str(value["page_id"]),
            doc_id=str(value["doc_id"]),
            page_no=int(value["page_no"]),
            image_path=str(value["image_path"]),
            ocr_path=str(value["ocr_path"]),
            is_sensitive=int(value["is_sensitive"]),
            split=str(value["split"]),
            source_split=str(value["source_split"]),
            label_source=str(value.get("label_source", "desensitized_membership")),
        )
        record.validate()
        return record


def _validate_relative_posix_path(value: str, page_id: str, field_name: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or ".." in path.parts:
        raise ValueError(
            f"{page_id}: {field_name} must be a safe relative POSIX path, got {value!r}"
        )


def iter_manifest(path: Path) -> Iterator[SensitivityRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("JSON value is not an object")
                yield SensitivityRecord.from_dict(raw)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid manifest row: {exc}") from exc


def load_manifest(path: Path) -> list[SensitivityRecord]:
    return list(iter_manifest(path))


def write_manifest(path: Path, records: Iterable[SensitivityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary_path.replace(path)
