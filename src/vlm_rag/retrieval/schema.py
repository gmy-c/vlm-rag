from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    query_id: str
    query_text: str
    positive_page_id: str
    doc_id: str
    image_path: str
    split: str
    source_split: str
    answers: tuple[str, ...] = ()
    hard_negative_page_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["answers"] = list(self.answers)
        value["hard_negative_page_ids"] = list(self.hard_negative_page_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalRecord":
        return cls(
            query_id=str(value["query_id"]),
            query_text=str(value["query_text"]),
            positive_page_id=str(value["positive_page_id"]),
            doc_id=str(value["doc_id"]),
            image_path=str(value["image_path"]),
            split=str(value["split"]),
            source_split=str(value["source_split"]),
            answers=tuple(str(item) for item in value.get("answers", ())),
            hard_negative_page_ids=tuple(
                str(item) for item in value.get("hard_negative_page_ids", ())
            ),
        )


def load_retrieval_manifest(path: Path) -> list[RetrievalRecord]:
    records: list[RetrievalRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            records.append(RetrievalRecord.from_dict(value))
    return records


def write_retrieval_manifest(
    path: Path,
    records: Iterable[RetrievalRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
