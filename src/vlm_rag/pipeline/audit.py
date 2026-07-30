from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

from .contracts import PipelineAnswer
from .provenance import sha256_text


class JsonlAuditWriter:
    """Append a privacy-minimised audit record for each pipeline request."""

    def __init__(self, path: Path, *, include_query: bool = False) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_query = include_query
        self._lock = threading.Lock()

    def write(self, query: str, result: PipelineAnswer) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_sha256": sha256_text(query),
            "result": result.to_dict(),
        }
        if self.include_query:
            record["query"] = query
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
