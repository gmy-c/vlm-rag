from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any


class JsonResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock:
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(path)
