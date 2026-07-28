from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .data import Query


def split_queries(
    queries: list[Query],
    output_dir: Path,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
) -> dict[str, list[Query]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")
    if train_ratio + dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio must be less than 1")

    grouped: dict[str, list[Query]] = {}
    for query in queries:
        grouped.setdefault(query.doc_type, []).append(query)

    splits = {"train": [], "dev": [], "test": []}
    for doc_type in sorted(grouped):
        doc_queries = sorted(grouped[doc_type], key=lambda item: item.query_id)
        total = len(doc_queries)
        train_end = max(1, int(total * train_ratio))
        dev_end = min(total, train_end + max(1, int(total * dev_ratio)))
        splits["train"].extend(doc_queries[:train_end])
        splits["dev"].extend(doc_queries[train_end:dev_end])
        splits["test"].extend(doc_queries[dev_end:])

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_queries_ in splits.items():
        (output_dir / f"{split_name}_queries.json").write_text(
            json.dumps([asdict(query) for query in split_queries_], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = {name: len(items) for name, items in splits.items()}
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return splits

