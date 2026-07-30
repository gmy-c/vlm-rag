from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.pipeline.provenance import sha256_file
from vlm_rag.sensitivity.catalog import SensitivityCatalog


def resolve_under(root: Path, value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    path = root.joinpath(*pure.parts).resolve()
    path.relative_to(root)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate page_id/redacted_path mappings and create a hash-pinned "
            "redaction manifest. Approval must be explicit."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Mark validated derivatives as approved for external API use.",
    )
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    catalog = SensitivityCatalog.load(args.catalog.expanduser().resolve())
    rows = []
    seen: set[str] = set()
    with args.mappings.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            page_id = str(value["page_id"])
            if page_id in seen:
                raise ValueError(
                    f"{args.mappings}:{line_number}: duplicate page_id={page_id}"
                )
            seen.add(page_id)
            entry = catalog.get(page_id)
            if entry is None:
                raise ValueError(f"Unknown catalog page_id: {page_id}")
            if entry.is_sensitive is not True:
                raise ValueError(
                    f"{page_id}: redaction is only registered for sensitive pages"
                )
            redacted_relative = str(value["redacted_path"]).replace("\\", "/")
            source = resolve_under(root, entry.image_path)
            redacted = resolve_under(root, redacted_relative)
            if not source.is_file() or not redacted.is_file():
                raise FileNotFoundError(
                    f"{page_id}: source or redacted image is missing"
                )
            rows.append(
                {
                    "page_id": page_id,
                    "redacted_path": redacted_relative,
                    "approved": bool(args.approve),
                    "source_sha256": sha256_file(source),
                    "redacted_sha256": sha256_file(redacted),
                }
            )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"records": len(rows), "approved": bool(args.approve), "output": str(output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
