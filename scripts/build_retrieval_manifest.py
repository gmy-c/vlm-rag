from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval import build_retrieval_manifests


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build document-disjoint retrieval manifests from all DocVQA QA files."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sensitivity-manifest-dir", type=Path, default=None)
    args = parser.parse_args()
    result = build_retrieval_manifests(
        args.data_root,
        args.output_dir,
        sensitivity_manifest_dir=args.sensitivity_manifest_dir,
    )
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "records": result.records,
        "pages": result.pages,
        "documents": result.documents,
        "split_counts": result.split_counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
