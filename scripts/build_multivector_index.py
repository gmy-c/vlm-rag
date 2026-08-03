from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.index import build_multivector_index
from vlm_rag.retrieval.model import (
    LateInteractionModelConfig,
    LateInteractionRetriever,
)
from vlm_rag.retrieval.schema import load_retrieval_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a sharded native ColPali multi-vector page index."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pages-per-shard", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.adapter is not None:
        model, _ = LateInteractionRetriever.from_adapter(
            args.adapter,
            checkpoint_path_override=args.model,
            device=args.device,
        )
    else:
        model = LateInteractionRetriever(
            LateInteractionModelConfig(checkpoint_path=args.model),
            device=args.device,
        )
    metadata = build_multivector_index(
        model,
        load_retrieval_manifest(args.manifest),
        args.data_root,
        args.output_dir,
        batch_size=args.batch_size,
        pages_per_shard=args.pages_per_shard,
        manifest_path=args.manifest,
        adapter_dir=args.adapter,
        base_model_path=Path(args.model),
        progress=not args.no_progress,
    )
    print(
        f"Indexed {metadata['page_count']} pages in {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
