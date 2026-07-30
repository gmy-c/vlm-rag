from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.index import MultiVectorIndex
from vlm_rag.retrieval.mining import mine_hard_negatives
from vlm_rag.retrieval.model import (
    LateInteractionModelConfig,
    LateInteractionRetriever,
)
from vlm_rag.retrieval.schema import load_retrieval_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine document-disjoint hard negatives using a page index."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-top-k", type=int, default=128)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--maxsim-backend",
        choices=("auto", "lik", "chunked"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = LateInteractionRetriever(
        LateInteractionModelConfig(checkpoint_path=args.model),
        device=args.device,
    )
    if args.adapter is not None:
        model.load_adapter(args.adapter)
    summary = mine_hard_negatives(
        model,
        MultiVectorIndex(args.index_dir),
        load_retrieval_manifest(args.manifest),
        args.output,
        coarse_top_k=args.coarse_top_k,
        negatives_per_query=args.negatives_per_query,
        backend=args.maxsim_backend,
        maximum_queries=args.max_queries,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
