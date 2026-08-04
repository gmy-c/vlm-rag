from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval.index import MultiVectorIndex
from vlm_rag.retrieval.model import (
    LateInteractionModelConfig,
    LateInteractionRetriever,
)
from vlm_rag.retrieval.schema import load_retrieval_manifest
from vlm_rag.pipeline.provenance import (
    fingerprint_adapter,
    fingerprint_base_model_metadata,
)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate global-coarse plus exact-MaxSim retrieval."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-top-k", type=int, default=128)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--maxsim-backend",
        choices=("auto", "lik", "chunked"),
        default="auto",
    )
    parser.add_argument(
        "--maxsim-normalization",
        choices=("mean", "sum"),
        default="mean",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the evaluation progress bar.",
    )
    args = parser.parse_args()

    records = load_retrieval_manifest(args.manifest)
    if args.max_queries is not None:
        records = records[: args.max_queries]
    model, _ = LateInteractionRetriever.from_adapter(
        args.adapter,
        checkpoint_path_override=args.model,
        device=args.device,
    )
    model.eval()
    index = MultiVectorIndex(
        args.index_dir,
        expected_adapter_sha256=fingerprint_adapter(args.adapter),
        expected_base_model_metadata_sha256=(
            fingerprint_base_model_metadata(Path(args.model))
        ),
    )
    reciprocal_ranks = []
    recall_hits = {1: 0, 5: 0, 10: 0}
    misses = []
    progress = tqdm(
        records,
        total=len(records),
        desc="evaluating retrieval",
        unit="query",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    reciprocal_rank_sum = 0.0
    for query_index, record in enumerate(progress, start=1):
        query_tokens, query_global = model.encode_queries(
            [record.query_text]
        )
        candidates = index.coarse_candidates(
            query_global,
            top_k=args.coarse_top_k,
        )
        ranking = index.rerank(
            query_tokens,
            candidates,
            backend=args.maxsim_backend,
            normalization=args.maxsim_normalization,
        )
        ranked_ids = [item[0] for item in ranking]
        try:
            rank = ranked_ids.index(record.positive_page_id) + 1
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)
            reciprocal_rank_sum += reciprocal_rank
        except ValueError:
            rank = None
            reciprocal_ranks.append(0.0)
            misses.append(record.query_id)
        for cutoff in recall_hits:
            recall_hits[cutoff] += int(
                record.positive_page_id in ranked_ids[:cutoff]
            )
        if query_index == 1 or query_index % 25 == 0 or query_index == len(records):
            progress.set_postfix(
                mrr=f"{reciprocal_rank_sum / query_index:.4f}",
                r5=f"{recall_hits[5] / query_index:.4f}",
                miss=len(misses),
                refresh=False,
            )
    count = len(records)
    if count == 0:
        raise ValueError("Evaluation manifest is empty")
    report = {
        "queries": count,
        "coarse_top_k": args.coarse_top_k,
        "mrr": sum(reciprocal_ranks) / count,
        **{
            f"recall_at_{cutoff}": hits / count
            for cutoff, hits in recall_hits.items()
        },
        "coarse_miss_count": len(misses),
        "coarse_miss_query_ids": misses[:100],
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved() / 2**30
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
