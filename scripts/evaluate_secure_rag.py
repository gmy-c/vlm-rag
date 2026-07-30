from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.generation import DoubaoClientConfig
from vlm_rag.pipeline.factory import build_secure_rag_engine
from vlm_rag.pipeline.metrics import secure_pipeline_metrics
from vlm_rag.retrieval.schema import load_retrieval_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the canonical secure multi-vector RAG pipeline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-catalog", type=Path, required=True)
    parser.add_argument("--redaction-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-real-api", action="store_true")
    args = parser.parse_args()

    if not args.allow_real_api:
        raise ValueError(
            "Secure RAG evaluation calls an external API; explicitly pass "
            "--allow-real-api after reviewing the sensitivity catalog."
        )
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise ValueError("ARK_API_KEY is not set")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    doubao = config.get("doubao", {})
    retrieval = config.get("retrieval", {})
    policy = config.get("policy", {})
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = build_secure_rag_engine(
        data_root=args.data_root,
        base_model=args.model,
        retrieval_adapter=args.adapter,
        index_dir=args.index_dir,
        sensitivity_catalog=args.sensitivity_catalog,
        redaction_manifest=args.redaction_manifest,
        api_key=api_key,
        doubao_config=DoubaoClientConfig(
            model=os.environ.get(
                "DOUBAO_MODEL",
                str(doubao.get("model", "doubao-seed-2-1-pro")),
            ),
            base_url=str(
                doubao.get(
                    "base_url",
                    "https://ark.cn-beijing.volces.com/api/v3",
                )
            ),
            max_tokens=int(doubao.get("max_tokens", 512)),
            temperature=float(doubao.get("temperature", 0.1)),
            timeout_seconds=float(doubao.get("timeout_seconds", 60)),
            max_attempts=int(doubao.get("max_attempts", 3)),
            minimum_interval_seconds=float(
                doubao.get("minimum_interval_seconds", 0)
            ),
            cache_dir=str(doubao.get("cache_dir", "outputs/doubao_cache")),
        ),
        coarse_top_k=int(retrieval.get("coarse_top_k", 128)),
        rerank_top_k=int(retrieval.get("rerank_top_k", 20)),
        answer_page_limit=int(policy.get("answer_page_limit", 3)),
        maxsim_backend=str(retrieval.get("maxsim_backend", "auto")),
        audit_path=output_dir / "audit.jsonl",
        device=args.device,
    )
    records = load_retrieval_manifest(args.manifest)
    rng = random.Random(args.seed)
    if args.max_queries > 0 and len(records) > args.max_queries:
        records = rng.sample(records, args.max_queries)
    rows = []
    result_path = output_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            result = engine.answer(record.query_text)
            rows.append((result, record.answers, record.positive_page_id))
            handle.write(
                json.dumps(
                    {
                        "query_id": record.query_id,
                        "positive_page_id": record.positive_page_id,
                        "answers": list(record.answers),
                        **result.to_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = secure_pipeline_metrics(rows)
    metrics["warning"] = (
        "Small-sample/API evaluation metrics are not formal model results."
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if (
        metrics["sensitive_original_exposure"] != 0
        or metrics["missing_catalog_exposure"] != 0
    ):
        raise RuntimeError("Security invariant violated; see evaluation output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
