from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.generation import DoubaoClientConfig
from vlm_rag.pipeline.factory import build_secure_rag_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run secure query -> multi-vector retrieval -> sensitivity gate -> "
            "Doubao visual answer."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query")
    source.add_argument("--query-file", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-catalog", type=Path, required=True)
    parser.add_argument("--redaction-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="Required acknowledgement before any document bytes are sent to Ark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_real_api:
        raise ValueError(
            "Refusing external transmission without --allow-real-api. "
            "Run tests without this flag; add it only after reviewing the catalog."
        )
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise ValueError(
            "Set ARK_API_KEY (preferred) or DOUBAO_API_KEY in the shell environment"
        )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    doubao = config.get("doubao", {})
    retrieval = config.get("retrieval", {})
    policy = config.get("policy", {})
    client_config = DoubaoClientConfig(
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
    )
    engine = build_secure_rag_engine(
        data_root=args.data_root,
        base_model=args.model,
        retrieval_adapter=args.adapter,
        index_dir=args.index_dir,
        sensitivity_catalog=args.sensitivity_catalog,
        redaction_manifest=args.redaction_manifest,
        api_key=api_key,
        doubao_config=client_config,
        coarse_top_k=int(retrieval.get("coarse_top_k", 128)),
        rerank_top_k=int(retrieval.get("rerank_top_k", 20)),
        answer_page_limit=int(policy.get("answer_page_limit", 3)),
        maxsim_backend=str(retrieval.get("maxsim_backend", "auto")),
        audit_path=args.audit,
        device=args.device,
    )
    queries = (
        [args.query]
        if args.query is not None
        else [
            line.strip()
            for line in args.query_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            result = engine.answer(query)
            row = {"query": query, **result.to_dict()}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "answer": result.answer,
                        "evidence_page_ids": result.evidence_page_ids,
                    },
                    ensure_ascii=False,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
