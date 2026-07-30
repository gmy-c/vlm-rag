"""
Multi-page visual generation evaluation script.

Evaluates the full pipeline: trained retriever + Doubao Vision API,
comparing per-page inference vs image-stitching baseline.

Usage:
    # 仅评估检索（不需要 API Key）
    python scripts/evaluate_generator.py --retrieval-only

    # 完整评估（检索 + 生成，需要 API Key）
    export DOUBAO_API_KEY="your-api-key"
    python scripts/evaluate_generator.py --sample 10
    python scripts/evaluate_generator.py --sample 50 --top-k 5

    # 加载训练好的 LoRA 权重
    python scripts/evaluate_generator.py \\
        --retriever-checkpoint models/colpali_retriever
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _mask_key(key: str) -> str:
    """Return a masked version of *key*: first 4 chars + ``****``."""
    if len(key) <= 4:
        return "****"
    return key[:4] + "****"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[retrieval/rag] Evaluate ColPali retrieval and visual generation."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to project config YAML",
    )
    parser.add_argument(
        "--retriever-checkpoint", default=None,
        help="Path to trained retriever checkpoint (e.g. models/colpali_retriever/). "
             "If not provided, uses base ColPali without fine-tuning.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override local base checkpoint path or Hugging Face model ID",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of pages to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Only evaluate N queries (0 = all)",
    )
    parser.add_argument(
        "--page-sample",
        type=int,
        default=0,
        help="Limit candidate pages for a smoke evaluation while always "
             "keeping selected queries' positive pages (0 = all pages)",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=None,
        help="Pages per GPU batch while building the evaluation index",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="DocVQA data root directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device for the retriever (default: cuda)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only evaluate retrieval metrics (MRR, Recall), skip generation."
             " Does NOT require DOUBAO_API_KEY.",
    )
    args = parser.parse_args()

    # ── 1. Config ──
    from vlm_rag.config import load_config, resolve_project_path

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))

    # ── 2. API Key (not needed for retrieval-only mode) ──
    api_key = ""
    if not args.retrieval_only:
        try:
            api_key = config.get_api_key()
        except ValueError as exc:
            print(f"ERROR: {exc}")
            print("  For retrieval-only evaluation, use --retrieval-only")
            sys.exit(1)

    # ── 3. Data root ──
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(
            os.environ.get(
                "DOCVQA_DATA_ROOT",
                str(PROJECT_ROOT / "data"),
            )
        )

    # ── 4. Load test data ──
    from vlm_rag.data import load_docvqa_dataset

    qa_path = data_root / "docvqa_extracted" / "val_v1.0_withQT.json"
    images_dir = data_root / "docvqa_images"

    if not qa_path.exists():
        print(f"ERROR: Q&A file not found: {qa_path}")
        print("  Make sure DocVQA data is placed in data/")
        sys.exit(1)

    print("Loading test data ...")
    pages, queries = load_docvqa_dataset(qa_path, images_dir, split_name="test")

    if args.sample > 0:
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, min(args.sample, len(queries)))

    if args.page_sample > 0:
        import random

        positive_ids = {
            page_id
            for query in queries
            for page_id in query.positive_page_ids
        }
        if args.page_sample < len(positive_ids):
            parser.error(
                "--page-sample must be at least the number of unique "
                f"positive pages ({len(positive_ids)})"
            )
        page_lookup = {page.page_id: page for page in pages}
        missing_ids = positive_ids - page_lookup.keys()
        if missing_ids:
            raise ValueError(f"Positive pages missing from corpus: {missing_ids}")
        positive_pages = [page_lookup[page_id] for page_id in sorted(positive_ids)]
        distractors = [
            page for page in pages if page.page_id not in positive_ids
        ]
        rng = random.Random(43)
        distractor_count = min(
            args.page_sample - len(positive_pages),
            len(distractors),
        )
        pages = positive_pages + rng.sample(distractors, distractor_count)
        print(
            "Smoke page subset: "
            f"{len(positive_pages)} positives + {distractor_count} distractors"
        )

    # ── 5. Init retriever ──
    from vlm_rag.encoders import ColPaliDualEncoder, ColPaliDualEncoderConfig
    from vlm_rag.retriever import DualTowerRetriever

    # Resolve base model path
    model_name = (
        args.model
        or os.environ.get("COLPALI_MODEL_PATH")
        or config.colpali_model
    )
    model_path = Path(model_name)
    for local_path in (
        model_path,
        PROJECT_ROOT / model_path,
        PROJECT_ROOT.parent / model_path,
    ):
        if local_path.is_dir():
            model_name = str(local_path.resolve())
            break

    print(f"\nInitialising retriever on {args.device} ...")
    print(f"  Base model: {model_name}")

    # If a trained checkpoint is provided, load via ColPaliDualEncoder.load()
    if args.retriever_checkpoint:
        ckpt_dir = Path(args.retriever_checkpoint)
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir
        print(f"  Checkpoint:  {ckpt_dir}")

        if not (ckpt_dir / "head_weights.pt").exists():
            print(f"ERROR: head_weights.pt not found in {ckpt_dir}")
            sys.exit(1)

        encoder = ColPaliDualEncoder.load(
            ckpt_dir,
            base_model=model_name,
            device=args.device,
        )
        print(f"  OK Trained weights loaded from {ckpt_dir}")
    else:
        print("  INFO No checkpoint provided - using base ColPali (no fine-tuning)")
        enc_cfg = ColPaliDualEncoderConfig(
            model_name=model_name,
            device=args.device,
            use_lora=False,
        )
        encoder = ColPaliDualEncoder(enc_cfg)

    retriever = DualTowerRetriever(encoder)
    retriever.index(
        pages,
        batch_size=args.index_batch_size or config.index_batch_size,
    )

    # ═══════════════════════════════════════════════════════════════
    # 6a. Retrieval metrics (always computed, no API needed)
    # ═══════════════════════════════════════════════════════════════
    from vlm_rag.metrics import mrr_at_k, recall_at_k

    print("\n" + "=" * 60)
    print("Retrieval Evaluation (no API required)")
    print("=" * 60)

    ranked: dict[str, list] = {}
    for q in queries:
        ranked[q.query_id] = retriever.search(q.text, top_k=10)

    retrieval_metrics = {
        "mrr@10": round(mrr_at_k(queries, ranked, 10), 4),
        "recall@3": round(recall_at_k(queries, ranked, 3), 4),
        "recall@5": round(recall_at_k(queries, ranked, 5), 4),
        "recall@10": round(recall_at_k(queries, ranked, 10), 4),
    }
    for name, val in retrieval_metrics.items():
        print(f"  {name:14s} {val:.4f}")

    if args.retrieval_only:
        # Save retrieval results
        output_dir = (
            Path(args.output)
            if args.output
            else Path(
                os.environ.get(
                    "RAG_OUTPUT_DIR",
                    str(resolve_project_path(PROJECT_ROOT, config.output_dir)),
                )
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "retrieval_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fp:
            writer = csv.DictWriter(fp, fieldnames=list(retrieval_metrics.keys()))
            writer.writeheader()
            writer.writerow(retrieval_metrics)
        print(f"\nRetrieval metrics saved: {output_dir / 'retrieval_metrics.csv'}")
        print("\nTo run full evaluation with generation, remove --retrieval-only")
        print("and set: export DOUBAO_API_KEY='your-key'")
        return

    # ═══════════════════════════════════════════════════════════════
    # 6b. Generation evaluation (requires DOUBAO API)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"  API Key:       {_mask_key(api_key)}")
    print(f"  Model:         {config.generator_doubao_model}")
    print(f"  Base URL:      {config.generator_doubao_base_url}")
    print(f"  Test queries:  {len(queries)}")
    print(f"  Test pages:    {len(pages)}")
    print(f"  Top-K:         {args.top_k}")
    print(f"  Device:        {args.device}")
    print(f"  Start time:    {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60 + "\n")

    from vlm_rag.baselines import evaluate_generation_methods

    total_start = time.perf_counter()

    gen_results = evaluate_generation_methods(
        retriever=retriever,
        pages=pages,
        queries=queries,
        api_key=api_key,
        top_k=args.top_k,
        model=config.generator_doubao_model,
        base_url=config.generator_doubao_base_url,
        max_tokens=config.generator_max_tokens,
        temperature=config.generator_temperature,
        timeout=config.generator_timeout,
    )

    total_elapsed = time.perf_counter() - total_start

    # ═══════════════════════════════════════════════════════════════
    # 7. Combined results
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Combined Results")
    print("=" * 60)
    print(f"  {'Metric':20s} {'Value':>8s}")
    print(f"  {'-' * 20} {'-' * 8}")
    for name, val in retrieval_metrics.items():
        print(f"  {name:20s} {val:>8.4f}")
    print(f"  {'-' * 20} {'-' * 8}")
    for method, metrics in gen_results.items():
        print(f"  gen/{method:13s}  {metrics['accuracy']:>8.4f}")
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Avg per query:  {total_elapsed / max(1, len(queries)):.2f}s")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 8. Save reports
    # ═══════════════════════════════════════════════════════════════
    output_dir = (
        Path(args.output)
        if args.output
        else Path(
            os.environ.get(
                "RAG_OUTPUT_DIR",
                str(resolve_project_path(PROJECT_ROOT, config.output_dir)),
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval
    ret_path = output_dir / "retrieval_metrics.csv"
    with ret_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(retrieval_metrics.keys()))
        writer.writeheader()
        writer.writerow(retrieval_metrics)

    # Generation
    gen_path = output_dir / "generator_metrics.csv"
    with gen_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["method", "accuracy", "em"])
        writer.writeheader()
        for method, metrics in gen_results.items():
            writer.writerow({"method": method, **metrics})

    # Combined
    combined_path = output_dir / "combined_metrics.csv"
    combined = {**retrieval_metrics}
    for method, metrics in gen_results.items():
        combined[f"gen_{method}_accuracy"] = metrics["accuracy"]
    with combined_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(combined.keys()))
        writer.writeheader()
        writer.writerow(combined)

    print(f"\nReports saved:")
    print(f"  {ret_path}")
    print(f"  {gen_path}")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()
