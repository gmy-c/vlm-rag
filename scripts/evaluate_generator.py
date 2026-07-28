"""
Multi-page visual generation evaluation script.

Evaluates per-page inference vs image-stitching baseline on the
DocVQA test set using the Doubao Vision API.

Usage:
    export DOUBAO_API_KEY="your-api-key"
    python scripts/evaluate_generator.py --config configs/config.yaml
    python scripts/evaluate_generator.py --sample 50 --top-k 3
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
        description="Evaluate multi-page visual generation methods"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to project config YAML",
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
    args = parser.parse_args()

    # ── 1. Config + API Key ──
    from vlm_rag.config import load_config, resolve_project_path

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    try:
        api_key = config.get_api_key()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # ── 2. Data root ──
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path(
            os.environ.get(
                "DOCVQA_DATA_ROOT",
                str(PROJECT_ROOT.parent / "__MACOSX" / "aiproject" / "data"),
            )
        )

    # ── 3. Load test data ──
    from vlm_rag.data import load_docvqa_dataset

    qa_path = data_root / "docvqa_extracted" / "val_v1.0_withQT.json"
    images_dir = data_root / "docvqa_images"

    if not qa_path.exists():
        print(f"ERROR: Q&A file not found: {qa_path}")
        sys.exit(1)

    print("Loading test data ...")
    pages, queries = load_docvqa_dataset(qa_path, images_dir, split_name="test")

    if args.sample > 0:
        import random
        rng = random.Random(42)
        queries = rng.sample(queries, min(args.sample, len(queries)))

    # ── 4. Init retriever (ColPali, no fine-tuning) ──
    from vlm_rag.encoders import ColPaliDualEncoder, ColPaliDualEncoderConfig
    from vlm_rag.retriever import DualTowerRetriever

    print(f"Initialising retriever on {args.device} ...")
    enc_cfg = ColPaliDualEncoderConfig(
        device=args.device,
        use_lora=False,
    )
    encoder = ColPaliDualEncoder(enc_cfg)
    retriever = DualTowerRetriever(encoder)
    retriever.index(pages)

    # ── 5. Config summary (key masked) ──
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

    # ── 6. Run evaluation ──
    from vlm_rag.baselines import evaluate_generation_methods

    total_start = time.perf_counter()

    results = evaluate_generation_methods(
        retriever=retriever,
        pages=pages,
        queries=queries,
        api_key=api_key,
        top_k=args.top_k,
    )

    total_elapsed = time.perf_counter() - total_start

    # ── 7. Print results ──
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for method, metrics in results.items():
        print(f"  {method:12s}  Accuracy={metrics['accuracy']:.4f}  "
              f"EM={metrics['em']:.4f}")
    print(f"\n  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Avg per query:  {total_elapsed / max(1, len(queries)):.2f}s")
    print("=" * 60)

    # ── 8. Save to CSV ──
    output_dir = (
        Path(args.output)
        if args.output
        else resolve_project_path(PROJECT_ROOT, config.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generator_metrics.csv"

    from vlm_rag.metrics import accuracy as calc_accuracy

    with report_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["method", "accuracy", "em"])
        writer.writeheader()
        for method, metrics in results.items():
            writer.writerow({"method": method, **metrics})

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
