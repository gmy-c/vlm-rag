from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, resolve_project_path
from .workflows import (
    build_dataset_workflow,
    build_index_workflow,
    demo_workflow,
    evaluate_workflow,
    train_workflow,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-RAG engineering command line.")
    parser.add_argument("--config", default="configs/config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-dataset")
    subparsers.add_parser("build-index")
    subparsers.add_parser("train")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("demo")
    subparsers.add_parser("all")
    args = parser.parse_args()

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    if args.command in {"build-dataset", "all"}:
        summary = build_dataset_workflow(PROJECT_ROOT, config)
        print(f"dataset: {summary}")
    if args.command in {"build-index", "all"}:
        metadata = build_index_workflow(PROJECT_ROOT, config)
        print(f"index: {metadata}")
    if args.command in {"train", "all"}:
        state = train_workflow(PROJECT_ROOT, config)
        print(f"train: best_epoch={state['epoch']} mrr@10={state['mrr@10']}")
    if args.command in {"evaluate", "all"}:
        rows = evaluate_workflow(PROJECT_ROOT, config)
        print(f"evaluate: methods={len(rows)}")
    if args.command in {"demo", "all"}:
        metrics = demo_workflow(PROJECT_ROOT, config)
        print(f"demo: accuracy={metrics['accuracy']} mrr@10={metrics['mrr@10']}")


if __name__ == "__main__":
    main()

