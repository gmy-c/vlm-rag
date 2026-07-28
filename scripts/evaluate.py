from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.baselines import evaluate_method
from vlm_rag.config import load_config, resolve_project_path
from vlm_rag.data import build_sample_pages, build_sample_queries, load_pages, load_queries, save_dataset


METHODS = ("vlm_rag", "ocr_rag", "siglip", "colpali")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM-RAG and baseline methods.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    data_dir = resolve_project_path(PROJECT_ROOT, config.data_dir)
    output_dir = resolve_project_path(PROJECT_ROOT, config.output_dir)
    _ensure_dataset(data_dir)

    pages = load_pages(data_dir / "pages.json")
    queries = load_queries(data_dir / "queries.json")
    rows = [
        evaluate_method(method, pages, queries, top_k=config.top_k, embedding_dim=config.embedding_dim)
        for method in METHODS
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "metrics_report.csv"
    fieldnames = ["method", "mrr@10", f"recall@{config.top_k}", "em", "accuracy"]
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Evaluation finished.")
    for row in rows:
        print(
            f"{row['method']}: mrr@10={row['mrr@10']}, "
            f"recall@{config.top_k}={row[f'recall@{config.top_k}']}, accuracy={row['accuracy']}"
        )
    print(f"metrics_report: {_relative_to_project(report_path)}")


def _ensure_dataset(data_dir: Path) -> None:
    if (data_dir / "pages.json").exists() and (data_dir / "queries.json").exists():
        return
    pages = build_sample_pages(data_dir / "sample_pages", path_root=PROJECT_ROOT)
    queries = build_sample_queries()
    save_dataset(pages, queries, data_dir)


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()

