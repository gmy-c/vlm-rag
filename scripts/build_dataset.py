from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.config import load_config, resolve_project_path
from vlm_rag.data import build_sample_pages, build_sample_queries, save_dataset
from vlm_rag.dataset_split import split_queries
from vlm_rag.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic VLM-RAG dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    data_dir = resolve_project_path(PROJECT_ROOT, args.data_dir or config.data_dir)
    logger = configure_logging(resolve_project_path(PROJECT_ROOT, config.log_dir))
    pages = build_sample_pages(data_dir / "sample_pages", path_root=PROJECT_ROOT)
    queries = build_sample_queries()
    save_dataset(pages, queries, data_dir)
    splits = split_queries(queries, data_dir / "splits", config.train_ratio, config.dev_ratio)
    logger.info(
        "dataset built: pages=%s queries=%s train=%s dev=%s test=%s",
        len(pages),
        len(queries),
        len(splits["train"]),
        len(splits["dev"]),
        len(splits["test"]),
    )

    print("Dataset built.")
    print(f"pages: {len(pages)}")
    print(f"queries: {len(queries)}")
    print(f"train_queries: {len(splits['train'])}")
    print(f"dev_queries: {len(splits['dev'])}")
    print(f"test_queries: {len(splits['test'])}")
    print(f"pages_json: {_relative_to_project(data_dir / 'pages.json')}")
    print(f"queries_json: {_relative_to_project(data_dir / 'queries.json')}")


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
