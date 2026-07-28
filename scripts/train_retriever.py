from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.config import load_config, resolve_project_path
from vlm_rag.data import build_sample_pages, build_sample_queries, load_pages, load_queries, save_dataset
from vlm_rag.training import train_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight dual-tower retriever.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    data_dir = resolve_project_path(PROJECT_ROOT, config.data_dir)
    model_dir = resolve_project_path(PROJECT_ROOT, config.model_dir)
    _ensure_dataset(data_dir)

    pages = load_pages(data_dir / "pages.json")
    queries = load_queries(data_dir / "queries.json")
    best_state = train_retriever(
        pages,
        queries,
        model_dir=model_dir,
        embedding_dim=config.embedding_dim,
        temperature=config.temperature,
        epochs=config.epochs,
    )

    print("Retriever training finished.")
    print(f"best_epoch: {best_state['epoch']}")
    print(f"best_weights: {','.join(str(weight) for weight in best_state['hidden_layer_weights'])}")
    print(f"mrr@10: {best_state['mrr@10']}")
    print(f"recall@3: {best_state['recall@3']}")
    print(f"model_config: {_relative_to_project(model_dir / 'retriever_config.json')}")
    print(f"training_log: {_relative_to_project(model_dir / 'training_log.csv')}")


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

