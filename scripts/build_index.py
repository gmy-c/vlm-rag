from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.config import load_config, resolve_project_path
from vlm_rag.data import build_sample_pages, build_sample_queries, load_pages, save_dataset
from vlm_rag.encoders import EncoderConfig, HashingVLMEncoder
from vlm_rag.index_store import build_vector_index
from vlm_rag.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist page vector index.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))
    data_dir = resolve_project_path(PROJECT_ROOT, config.data_dir)
    index_dir = resolve_project_path(PROJECT_ROOT, config.index_dir)
    logger = configure_logging(resolve_project_path(PROJECT_ROOT, config.log_dir))
    _ensure_dataset(data_dir)

    pages = load_pages(data_dir / "pages.json")
    encoder_config = EncoderConfig(
        dim=config.embedding_dim,
        hidden_layer_weights=config.hidden_layer_weights,
    )
    encoder = HashingVLMEncoder(encoder_config)
    index = build_vector_index(pages, encoder, index_dir, encoder_config)
    logger.info("index built: pages=%s dim=%s", index.metadata["page_count"], index.metadata["embedding_dim"])

    print("Index built.")
    print(f"page_count: {index.metadata['page_count']}")
    print(f"index_dir: {_relative_to_project(index_dir)}")


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

