from __future__ import annotations

import csv
from pathlib import Path

from .baselines import evaluate_method
from .config import ProjectConfig, resolve_project_path
from .data import build_sample_pages, build_sample_queries, load_pages, load_queries, save_dataset
from .dataset_split import split_queries
from .encoders import EncoderConfig, HashingVLMEncoder
from .index_store import build_vector_index
from .logging_utils import configure_logging
from .pipeline import run_demo
from .training import train_retriever


METHODS = ("vlm_rag", "ocr_rag", "siglip", "colpali")


def build_dataset_workflow(project_root: Path, config: ProjectConfig) -> dict[str, int]:
    data_dir = resolve_project_path(project_root, config.data_dir)
    logger = configure_logging(resolve_project_path(project_root, config.log_dir))
    pages = build_sample_pages(data_dir / "sample_pages", path_root=project_root)
    queries = build_sample_queries()
    save_dataset(pages, queries, data_dir)
    splits = split_queries(queries, data_dir / "splits", config.train_ratio, config.dev_ratio)
    summary = {
        "pages": len(pages),
        "queries": len(queries),
        "train": len(splits["train"]),
        "dev": len(splits["dev"]),
        "test": len(splits["test"]),
    }
    logger.info("dataset workflow finished: %s", summary)
    return summary


def build_index_workflow(project_root: Path, config: ProjectConfig) -> dict[str, object]:
    data_dir = resolve_project_path(project_root, config.data_dir)
    index_dir = resolve_project_path(project_root, config.index_dir)
    _ensure_dataset(project_root, data_dir)
    pages = load_pages(data_dir / "pages.json")
    encoder_config = EncoderConfig(config.embedding_dim, config.hidden_layer_weights)
    encoder = HashingVLMEncoder(encoder_config)
    index = build_vector_index(pages, encoder, index_dir, encoder_config)
    configure_logging(resolve_project_path(project_root, config.log_dir)).info(
        "index workflow finished: %s",
        index.metadata,
    )
    return index.metadata


def train_workflow(project_root: Path, config: ProjectConfig) -> dict[str, object]:
    data_dir = resolve_project_path(project_root, config.data_dir)
    model_dir = resolve_project_path(project_root, config.model_dir)
    _ensure_dataset(project_root, data_dir)
    pages = load_pages(data_dir / "pages.json")
    queries = load_queries(data_dir / "queries.json")
    state = train_retriever(
        pages,
        queries,
        model_dir=model_dir,
        embedding_dim=config.embedding_dim,
        temperature=config.temperature,
        epochs=config.epochs,
    )
    configure_logging(resolve_project_path(project_root, config.log_dir)).info(
        "training workflow finished: %s",
        state,
    )
    return state


def evaluate_workflow(project_root: Path, config: ProjectConfig) -> list[dict[str, float | str]]:
    data_dir = resolve_project_path(project_root, config.data_dir)
    output_dir = resolve_project_path(project_root, config.output_dir)
    _ensure_dataset(project_root, data_dir)
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
    configure_logging(resolve_project_path(project_root, config.log_dir)).info(
        "evaluation workflow finished: report=%s",
        _relative_to_project(project_root, report_path),
    )
    return rows


def demo_workflow(project_root: Path, config: ProjectConfig) -> dict[str, float]:
    metrics = run_demo(
        resolve_project_path(project_root, config.data_dir),
        resolve_project_path(project_root, config.output_dir),
        top_k=config.top_k,
        project_root=project_root,
    )
    configure_logging(resolve_project_path(project_root, config.log_dir)).info(
        "demo workflow finished: %s",
        metrics,
    )
    return metrics


def _ensure_dataset(project_root: Path, data_dir: Path) -> None:
    if (data_dir / "pages.json").exists() and (data_dir / "queries.json").exists():
        return
    pages = build_sample_pages(data_dir / "sample_pages", path_root=project_root)
    queries = build_sample_queries()
    save_dataset(pages, queries, data_dir)


def _relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()
