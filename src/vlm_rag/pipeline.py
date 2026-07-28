from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .data import build_sample_pages, build_sample_queries, save_dataset
from .config import load_config, resolve_project_path
from .encoders import HashingVLMEncoder, info_nce_loss
from .generator import DoubaoVisionGenerator
from .metrics import accuracy, mrr_at_k, recall_at_k
from .retriever import DualTowerRetriever


def run_demo(
    data_dir: Path,
    output_dir: Path,
    top_k: int = 3,
    project_root: Path | None = None,
) -> dict[str, float]:
    start_time = time.perf_counter()
    if project_root is None:
        project_root = data_dir.resolve().parent
    # Demo 每次运行都会重建一份相对路径数据，保证项目拷贝到其他电脑仍可直接运行。
    pages = build_sample_pages(data_dir / "sample_pages", path_root=project_root)
    queries = build_sample_queries()
    save_dataset(pages, queries, data_dir)

    encoder = HashingVLMEncoder()
    retriever = DualTowerRetriever(encoder)
    retriever.index(pages)
    config_path = resolve_project_path(
        project_root or data_dir.resolve().parent, "configs/config.yaml"
    )
    config = load_config(config_path)

    generator = DoubaoVisionGenerator(
        api_key=config.get_api_key(),
        model=config.generator_doubao_model,
        base_url=config.generator_doubao_base_url,
        max_tokens=config.generator_max_tokens,
        temperature=config.generator_temperature,
        timeout=config.generator_timeout,
    )

    ranked_hits = {}
    predictions = {}
    results = []

    for query in queries:
        # 端到端流程：文本问题 -> Top-K 页面检索 -> 多页证据加权生成。
        hits = retriever.search(query.text, top_k=top_k)
        answer = generator.answer(query.text, hits)
        ranked_hits[query.query_id] = hits
        predictions[query.query_id] = answer.text
        results.append(
            {
                "query": asdict(query),
                "prediction": asdict(answer),
                "hits": [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "page_id": hit.page.page_id,
                        "title": hit.page.title,
                        "image_path": hit.page.image_path,
                    }
                    for hit in hits
                ],
            }
        )

    query_vectors = [encoder.encode_query(query.text) for query in queries]
    positive_pages = [next(page for page in pages if page.page_id == query.positive_page_ids[0]) for query in queries]
    positive_vectors = [encoder.encode_page(page) for page in positive_pages]

    # 同时输出检索、生成、训练目标和耗时指标，方便和基线实验对齐。
    metrics = {
        "mrr@10": round(mrr_at_k(queries, ranked_hits, 10), 4),
        "recall@3": round(recall_at_k(queries, ranked_hits, 3), 4),
        "em": round(accuracy(predictions, queries), 4),
        "accuracy": round(accuracy(predictions, queries), 4),
        "info_nce_loss": round(info_nce_loss(query_vectors, positive_vectors), 4),
        "avg_latency_ms": round((time.perf_counter() - start_time) * 1000 / len(queries), 2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_results.json").write_text(
        json.dumps({"metrics": metrics, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
