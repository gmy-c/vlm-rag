from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .data import Page
from .encoders import EncoderConfig, HashingVLMEncoder, cosine_similarity
from .retriever import SearchHit


@dataclass(frozen=True)
class VectorIndex:
    pages: list[Page]
    vectors: dict[str, list[float]]
    metadata: dict[str, object]

    def search(self, encoder: HashingVLMEncoder, query: str, top_k: int) -> list[SearchHit]:
        query_vector = encoder.encode_query(query)
        scored = [
            (page, cosine_similarity(query_vector, self.vectors[page.page_id]))
            for page in self.pages
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchHit(page=page, score=score, rank=rank)
            for rank, (page, score) in enumerate(scored[:top_k], start=1)
        ]


def build_vector_index(
    pages: list[Page],
    encoder: HashingVLMEncoder,
    index_dir: Path,
    config: EncoderConfig,
) -> VectorIndex:
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = {page.page_id: encoder.encode_page(page) for page in pages}
    metadata = {
        "page_count": len(pages),
        "embedding_dim": config.dim,
        "hidden_layer_weights": config.hidden_layer_weights,
    }
    index = VectorIndex(pages=pages, vectors=vectors, metadata=metadata)
    save_vector_index(index, index_dir)
    return index


def save_vector_index(index: VectorIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "page_vectors.json").write_text(
        json.dumps(index.vectors, ensure_ascii=False),
        encoding="utf-8",
    )
    (index_dir / "index_metadata.json").write_text(
        json.dumps(index.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vector_index(pages: list[Page], index_dir: Path) -> VectorIndex:
    vectors_path = index_dir / "page_vectors.json"
    metadata_path = index_dir / "index_metadata.json"
    if not vectors_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("vector index files are missing; run build-index first")
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return VectorIndex(pages=pages, vectors=vectors, metadata=metadata)

