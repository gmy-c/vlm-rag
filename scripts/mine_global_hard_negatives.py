from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.encoders import ColPaliDualEncoder
from vlm_rag.retrieval.dataset import resolve_retrieval_path
from vlm_rag.retrieval.index import unique_pages
from vlm_rag.retrieval.schema import load_retrieval_manifest


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine first-round hard negatives with the global retriever."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = load_retrieval_manifest(args.manifest)
    query_records = (
        records[: args.max_queries]
        if args.max_queries is not None
        else records
    )
    pages = unique_pages(records)
    model = ColPaliDualEncoder.load(
        args.checkpoint,
        base_model=args.base_model,
        device=args.device,
    ).eval()
    page_vectors = []
    for start in range(0, len(pages), args.page_batch_size):
        batch = pages[start : start + args.page_batch_size]
        images = []
        for record in batch:
            with Image.open(
                resolve_retrieval_path(args.data_root, record.image_path)
            ) as image:
                images.append(image.convert("RGB"))
        page_vectors.append(model._encode_images(images).detach().cpu())
    page_matrix = torch.cat(page_vectors).to(
        next(model.parameters()).device
    )
    page_ids = [record.positive_page_id for record in pages]
    doc_ids = [record.doc_id for record in pages]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(query_records), args.query_batch_size):
            batch = query_records[start : start + args.query_batch_size]
            queries = model.encode_query_batch(
                [record.query_text for record in batch]
            )
            scores = queries @ page_matrix.T
            candidate_count = min(
                args.candidate_top_k,
                scores.shape[1],
            )
            candidate_indices = torch.topk(
                scores,
                k=candidate_count,
                dim=1,
            ).indices.cpu()
            for record, row in zip(batch, candidate_indices):
                selected = []
                for index in row.tolist():
                    if (
                        page_ids[index] == record.positive_page_id
                        or doc_ids[index] == record.doc_id
                    ):
                        continue
                    selected.append(page_ids[index])
                    if len(selected) >= args.negatives_per_query:
                        break
                handle.write(
                    json.dumps(
                        {
                            "query_id": record.query_id,
                            "positive_page_id": record.positive_page_id,
                            "negative_page_ids": selected,
                            "source": "global_retriever",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(
        {
            "queries": len(query_records),
            "pages": len(pages),
            "output": str(args.output.resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
