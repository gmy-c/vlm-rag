"""Run one real ColPali optimisation step on a tiny DocVQA sample.

This is intentionally separate from the full training entry point.  It checks
model loading, dataset splitting, device placement, frozen-vision behaviour,
forward/backward propagation, and bounded-batch indexing without writing a
checkpoint.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _resolve_default_data_root() -> Path:
    environment_root = os.environ.get("DOCVQA_DATA_ROOT")
    candidates = (
        Path(environment_root) if environment_root else None,
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "data",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "docvqa_extracted").is_dir():
            return candidate
    return PROJECT_ROOT / "data"


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and path.is_dir():
        return str(path.resolve())

    candidates = (PROJECT_ROOT / path, PROJECT_ROOT.parent / path)
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return value


def _distinct_page_queries(queries: list, count: int, seed: int) -> list:
    rng = random.Random(seed)
    shuffled = list(queries)
    rng.shuffle(shuffled)

    chosen = []
    page_ids: set[str] = set()
    for query in shuffled:
        page_id = query.positive_page_ids[0]
        if page_id in page_ids:
            continue
        chosen.append(query)
        page_ids.add(page_id)
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Could not find {count} queries with distinct pages")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[retrieval/rag] One-step memory-safe ColPali smoke test."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_resolve_default_data_root(),
        help="DocVQA root containing docvqa_extracted/ and docvqa_images/",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("COLPALI_MODEL_PATH", "checkpoint"),
        help="Local checkpoint directory or Hugging Face model ID",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--index-pages", type=int, default=4)
    parser.add_argument("--index-batch-size", type=int, default=1)
    parser.add_argument("--max-query-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--optimizer",
        choices=("torch", "bnb"),
        default="torch",
        help="Optimiser backend used for the one-step test",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Skip backward and optimiser step",
    )
    args = parser.parse_args()

    if args.batch_size < 2 and not args.forward_only:
        parser.error("InfoNCE backward requires --batch-size >= 2")

    # Set allocator behaviour before importing torch.
    if os.name != "nt":
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    from vlm_rag.data import load_docvqa_dataset, split_by_document
    from vlm_rag.encoders import (
        ColPaliDualEncoder,
        ColPaliDualEncoderConfig,
        info_nce_loss,
    )
    from vlm_rag.retriever import DualTowerRetriever

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available in the current environment")

    qa_path = (
        args.data_root
        / "docvqa_extracted"
        / "train_v1.0_withQT.json"
    )
    images_dir = args.data_root / "docvqa_images"
    if not qa_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Invalid data root: {args.data_root.resolve()}"
        )

    pages, queries = load_docvqa_dataset(
        qa_path,
        images_dir,
        split_name="smoke-source",
    )
    splits = split_by_document(
        pages,
        queries,
        train_ratio=0.85,
        val_ratio=0.10,
        seed=args.seed,
    )
    train_pages, train_queries = splits["train"]
    batch_queries = _distinct_page_queries(
        train_queries,
        args.batch_size,
        args.seed,
    )
    page_lookup = {page.page_id: page for page in train_pages}
    batch_pages = [
        page_lookup[query.positive_page_ids[0]]
        for query in batch_queries
    ]

    model_name = _resolve_model_path(args.model)
    print("=" * 64)
    print("Task: retrieval/rag")
    print("ColPali one-step smoke test")
    print(f"  data_root:       {args.data_root.resolve()}")
    print(f"  model:           {model_name}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  index_pages:     {args.index_pages}")
    print(f"  device:          {args.device}")
    if torch.cuda.is_available():
        print(f"  gpu:             {torch.cuda.get_device_name(0)}")
        print(
            "  gpu_total_gb:    "
            f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}"
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("=" * 64)

    encoder = ColPaliDualEncoder(
        ColPaliDualEncoderConfig(
            model_name=model_name,
            device=args.device,
            max_query_length=args.max_query_length,
            gradient_checkpointing=not args.forward_only,
        )
    )
    encoder.train()

    vision_trainable = sum(
        parameter.numel()
        for parameter in encoder.vision_tower.parameters()
        if parameter.requires_grad
    )
    print(f"trainable_parameters: {encoder.trainable_param_count():,}")
    print(f"vision_trainable:     {vision_trainable:,}")
    if vision_trainable:
        raise AssertionError("Vision tower is not fully frozen")

    if args.optimizer == "bnb":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            encoder.trainable_parameters(),
            lr=5e-5,
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            encoder.trainable_parameters(),
            lr=5e-5,
            weight_decay=0.01,
        )
    optimizer.zero_grad(set_to_none=True)

    query_vectors = encoder.encode_query_batch(
        [query.text for query in batch_queries]
    )
    page_vectors = encoder.encode_page_batch(batch_pages)
    loss = info_nce_loss(query_vectors, page_vectors, temperature=0.07)
    print(f"loss:                 {loss.item():.6f}")

    if not args.forward_only:
        loss.backward()
        layer_grad = encoder.layer_weights.grad
        if layer_grad is None or not torch.isfinite(layer_grad).all():
            raise AssertionError("Layer pooling weights have no finite grad")
        if any(
            parameter.grad is not None
            for parameter in encoder.vision_tower.parameters()
        ):
            raise AssertionError("Frozen vision tower unexpectedly has grads")
        lora_grad_count = sum(
            1
            for parameter in encoder.language_model.parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
        )
        projection_grad_count = sum(
            1
            for module in (encoder.text_proj, encoder.vision_proj)
            for parameter in module.parameters()
            if parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
        )
        if not lora_grad_count:
            raise AssertionError("LoRA parameters have no finite gradients")
        if not projection_grad_count:
            raise AssertionError("Projection heads have no finite gradients")
        optimizer.step()
        print(
            "layer_weight_grad:    "
            + ",".join(f"{value:.6g}" for value in layer_grad.tolist())
        )
        print(f"lora_grad_tensors:    {lora_grad_count}")
        print(f"projection_grads:     {projection_grad_count}")

    # Exercise the memory-bounded indexing path as part of the same test.
    encoder.eval()
    index_pages = train_pages[: max(1, args.index_pages)]
    retriever = DualTowerRetriever(encoder)
    retriever.index(
        index_pages,
        batch_size=args.index_batch_size,
        storage_device="cpu",
    )
    hits = retriever.search(batch_queries[0].text, top_k=min(3, len(index_pages)))
    print(f"indexed_pages:         {len(index_pages)}")
    print(f"search_hits:           {len(hits)}")

    if torch.cuda.is_available():
        print(
            "cuda_allocated_gb:    "
            f"{torch.cuda.memory_allocated() / 2**30:.3f}"
        )
        print(
            "cuda_peak_gb:         "
            f"{torch.cuda.max_memory_allocated() / 2**30:.3f}"
        )
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
