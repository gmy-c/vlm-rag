"""
ColPali 双塔检索器完整训练入口。

Usage:
    python scripts/train_colpali.py

环境要求:
    - GPU: NVIDIA RTX 4090 24GB 或更高
    - CUDA + PyTorch + colpali-engine + peft + bitsandbytes + flash-attn
    - DocVQA 数据集已就绪于 data/ 目录
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[retrieval/rag] Train the ColPali dual-tower retriever on full DocVQA."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="Override local checkpoint path or Hugging Face model ID",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=None,
        help="Pages per GPU batch during validation indexing",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Full DocVQA root; never data/desensitized. "
             "Defaults to DOCVQA_DATA_ROOT or a discovered data/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Retrieval checkpoint directory. Defaults to RETRIEVAL_OUTPUT_DIR "
             "or models/colpali_retriever.",
    )
    parser.add_argument(
        "--train-qa",
        default="docvqa_extracted/train_v1.0_withQT.json",
        help="Path to train Q&A JSON",
    )
    parser.add_argument(
        "--images-dir",
        default="docvqa_images",
        help="Path to page images directory",
    )
    args = parser.parse_args()

    # ── 加载配置 ──
    from vlm_rag.config import load_config, resolve_project_path

    config = load_config(resolve_project_path(PROJECT_ROOT, args.config))

    device = args.device or config.device
    epochs = args.epochs or config.epochs
    batch_size = args.batch_size or config.batch_size
    index_batch_size = args.index_batch_size or config.index_batch_size
    lr = args.lr or config.learning_rate

    # ── 解析数据路径 ──
    data_root = _resolve_data_root(args.data_root)
    train_qa_path = data_root / args.train_qa if not Path(args.train_qa).is_absolute() else Path(args.train_qa)
    images_dir = data_root / args.images_dir if not Path(args.images_dir).is_absolute() else Path(args.images_dir)

    # ── 加载数据 ──
    from vlm_rag.data import load_docvqa_dataset, split_by_document

    print("=" * 60)
    print("Loading DocVQA dataset...")
    print(f"  QA JSON:    {train_qa_path}")
    print(f"  Images dir: {images_dir}")
    print("=" * 60)

    all_pages, all_queries = load_docvqa_dataset(
        train_qa_path,
        images_dir,
        split_name="train",
    )

    # 从 train 中分出 train/val（用于训练过程中验证）
    # test 用 DocVQA 自带的 test set，保持公平
    print("\nSplitting by document...")
    splits = split_by_document(
        all_pages,
        all_queries,
        train_ratio=0.85,  # 85% 用于训练
        val_ratio=0.10,    # 10% 用于验证
    )

    train_pages, train_queries = splits["train"]
    val_pages, val_queries = splits["val"]

    # ── 解析模型路径（本地路径 → 绝对路径，HF ID → 保持原样）──
    model_name = (
        args.model
        or os.environ.get("COLPALI_MODEL_PATH")
        or config.colpali_model
    )
    model_path = Path(model_name)
    local_candidates = (
        model_path,
        PROJECT_ROOT / model_path,
        PROJECT_ROOT.parent / model_path,
    )
    for local_path in local_candidates:
        if local_path.is_dir():
            model_name = str(local_path.resolve())
            print(f"Using local checkpoint: {model_name}")
            break
    else:
        print(f"Using HuggingFace model: {model_name}")
    from vlm_rag.training import train_colpali_retriever
    model_dir = Path(
        args.output_dir
        or os.environ.get(
            "RETRIEVAL_OUTPUT_DIR",
            str(PROJECT_ROOT / "models" / "colpali_retriever"),
        )
    ).expanduser().resolve()

    print("\n" + "=" * 60)
    print("Task:             retrieval/rag")
    print("Training Configuration:")
    print(f"  Device:           {device}")
    print(f"  Train pages:      {len(train_pages)}")
    print(f"  Train queries:    {len(train_queries)}")
    print(f"  Val pages:        {len(val_pages)}")
    print(f"  Val queries:      {len(val_queries)}")
    print(f"  Epochs:           {epochs}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Grad accum steps: {config.gradient_accumulation_steps}")
    print(f"  Index batch size: {index_batch_size}")
    print(
        f"  Effective batch:  "
        f"{batch_size * config.gradient_accumulation_steps}"
    )
    print(f"  Learning rate:    {lr}")
    print(f"  Temperature:      {config.temperature}")
    print(f"  Model dir:        {model_dir}")
    print("=" * 60 + "\n")

    best = train_colpali_retriever(
        train_pages,
        train_queries,
        val_pages,
        val_queries,
        model_dir=model_dir,
        model_name=model_name,
        proj_dim=config.embedding_dim,
        selected_layers=config.get_selected_layers(),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        max_query_length=config.max_query_length,
        batch_size=batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=lr,
        epochs=epochs,
        temperature=config.temperature,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        device=device,
        index_batch_size=index_batch_size,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best epoch:   {best['epoch']}")
    print(f"  Best MRR@10:  {best['mrr@10']}")
    print(f"  Best Recall@3:{best['recall@3']}")
    print(f"  Model saved:  {model_dir}")
    print("=" * 60)


def _resolve_data_root(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    elif os.environ.get("DOCVQA_DATA_ROOT"):
        candidates.append(Path(os.environ["DOCVQA_DATA_ROOT"]))
    candidates.extend((PROJECT_ROOT / "data", PROJECT_ROOT.parent / "data"))
    for candidate in candidates:
        if (
            (candidate / "docvqa_extracted").is_dir()
            and (candidate / "docvqa_images").is_dir()
        ):
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "Full DocVQA root not found. Pass --data-root or set DOCVQA_DATA_ROOT. "
        "The root must contain docvqa_extracted/ and docvqa_images/."
    )


if __name__ == "__main__":
    main()
