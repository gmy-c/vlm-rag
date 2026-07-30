from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.retrieval import RetrievalManifestDataset
from vlm_rag.retrieval.training import (
    GlobalTrainingConfig,
    LoaderConfig,
    train_global_retriever,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the large-batch global stage of hybrid retrieval."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval_global_5090.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = args.manifest or (
        args.data_root / "manifests" / "retrieval" / "train.jsonl"
    )
    output = args.output_dir or PROJECT_ROOT / str(raw["output_dir"])
    dataset = RetrievalManifestDataset(manifest, args.data_root, decode_images=True)
    if args.max_train_samples is not None:
        dataset.records = _diverse_subset(
            dataset.records,
            args.max_train_samples,
        )
    training = GlobalTrainingConfig(
        batch_size=args.batch_size or int(raw["batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        epochs=args.epochs or int(raw["epochs"]),
        lora_rank=int(raw["lora_rank"]),
        lora_alpha=int(raw["lora_alpha"]),
        lora_dropout=float(raw.get("lora_dropout", 0.05)),
        use_rslora=bool(raw.get("use_rslora", True)),
        max_query_length=int(raw["max_query_length"]),
        selected_layers=tuple(int(item) for item in raw["selected_layers"]),
        gradient_checkpointing=bool(raw["gradient_checkpointing"]),
        lora_learning_rate=float(raw["lora_learning_rate"]),
        projection_learning_rate=float(raw["projection_learning_rate"]),
        layer_weight_learning_rate=float(raw["layer_weight_learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        temperature=float(raw["temperature"]),
        warmup_ratio=float(raw["warmup_ratio"]),
        max_grad_norm=float(raw["max_grad_norm"]),
    )
    loader = LoaderConfig(
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(raw["num_workers"])
        ),
        pin_memory=bool(raw["pin_memory"]),
        persistent_workers=bool(raw["persistent_workers"]),
        prefetch_factor=int(raw["prefetch_factor"]),
    )
    summary = train_global_retriever(
        args.model,
        dataset,
        output.resolve(),
        training,
        loader,
        device=args.device or str(raw["device"]),
    )
    print(summary)
    return 0


def _diverse_subset(records, maximum: int):
    first_by_doc = {}
    remainder = []
    for record in records:
        if record.doc_id not in first_by_doc:
            first_by_doc[record.doc_id] = record
        else:
            remainder.append(record)
    return (list(first_by_doc.values()) + remainder)[:maximum]


if __name__ == "__main__":
    raise SystemExit(main())
