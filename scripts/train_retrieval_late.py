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
from vlm_rag.retrieval.dataset import load_hard_negative_map
from vlm_rag.retrieval.losses import HybridLossConfig
from vlm_rag.retrieval.model import LateInteractionModelConfig
from vlm_rag.retrieval.schema import load_retrieval_manifest
from vlm_rag.retrieval.training import (
    LateTrainingConfig,
    LoaderConfig,
    train_late_interaction_retriever,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train native ColPali multi-vector late interaction."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval_late_5090.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--hard-negatives", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--pages-per-batch", type=int, default=None)
    parser.add_argument("--queries-per-page", type=int, default=None)
    parser.add_argument("--hard-negatives-per-query", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--retrieval-validation-queries", type=int, default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help=(
            "Initialize all late-interaction weights from an adapter without "
            "restoring optimizer state (use this for legacy checkpoints)."
        ),
    )
    parser.add_argument(
        "--init-global",
        type=Path,
        default=None,
        help="Path to the global-stage `lora` directory.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_dir = args.manifest_dir or (
        args.data_root / "manifests" / "retrieval"
    )
    all_records = load_retrieval_manifest(manifest_dir / "all.jsonl")
    page_paths = {
        record.positive_page_id: record.image_path for record in all_records
    }
    hard_path = args.hard_negatives
    if hard_path is None:
        configured = Path(str(raw["hard_negatives"]))
        hard_path = (
            configured
            if configured.is_absolute()
            else args.data_root / "manifests" / "retrieval" / configured.name
        )
    hard_map = (
        load_hard_negative_map(hard_path)
        if hard_path.is_file()
        else {}
    )
    grouped_training = (
        args.pages_per_batch
        if args.pages_per_batch is not None
        else int(raw.get("pages_per_batch", 0))
    ) > 0
    configured_queries_per_page = (
        args.queries_per_page
        if args.queries_per_page is not None
        else int(raw.get("queries_per_page", 1))
    )
    train_dataset = RetrievalManifestDataset(
        manifest_dir / "train.jsonl",
        args.data_root,
        decode_images=not grouped_training,
        page_paths=page_paths,
        hard_negative_map=hard_map,
        hard_negatives_per_query=(
            args.hard_negatives_per_query
            if args.hard_negatives_per_query is not None
            else (
                int(raw["hard_negatives_per_query"])
                if raw.get("hard_negatives_per_query") is not None
                else None
            )
        ),
        rotate_hard_negatives=bool(raw.get("rotate_hard_negatives", False)),
    )
    val_dataset = RetrievalManifestDataset(
        manifest_dir / "val.jsonl",
        args.data_root,
        page_paths=page_paths,
    )
    if args.max_train_samples is not None:
        train_dataset.records = (
            _page_grouped_subset(
                train_dataset.records,
                args.max_train_samples,
                configured_queries_per_page,
            )
            if grouped_training
            else _diverse_subset(
                train_dataset.records,
                args.max_train_samples,
            )
        )
    if args.max_val_samples is not None:
        val_dataset.records = _diverse_subset(
            val_dataset.records,
            args.max_val_samples,
        )
    loss_raw = raw["loss"]
    loss = HybridLossConfig(
        global_weight=float(loss_raw["global_weight"]),
        late_weight=float(loss_raw["late_weight"]),
        hard_negative_weight=float(loss_raw["hard_negative_weight"]),
        global_temperature=float(loss_raw["global_temperature"]),
        late_temperature=float(loss_raw["late_temperature"]),
        hard_negative_margin=float(loss_raw["hard_negative_margin"]),
        maxsim_backend=str(loss_raw["maxsim_backend"]),
        maxsim_normalization=str(
            loss_raw.get("maxsim_normalization", "mean")
        ),
        query_batch_chunk=int(loss_raw["query_batch_chunk"]),
        document_batch_chunk=int(loss_raw["document_batch_chunk"]),
        query_token_chunk=int(loss_raw["query_token_chunk"]),
    )
    model_config = LateInteractionModelConfig(
        checkpoint_path=args.model,
        lora_rank=int(raw["lora_rank"]),
        lora_alpha=int(raw["lora_alpha"]),
        lora_dropout=float(raw["lora_dropout"]),
        use_rslora=bool(raw["use_rslora"]),
        max_query_length=int(raw["max_query_length"]),
        gradient_checkpointing=(
            args.gradient_checkpointing
            if args.gradient_checkpointing is not None
            else bool(raw["gradient_checkpointing"])
        ),
        image_only_tokens=bool(raw["image_only_tokens"]),
        global_dim=int(raw["global_dim"]),
    )
    training = LateTrainingConfig(
        micro_batch_size=args.micro_batch_size or int(raw["micro_batch_size"]),
        pages_per_batch=(
            args.pages_per_batch
            if args.pages_per_batch is not None
            else int(raw.get("pages_per_batch", 0))
        ),
        queries_per_page=configured_queries_per_page,
        hard_negatives_per_query=(
            args.hard_negatives_per_query
            if args.hard_negatives_per_query is not None
            else (
                int(raw["hard_negatives_per_query"])
                if raw.get("hard_negatives_per_query") is not None
                else None
            )
        ),
        rotate_hard_negatives=bool(raw.get("rotate_hard_negatives", False)),
        page_forward_chunk_size=int(raw.get("page_forward_chunk_size", 8)),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            or int(raw["gradient_accumulation_steps"])
        ),
        epochs=args.epochs or int(raw["epochs"]),
        lora_learning_rate=float(raw["lora_learning_rate"]),
        projection_learning_rate=float(raw["projection_learning_rate"]),
        pooling_learning_rate=float(raw["pooling_learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        warmup_ratio=float(raw["warmup_ratio"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        queue_size=int(raw["queue_size"]),
        validation_batches=int(raw["validation_batches"]),
        retrieval_validation_queries=(
            args.retrieval_validation_queries
            if args.retrieval_validation_queries is not None
            else int(raw.get("retrieval_validation_queries", 0))
        ),
        retrieval_coarse_top_k=int(raw.get("retrieval_coarse_top_k", 128)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 0)),
        progress=(
            args.progress
            if args.progress is not None
            else bool(raw.get("progress", True))
        ),
        log_every=int(raw.get("log_every", 25)),
        loss=loss,
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
    output = args.output_dir or PROJECT_ROOT / str(raw["output_dir"])
    summary = train_late_interaction_retriever(
        model_config,
        train_dataset,
        val_dataset,
        output.resolve(),
        training,
        loader,
        device=args.device or str(raw["device"]),
        resume_from=args.resume,
        initialize_lora_from=args.init_global,
        initialize_adapter_from=args.init_adapter,
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


def _page_grouped_subset(records, maximum: int, queries_per_page: int):
    by_page = {}
    for record in records:
        by_page.setdefault(record.positive_page_id, []).append(record)
    result = []
    for rows in by_page.values():
        result.extend(rows[:queries_per_page])
        if len(result) >= maximum:
            return result[:maximum]
    return result


if __name__ == "__main__":
    raise SystemExit(main())
