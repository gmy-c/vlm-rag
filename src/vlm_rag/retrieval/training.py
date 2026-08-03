from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vlm_rag.encoders import ColPaliDualEncoder, ColPaliDualEncoderConfig

from .dataset import (
    DocumentUniqueBatchSampler,
    PageGroupedBatchSampler,
    PageGroupedRetrievalBatch,
    RetrievalManifestDataset,
    page_grouped_retrieval_collate,
    retrieval_collate,
)
from .index import MultiVectorIndex, build_multivector_index
from .losses import (
    HybridLossConfig,
    MultiVectorMemoryQueue,
    hybrid_retrieval_loss,
    symmetric_global_info_nce,
)
from .maxsim import resolve_maxsim_backend
from .model import LateInteractionModelConfig, LateInteractionRetriever


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass(frozen=True, slots=True)
class GlobalTrainingConfig:
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    max_query_length: int = 96
    selected_layers: tuple[int, ...] = (0, 8, 16, 23, 27)
    gradient_checkpointing: bool = False
    lora_learning_rate: float = 5e-5
    projection_learning_rate: float = 2e-4
    layer_weight_learning_rate: float = 5e-4
    weight_decay: float = 0.01
    temperature: float = 0.05
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass(frozen=True, slots=True)
class LateTrainingConfig:
    micro_batch_size: int = 4
    pages_per_batch: int = 0
    queries_per_page: int = 1
    hard_negatives_per_query: int | None = None
    rotate_hard_negatives: bool = False
    page_forward_chunk_size: int = 8
    gradient_accumulation_steps: int = 24
    epochs: int = 8
    lora_learning_rate: float = 5e-5
    projection_learning_rate: float = 1e-4
    pooling_learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    queue_size: int = 256
    validation_batches: int = 20
    retrieval_validation_queries: int = 0
    retrieval_coarse_top_k: int = 128
    early_stopping_patience: int = 0
    progress: bool = True
    log_every: int = 25
    seed: int = 42
    loss: HybridLossConfig = HybridLossConfig()


def build_retrieval_loader(
    dataset: RetrievalManifestDataset,
    batch_size: int,
    loader: LoaderConfig,
    *,
    seed: int,
    drop_last: bool,
) -> tuple[DataLoader, DocumentUniqueBatchSampler]:
    sampler = DocumentUniqueBatchSampler(
        dataset.records,
        batch_size,
        seed=seed,
        drop_last=drop_last,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": retrieval_collate,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs), sampler


def build_page_grouped_retrieval_loader(
    dataset: RetrievalManifestDataset,
    config: LateTrainingConfig,
    loader: LoaderConfig,
) -> tuple[DataLoader, PageGroupedBatchSampler]:
    sampler = PageGroupedBatchSampler(
        dataset.records,
        config.pages_per_batch,
        config.queries_per_page,
        seed=config.seed,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": page_grouped_retrieval_collate,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs), sampler


def train_global_retriever(
    model_path: str,
    train_dataset: RetrievalManifestDataset,
    output_dir: Path,
    config: GlobalTrainingConfig,
    loader_config: LoaderConfig,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    _seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = ColPaliDualEncoder(
        ColPaliDualEncoderConfig(
            model_name=model_path,
            device=device,
            proj_dim=768,
            selected_layers=config.selected_layers,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            max_query_length=config.max_query_length,
            gradient_checkpointing=config.gradient_checkpointing,
        )
    )
    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in encoder.language_model.parameters()
                if parameter.requires_grad
            ],
            "lr": config.lora_learning_rate,
            "name": "language_lora",
        },
        {
            "params": list(encoder.vision_proj.parameters())
            + list(encoder.text_proj.parameters()),
            "lr": config.projection_learning_rate,
            "name": "projection_heads",
        },
        {
            "params": [encoder.layer_weights],
            "lr": config.layer_weight_learning_rate,
            "name": "layer_weights",
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
        fused=device.startswith("cuda"),
    )
    data_loader, sampler = build_retrieval_loader(
        train_dataset,
        config.batch_size,
        loader_config,
        seed=config.seed,
        drop_last=True,
    )
    update_steps = max(
        1,
        math.ceil(
            len(data_loader) / config.gradient_accumulation_steps
        )
        * config.epochs,
    )
    scheduler = _warmup_cosine_scheduler(
        optimizer,
        update_steps,
        config.warmup_ratio,
    )
    global_step = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        micro_steps = 0
        for batch_index, batch in enumerate(data_loader, start=1):
            query_vectors = encoder.encode_query_batch(batch.queries)
            page_vectors = encoder._encode_images(batch.positive_images)
            raw_loss = symmetric_global_info_nce(
                query_vectors,
                page_vectors,
                temperature=config.temperature,
            )
            (raw_loss / config.gradient_accumulation_steps).backward()
            running += float(raw_loss.detach())
            micro_steps += 1
            if (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(data_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    encoder.trainable_parameters(),
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        average = running / max(1, micro_steps)
        history.append({"epoch": epoch, "loss": average})
        encoder.save(output_dir)
        _write_json(
            output_dir / "global_training_state.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "loss": average,
                "config": asdict(config),
                "history": history,
                "peak_allocated_gb": _peak_allocated(),
                "peak_reserved_gb": _peak_reserved(),
            },
        )
        print(f"global epoch {epoch}: loss={average:.6f}")
    return json.loads(
        (output_dir / "global_training_state.json").read_text(encoding="utf-8")
    )


def train_late_interaction_retriever(
    model_config: LateInteractionModelConfig,
    train_dataset: RetrievalManifestDataset,
    val_dataset: RetrievalManifestDataset,
    output_dir: Path,
    config: LateTrainingConfig,
    loader_config: LoaderConfig,
    *,
    device: str = "cuda",
    resume_from: Path | None = None,
    initialize_lora_from: Path | None = None,
    initialize_adapter_from: Path | None = None,
) -> dict[str, Any]:
    _seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _training_logger(output_dir)
    if config.pages_per_batch > 0 and config.loss.maxsim_backend == "auto":
        raise ValueError(
            "Grouped production training requires an explicit MaxSim backend; "
            "set loss.maxsim_backend to 'lik' or 'chunked'."
        )
    resolved_backend = resolve_maxsim_backend(config.loss.maxsim_backend)
    if config.loss.maxsim_normalization not in ("mean", "sum"):
        raise ValueError(
            "loss.maxsim_normalization must be 'mean' or 'sum'"
        )
    model = LateInteractionRetriever(model_config, device=device)
    initializers = [
        value
        for value in (
            resume_from,
            initialize_lora_from,
            initialize_adapter_from,
        )
        if value is not None
    ]
    if len(initializers) > 1:
        raise ValueError(
            "resume_from, initialize_lora_from and initialize_adapter_from "
            "are mutually exclusive"
        )
    if initialize_lora_from is not None:
        model.load_language_adapter(initialize_lora_from)
        logger.info("Initialized language LoRA from %s", initialize_lora_from)
    if initialize_adapter_from is not None:
        model.load_adapter(initialize_adapter_from)
        logger.info("Initialized full adapter from %s", initialize_adapter_from)
    if resume_from is not None:
        model.load_adapter(resume_from)
        logger.info("Loaded adapter for exact resume from %s", resume_from)
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            lora_lr=config.lora_learning_rate,
            projection_lr=config.projection_learning_rate,
            pooling_lr=config.pooling_learning_rate,
            weight_decay=config.weight_decay,
        ),
        fused=device.startswith("cuda"),
    )
    grouped = config.pages_per_batch > 0
    if grouped:
        train_loader, sampler = build_page_grouped_retrieval_loader(
            train_dataset,
            config,
            loader_config,
        )
    else:
        train_loader, sampler = build_retrieval_loader(
            train_dataset,
            config.micro_batch_size,
            loader_config,
            seed=config.seed,
            drop_last=True,
        )
    val_loader, _ = build_retrieval_loader(
        val_dataset,
        config.micro_batch_size,
        LoaderConfig(num_workers=0),
        seed=config.seed + 1,
        drop_last=False,
    )
    total_updates = max(
        1,
        math.ceil(
            len(train_loader) / config.gradient_accumulation_steps
        )
        * config.epochs,
    )
    scheduler = _warmup_cosine_scheduler(
        optimizer,
        total_updates,
        config.warmup_ratio,
    )
    queue = MultiVectorMemoryQueue(config.queue_size)
    global_step = 0
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    best_retrieval: dict[str, Any] | None = None
    baseline_retrieval: dict[str, Any] | None = None
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_from is not None:
        state_path = resume_from / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(
                "Exact --resume requires training_state.pt. Use "
                "--init-adapter for legacy adapter-only checkpoints: "
                f"{state_path}"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_training_config = state.get("training_config")
        current_training_config = {
            **asdict(config),
            "loss": asdict(config.loss),
        }
        if (
            saved_training_config is not None
            and saved_training_config != current_training_config
        ):
            raise ValueError(
                "Exact --resume requires the unchanged training config. "
                "Use --init-adapter to start a new fine-tuning run."
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        queue.load_state_dict(state.get("queue", {}), device=device)
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        history = list(state.get("history", []))
        best_retrieval = state.get("best_retrieval")
        baseline_retrieval = state.get("baseline_retrieval")
        epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        logger.info(
            "Restored optimizer/scheduler at epoch=%d global_step=%d",
            start_epoch - 1,
            global_step,
        )
    logger.info(
        "Late training start: grouped=%s train_queries=%d val_queries=%d "
        "batches_per_epoch=%d epochs=%d..%d parameters=%s",
        grouped,
        len(train_dataset),
        len(val_dataset),
        len(train_loader),
        start_epoch,
        config.epochs,
        model.parameter_summary(),
    )
    tqdm.write(
        "[setup] "
        f"grouped={grouped} train_queries={len(train_dataset)} "
        f"unique_train_pages={len({record.positive_page_id for record in train_dataset.records})} "
        f"val_queries={len(val_dataset)} batches/epoch={len(train_loader)} "
        f"epochs={start_epoch}..{config.epochs} "
        f"pages_per_batch={config.pages_per_batch} "
        f"queries_per_page={config.queries_per_page} "
        f"maxsim={resolved_backend}/{config.loss.maxsim_normalization} "
        f"avg_queries/batch={len(train_dataset) / max(1, len(train_loader)):.1f} "
        f"effective_queries/update≈{len(train_dataset) / max(1, len(train_loader)) * config.gradient_accumulation_steps:.1f}"
    )
    if config.retrieval_validation_queries > 0 and start_epoch == 1:
        epoch_zero = output_dir / "epoch-000"
        model.save_adapter(
            epoch_zero,
            {
                "epoch": 0,
                "global_step": 0,
                "maxsim_backend": config.loss.maxsim_backend,
                "maxsim_normalization": config.loss.maxsim_normalization,
            },
        )
        baseline_retrieval = _evaluate_corpus_retrieval(
            model,
            val_dataset,
            output_dir,
            epoch_zero,
            config,
            label="epoch 0 baseline",
        )
        best_retrieval = baseline_retrieval
        model.save_adapter(
            output_dir / "best",
            {
                "epoch": 0,
                "global_step": 0,
                "retrieval_metrics": baseline_retrieval,
                "maxsim_backend": config.loss.maxsim_backend,
                "maxsim_normalization": config.loss.maxsim_normalization,
            },
        )
        logger.info("Baseline retrieval metrics: %s", baseline_retrieval)
        tqdm.write(f"[epoch 0 baseline] {baseline_retrieval}")
    for epoch in range(start_epoch, config.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen_queries = 0
        epoch_started = time.perf_counter()
        progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"late train {epoch}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
            disable=not config.progress,
        )
        for batch_index, batch in progress:
            query_tokens, query_global = model.encode_queries(batch.queries)
            if grouped:
                (
                    page_tokens,
                    page_global,
                    hard_tokens,
                    hard_mask,
                    positive_indices,
                ) = _encode_grouped_pages(model, batch, config)
            else:
                page_tokens, page_global = model.encode_pages(
                    batch.positive_images
                )
                hard_tokens = _encode_hard_negatives(
                    model, batch.negative_images
                )
                hard_mask = None
                positive_indices = None
            raw_loss, parts = hybrid_retrieval_loss(
                query_tokens,
                page_tokens,
                query_global=query_global,
                page_global=page_global,
                hard_negative_tokens=hard_tokens,
                hard_negative_mask=hard_mask,
                queue_tokens=queue.get(),
                positive_page_indices=positive_indices,
                config=config.loss,
            )
            (raw_loss / config.gradient_accumulation_steps).backward()
            queue.add(page_tokens)
            query_count = len(batch.queries)
            running += float(raw_loss.detach()) * query_count
            seen_queries += query_count
            if (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if batch_index % max(1, config.log_every) == 0:
                logger.info(
                    "epoch=%d batch=%d/%d queries=%d avg_loss=%.6f "
                    "lr=%.3e allocated_gb=%.2f reserved_gb=%.2f parts=%s",
                    epoch,
                    batch_index,
                    len(train_loader),
                    seen_queries,
                    running / max(1, seen_queries),
                    optimizer.param_groups[0]["lr"],
                    _current_allocated(),
                    _current_reserved(),
                    {
                        key: round(float(value.detach()), 6)
                        for key, value in parts.items()
                        if key != "late_scores"
                    },
                )
            progress.set_postfix(
                loss=f"{float(raw_loss.detach()):.4f}",
                avg=f"{running / max(1, seen_queries):.4f}",
                q=seen_queries,
                pages=(len(batch.page_images) if grouped else len(batch.positive_images)),
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                gpu=f"{_current_reserved():.1f}G",
                refresh=False,
            )
        progress.close()
        train_loss = running / max(1, seen_queries)
        val_loss = evaluate_late_loss(
            model,
            val_loader,
            config.loss,
            maximum_batches=config.validation_batches,
            progress=config.progress,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "queries": float(seen_queries),
            "seconds": epoch_seconds,
        }
        epoch_dir = output_dir / f"epoch-{epoch:03d}"
        model.save_adapter(
            epoch_dir,
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "maxsim_backend": config.loss.maxsim_backend,
                "maxsim_normalization": config.loss.maxsim_normalization,
            },
        )
        retrieval_metrics = None
        if config.retrieval_validation_queries > 0:
            retrieval_metrics = _evaluate_corpus_retrieval(
                model,
                val_dataset,
                output_dir,
                epoch_dir,
                config,
                label=f"epoch {epoch} retrieval",
            )
            record.update(
                {
                    f"retrieval_{key}": float(value)
                    for key, value in retrieval_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
        history.append(record)
        improved = (
            _retrieval_is_better(retrieval_metrics, best_retrieval)
            if retrieval_metrics is not None
            else val_loss < best_val_loss
        )
        if improved:
            best_val_loss = val_loss
            best_retrieval = retrieval_metrics or best_retrieval
            epochs_without_improvement = 0
            model.save_adapter(
                output_dir / "best",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "retrieval_metrics": retrieval_metrics,
                    "maxsim_backend": config.loss.maxsim_backend,
                    "maxsim_normalization": config.loss.maxsim_normalization,
                },
            )
        else:
            epochs_without_improvement += 1
            best_val_loss = min(best_val_loss, val_loss)
        last_extra = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "retrieval_metrics": retrieval_metrics,
            "maxsim_backend": config.loss.maxsim_backend,
            "maxsim_normalization": config.loss.maxsim_normalization,
        }
        model.save_adapter(output_dir / "last", last_extra)
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "queue": queue.state_dict(),
            "best_val_loss": best_val_loss,
            "best_retrieval": best_retrieval,
            "baseline_retrieval": baseline_retrieval,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "training_config": {
                **asdict(config),
                "loss": asdict(config.loss),
            },
        }
        torch.save(state, output_dir / "last" / "training_state.pt")
        _append_jsonl(
            output_dir / "metrics.jsonl",
            {**record, "retrieval": retrieval_metrics},
        )
        _write_json(
            output_dir / "training_summary.json",
            {
                "best_val_loss": best_val_loss,
                "baseline_retrieval": baseline_retrieval,
                "best_retrieval": best_retrieval,
                "global_step": global_step,
                "model": asdict(model_config),
                "training": {
                    **asdict(config),
                    "loss": asdict(config.loss),
                },
                "history": history,
                "parameters": model.parameter_summary(),
                "peak_allocated_gb": _peak_allocated(),
                "peak_reserved_gb": _peak_reserved(),
                "stopped_early": False,
            },
        )
        retrieval_text = (
            ""
            if retrieval_metrics is None
            else (
                f" R@5={retrieval_metrics['recall_at_5']:.4f}"
                f" MRR={retrieval_metrics['mrr']:.4f}"
                f" R@10={retrieval_metrics['recall_at_10']:.4f}"
                f" coarse_miss={retrieval_metrics['coarse_miss_count']}"
            )
        )
        message = (
            f"late epoch {epoch}/{config.epochs}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"queries={seen_queries} time={epoch_seconds / 3600:.2f}h "
            f"peak_reserved={_peak_reserved():.2f}GB improved={improved}"
            f"{retrieval_text}"
        )
        tqdm.write(message)
        logger.info(message)
        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            logger.info(
                "Early stopping after %d epoch(s) without retrieval improvement",
                epochs_without_improvement,
            )
            summary_path = output_dir / "training_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["stopped_early"] = True
            summary["stopped_at_epoch"] = epoch
            _write_json(summary_path, summary)
            break
    return json.loads(
        (output_dir / "training_summary.json").read_text(encoding="utf-8")
    )


@torch.no_grad()
def evaluate_late_loss(
    model: LateInteractionRetriever,
    data_loader: DataLoader,
    loss_config: HybridLossConfig,
    *,
    maximum_batches: int,
    progress: bool = False,
) -> float:
    model.eval()
    losses: list[float] = []
    total = min(len(data_loader), maximum_batches)
    iterator = tqdm(
        enumerate(data_loader),
        total=total,
        desc="validation loss",
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    )
    for index, batch in iterator:
        if index >= maximum_batches:
            break
        query_tokens, query_global = model.encode_queries(batch.queries)
        page_tokens, page_global = model.encode_pages(batch.positive_images)
        value, _ = hybrid_retrieval_loss(
            query_tokens,
            page_tokens,
            query_global=query_global,
            page_global=page_global,
            config=loss_config,
        )
        losses.append(float(value))
        iterator.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)
    iterator.close()
    if not losses:
        raise ValueError("Validation loader produced no late-interaction batches")
    return sum(losses) / len(losses)


def _encode_grouped_pages(
    model: LateInteractionRetriever,
    batch: PageGroupedRetrievalBatch,
    config: LateTrainingConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    token_parts: list[torch.Tensor] = []
    global_parts: list[torch.Tensor] = []
    chunk = max(1, config.page_forward_chunk_size)
    for start in range(0, len(batch.page_images), chunk):
        tokens, globals_ = model.encode_pages(
            batch.page_images[start : start + chunk]
        )
        token_parts.append(tokens)
        global_parts.append(globals_)
    all_tokens = torch.cat(token_parts, dim=0)
    all_globals = torch.cat(global_parts, dim=0)
    device = all_tokens.device
    positive_positions = torch.tensor(
        batch.positive_page_positions,
        device=device,
        dtype=torch.long,
    )
    positive_tokens = all_tokens.index_select(0, positive_positions)
    positive_globals = all_globals.index_select(0, positive_positions)
    query_positive_indices = torch.tensor(
        batch.query_positive_indices,
        device=device,
        dtype=torch.long,
    )
    if not batch.negative_page_positions or not batch.negative_page_positions[0]:
        return (
            positive_tokens,
            positive_globals,
            None,
            None,
            query_positive_indices,
        )
    negative_positions = torch.tensor(
        batch.negative_page_positions,
        device=device,
        dtype=torch.long,
    )
    negative_mask = torch.tensor(
        batch.negative_mask,
        device=device,
        dtype=torch.bool,
    )
    safe_positions = negative_positions.clamp_min(0)
    hard_tokens = all_tokens[safe_positions]
    return (
        positive_tokens,
        positive_globals,
        hard_tokens,
        negative_mask,
        query_positive_indices,
    )


@torch.no_grad()
def _evaluate_corpus_retrieval(
    model: LateInteractionRetriever,
    dataset: RetrievalManifestDataset,
    output_dir: Path,
    adapter_dir: Path,
    config: LateTrainingConfig,
    *,
    label: str,
) -> dict[str, Any]:
    model.eval()
    index_dir = output_dir / "validation_index_current"
    build_multivector_index(
        model,
        dataset.records,
        dataset.data_root,
        index_dir,
        batch_size=max(1, min(8, config.page_forward_chunk_size)),
        pages_per_shard=128,
        manifest_path=dataset.manifest_path,
        adapter_dir=adapter_dir,
        base_model_path=Path(model.config.checkpoint_path),
        progress=config.progress,
    )
    index = MultiVectorIndex(index_dir)
    records = dataset.records[: config.retrieval_validation_queries]
    reciprocal_ranks: list[float] = []
    hits = {1: 0, 5: 0, 10: 0}
    misses = 0
    iterator = tqdm(
        records,
        desc=label,
        unit="query",
        dynamic_ncols=True,
        disable=not config.progress,
    )
    for record in iterator:
        query_tokens, query_global = model.encode_queries([record.query_text])
        candidates = index.coarse_candidates(
            query_global,
            top_k=config.retrieval_coarse_top_k,
        )
        ranking = index.rerank(
            query_tokens,
            candidates,
            backend=config.loss.maxsim_backend,
            normalization=config.loss.maxsim_normalization,
        )
        ranked_ids = [page_id for page_id, _ in ranking]
        try:
            rank = ranked_ids.index(record.positive_page_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
            misses += 1
        for cutoff in hits:
            hits[cutoff] += int(record.positive_page_id in ranked_ids[:cutoff])
        completed = len(reciprocal_ranks)
        if completed % 25 == 0:
            iterator.set_postfix(
                mrr=f"{sum(reciprocal_ranks) / completed:.4f}",
                r5=f"{hits[5] / completed:.4f}",
                miss=misses,
                refresh=False,
            )
    iterator.close()
    count = len(records)
    if count == 0:
        raise ValueError("Retrieval validation contains no queries")
    report = {
        "queries": count,
        "coarse_top_k": config.retrieval_coarse_top_k,
        "mrr": sum(reciprocal_ranks) / count,
        "recall_at_1": hits[1] / count,
        "recall_at_5": hits[5] / count,
        "recall_at_10": hits[10] / count,
        "coarse_miss_count": misses,
    }
    validation_dir = output_dir / "validation_metrics"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(validation_dir / f"{adapter_dir.name}.json", report)
    tqdm.write(f"[{label}] {report}")
    return report


def _retrieval_is_better(
    current: dict[str, Any] | None,
    best: dict[str, Any] | None,
) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    current_r5 = float(current["recall_at_5"])
    best_r5 = float(best["recall_at_5"])
    if current_r5 > best_r5 + 1e-12:
        return True
    if abs(current_r5 - best_r5) <= 1e-12:
        return float(current["mrr"]) > float(best["mrr"])
    return False


def _encode_hard_negatives(
    model: LateInteractionRetriever,
    images: list[list[Any]],
) -> torch.Tensor | None:
    if not images or not all(group for group in images):
        return None
    count = min(len(group) for group in images)
    columns = []
    for negative_index in range(count):
        tokens, _ = model.encode_pages(
            [group[negative_index] for group in images]
        )
        columns.append(tokens)
    return torch.stack(columns, dim=1)


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def _peak_allocated() -> float:
    return (
        torch.cuda.max_memory_allocated() / 2**30
        if torch.cuda.is_available()
        else 0.0
    )


def _peak_reserved() -> float:
    return (
        torch.cuda.max_memory_reserved() / 2**30
        if torch.cuda.is_available()
        else 0.0
    )


def _current_allocated() -> float:
    return (
        torch.cuda.memory_allocated() / 2**30
        if torch.cuda.is_available()
        else 0.0
    )


def _current_reserved() -> float:
    return (
        torch.cuda.memory_reserved() / 2**30
        if torch.cuda.is_available()
        else 0.0
    )


def _training_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"vlm_rag.retrieval.{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(
            output_dir / "training.log",
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
