from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader

from vlm_rag.encoders import ColPaliDualEncoder, ColPaliDualEncoderConfig

from .dataset import (
    DocumentUniqueBatchSampler,
    RetrievalManifestDataset,
    retrieval_collate,
)
from .losses import (
    HybridLossConfig,
    MultiVectorMemoryQueue,
    hybrid_retrieval_loss,
    symmetric_global_info_nce,
)
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
) -> dict[str, Any]:
    _seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = LateInteractionRetriever(model_config, device=device)
    if resume_from is not None and initialize_lora_from is not None:
        raise ValueError(
            "resume_from and initialize_lora_from are mutually exclusive"
        )
    if initialize_lora_from is not None:
        model.load_language_adapter(initialize_lora_from)
    if resume_from is not None:
        model.load_adapter(resume_from)
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            lora_lr=config.lora_learning_rate,
            projection_lr=config.projection_learning_rate,
            pooling_lr=config.pooling_learning_rate,
            weight_decay=config.weight_decay,
        ),
        fused=device.startswith("cuda"),
    )
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
    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        micro_steps = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            query_tokens, query_global = model.encode_queries(batch.queries)
            page_tokens, page_global = model.encode_pages(batch.positive_images)
            hard_tokens = _encode_hard_negatives(model, batch.negative_images)
            raw_loss, parts = hybrid_retrieval_loss(
                query_tokens,
                page_tokens,
                query_global=query_global,
                page_global=page_global,
                hard_negative_tokens=hard_tokens,
                queue_tokens=queue.get(),
                config=config.loss,
            )
            (raw_loss / config.gradient_accumulation_steps).backward()
            queue.add(page_tokens)
            running += float(raw_loss.detach())
            micro_steps += 1
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
        train_loss = running / max(1, micro_steps)
        val_loss = evaluate_late_loss(
            model,
            val_loader,
            config.loss,
            maximum_batches=config.validation_batches,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        model.save_adapter(
            output_dir / "last",
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_adapter(
                output_dir / "best",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
            )
        _write_json(
            output_dir / "training_summary.json",
            {
                "best_val_loss": best_val_loss,
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
            },
        )
        print(
            f"late epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )
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
) -> float:
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(data_loader):
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
    if not losses:
        raise ValueError("Validation loader produced no late-interaction batches")
    return sum(losses) / len(losses)


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
