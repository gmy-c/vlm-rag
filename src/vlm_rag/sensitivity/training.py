from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .metrics import BinaryMetrics, binary_metrics_from_logits
from .model import SensitivityClassifier, save_sensitivity_checkpoint


@dataclass(frozen=True, slots=True)
class SensitivityTrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    head_learning_rate: float | None = None
    vision_learning_rate: float = 1e-5
    layer_weight_learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    num_workers: int = 0
    prefetch_factor: int = 2
    target_recall: float = 0.90
    seed: int = 42

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("head_learning_rate", self.resolved_head_learning_rate),
            ("vision_learning_rate", self.vision_learning_rate),
            ("layer_weight_learning_rate", self.layer_weight_learning_rate),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if not 0 < self.target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")

    @property
    def resolved_head_learning_rate(self) -> float:
        return (
            self.learning_rate
            if self.head_learning_rate is None
            else self.head_learning_rate
        )


class SensitivityBatchCollator:
    """Decode no text and pass only page pixels plus binary labels."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        processed = self.image_processor(
            images=[item["image"] for item in items],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed["pixel_values"],
            "labels": torch.tensor(
                [item["label"] for item in items],
                dtype=torch.float32,
            ),
            "page_ids": [item["page_id"] for item in items],
        }


def compute_pos_weight(dataset: Sequence[dict[str, Any]]) -> float:
    labels = _dataset_labels(dataset)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training data must contain both labels; "
            f"positive={positives}, negative={negatives}"
        )
    return negatives / positives


def train_sensitivity_classifier(
    model: SensitivityClassifier,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    config: SensitivityTrainingConfig,
    *,
    resume_from: Path | None = None,
    initialize_from: Path | None = None,
) -> dict[str, Any]:
    config.validate()
    _set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    pos_weight_value = compute_pos_weight(train_dataset)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            vision_learning_rate=config.vision_learning_rate,
            head_learning_rate=config.resolved_head_learning_rate,
            layer_weight_learning_rate=config.layer_weight_learning_rate,
            weight_decay=config.weight_decay,
        ),
        fused=device.type == "cuda",
    )
    trainable_parameters = model.trainable_parameters()

    collator = SensitivityBatchCollator(model.image_processor)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader_options: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_options["prefetch_factor"] = config.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        **loader_options,
    )
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / config.gradient_accumulation_steps),
    )
    total_steps = max(1, config.epochs * optimizer_steps_per_epoch)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _warmup_cosine_lambda(total_steps, warmup_steps),
    )

    start_epoch = 1
    global_step = 0
    best_f1 = -1.0
    best_selection = (-1, -1.0, -1.0, -1.0)
    best_epoch = 0
    epochs_without_improvement = 0
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    if initialize_from is not None:
        payload = torch.load(
            initialize_from,
            map_location="cpu",
            weights_only=True,
        )
        model.load_trainable_state_dict(
            payload["model_state"],
            allow_missing_newly_unfrozen_vision=True,
        )
    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        model.load_trainable_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload.get("global_step", 0))
        best_f1 = float(payload.get("best_metric", -1.0))
        best_epoch = int(payload.get("extra", {}).get("best_epoch", 0))
        validation = payload.get("extra", {}).get("validation", {})
        if validation:
            best_selection = _selection_key(
                validation,
                config.target_recall,
            )

    log_path = output_dir / "training_log.jsonl"
    log_mode = "a" if resume_from is not None else "w"
    with log_path.open(log_mode, encoding="utf-8", newline="\n") as log:
        for epoch in range(start_epoch, config.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            sample_count = 0
            remainder = len(train_loader) % config.gradient_accumulation_steps

            for batch_index, batch in enumerate(train_loader, start=1):
                labels = batch["labels"].to(device=device, non_blocking=True)
                logits = model(batch["pixel_values"])
                raw_loss = criterion(logits, labels)
                in_partial_final_group = (
                    remainder > 0
                    and batch_index > len(train_loader) - remainder
                )
                accumulation_denominator = (
                    remainder
                    if in_partial_final_group
                    else config.gradient_accumulation_steps
                )
                (raw_loss / accumulation_denominator).backward()
                running_loss += float(raw_loss.detach()) * labels.numel()
                sample_count += labels.numel()

                is_last_batch = batch_index == len(train_loader)
                if (
                    batch_index % config.gradient_accumulation_steps == 0
                    or is_last_batch
                ):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        config.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch {epoch}: {grad_norm}"
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            train_loss = running_loss / max(1, sample_count)
            val_metrics = evaluate_sensitivity_classifier(
                model,
                val_loader,
                criterion=criterion,
            )
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "pos_weight": pos_weight_value,
                "val": val_metrics.to_dict(),
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(
                f"Epoch {epoch}/{config.epochs}: train_loss={train_loss:.6f}, "
                f"val_loss={val_metrics.loss:.6f}, val_f1={val_metrics.f1:.4f}, "
                f"val_recall={val_metrics.recall:.4f}"
            )

            selection = _selection_key(
                val_metrics.to_dict(),
                config.target_recall,
            )
            improved = selection > best_selection
            if improved:
                best_selection = selection
                best_f1 = val_metrics.f1
                best_epoch = epoch
                epochs_without_improvement = 0
                save_sensitivity_checkpoint(
                    output_dir / "best.pt",
                    model,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_f1,
                    pos_weight=pos_weight_value,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra={
                        "best_epoch": best_epoch,
                        "training_config": asdict(config),
                        "validation": val_metrics.to_dict(),
                        "selection_policy": {
                            "target_recall": config.target_recall,
                            "priority": (
                                "meet_target_recall_then_precision_then_pr_auc"
                            ),
                        },
                    },
                )
            else:
                epochs_without_improvement += 1

            save_sensitivity_checkpoint(
                output_dir / "last.pt",
                model,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_f1,
                pos_weight=pos_weight_value,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
                    "best_epoch": best_epoch,
                    "training_config": asdict(config),
                    "validation": val_metrics.to_dict(),
                    "selection_policy": {
                        "target_recall": config.target_recall,
                        "priority": (
                            "meet_target_recall_then_precision_then_pr_auc"
                        ),
                    },
                },
            )
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping: no validation operating-point "
                    f"improvement for {epochs_without_improvement} epochs."
                )
                break

    summary = {
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "selection_policy": {
            "target_recall": config.target_recall,
            "priority": "meet_target_recall_then_precision_then_pr_auc",
        },
        "global_step": global_step,
        "pos_weight": pos_weight_value,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "parameter_summary": model.parameter_summary(),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "peak_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def evaluate_sensitivity_classifier(
    model: SensitivityClassifier,
    data_loader: DataLoader,
    *,
    criterion: torch.nn.Module | None = None,
) -> BinaryMetrics:
    model.eval()
    device = next(model.parameters()).device
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    loss_sum = 0.0
    sample_count = 0
    for batch in data_loader:
        labels = batch["labels"].to(device=device, non_blocking=True)
        logits = model(batch["pixel_values"])
        if criterion is not None:
            loss_sum += float(criterion(logits, labels)) * labels.numel()
        sample_count += labels.numel()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_logits:
        raise ValueError("Cannot evaluate an empty dataset")
    return binary_metrics_from_logits(
        torch.cat(all_logits),
        torch.cat(all_labels),
        loss=loss_sum / max(1, sample_count),
    )


def stratified_subset(
    dataset: Dataset,
    maximum: int | None,
    *,
    seed: int,
) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 2:
        raise ValueError("A limited classification subset must contain at least 2 items")
    labels = _dataset_labels(dataset)
    by_label = {
        0: [index for index, label in enumerate(labels) if label == 0],
        1: [index for index, label in enumerate(labels) if label == 1],
    }
    rng = random.Random(seed)
    for indices in by_label.values():
        rng.shuffle(indices)
    positive_target = max(1, round(maximum * len(by_label[1]) / len(labels)))
    positive_target = min(positive_target, len(by_label[1]), maximum - 1)
    negative_target = min(maximum - positive_target, len(by_label[0]))
    remaining = maximum - positive_target - negative_target
    if remaining:
        extra_positive = min(remaining, len(by_label[1]) - positive_target)
        positive_target += extra_positive
        negative_target += remaining - extra_positive
    selected = (
        by_label[1][:positive_target]
        + by_label[0][:negative_target]
    )
    rng.shuffle(selected)
    return Subset(dataset, selected)


def _dataset_labels(dataset: Sequence[dict[str, Any]]) -> list[int]:
    if isinstance(dataset, Subset):
        base_labels = _dataset_labels(dataset.dataset)
        return [base_labels[int(index)] for index in dataset.indices]
    records = getattr(dataset, "records", None)
    if records is not None:
        return [int(record.is_sensitive) for record in records]
    return [int(dataset[index]["label"]) for index in range(len(dataset))]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def _selection_key(
    metrics: dict[str, Any],
    target_recall: float,
) -> tuple[int, float, float, float]:
    recall = float(metrics["recall"])
    precision = float(metrics["precision"])
    pr_auc_value = metrics.get("pr_auc")
    pr_auc = -1.0 if pr_auc_value is None else float(pr_auc_value)
    f1 = float(metrics["f1"])
    if recall + 1e-12 >= target_recall:
        return (1, precision, pr_auc, f1)
    return (0, recall, pr_auc, f1)


def _warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
):
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, max(0.0, progress)))
        )

    return scale
