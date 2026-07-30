"""InfoNCE training loop for ColPali dual-tower retriever."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import Page, Query
from .encoders import (
    ColPaliDualEncoder,
    ColPaliDualEncoderConfig,
    info_nce_loss,
)
from .metrics import mrr_at_k, recall_at_k
from .retriever import DualTowerRetriever


def train_colpali_retriever(
    train_pages: list[Page],
    train_queries: list[Query],
    val_pages: list[Page],
    val_queries: list[Query],
    model_dir: Path,
    *,
    model_name: str = "checkpoint",
    proj_dim: int = 768,
    selected_layers: tuple[int, ...] = (0, 8, 16, 23),
    lora_rank: int = 32,
    lora_alpha: int = 32,
    max_query_length: int = 64,
    gradient_checkpointing: bool = True,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    epochs: int = 5,
    temperature: float = 0.07,
    warmup_ratio: float = 0.025,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    index_batch_size: int = 2,
) -> dict[str, object]:
    """Train the ColPali dual-tower retriever with InfoNCE loss.

    Trainable components:
        - LoRA adapters on language_model  (~18M params)
        - layer_weights for hidden-layer pooling (4 scalars)
        - vision_proj + text_proj projection heads (~2M params)

    Frozen:
        - PaliGemma-3B backbone (SigLIP ViT + Gemma-2B base weights)

    Loss:
        InfoNCE with in-batch negatives — each query's positive page is the
        corresponding page in the batch; all other pages serve as negatives.

    Args:
        train_pages: All training-set pages.
        train_queries: All training-set queries (each references ≥1 positive page).
        val_pages: Validation-set pages.
        val_queries: Validation-set queries.
        model_dir: Directory for checkpoints and logs.
        batch_size: Per-step batch size (before gradient accumulation).
        gradient_accumulation_steps: Number of steps between optimizer updates.
        learning_rate: Peak learning rate (after warmup).
        epochs: Number of passes over the full training set.
        temperature: InfoNCE temperature τ (lower = sharper distribution).
        warmup_ratio: Fraction of total steps used for linear warmup.
        max_grad_norm: Gradient clipping threshold.
        device: Torch device string.

    Returns:
        Dict with best epoch, step, mrr@10 and recall@3.
    """
    if batch_size < 2:
        raise ValueError("InfoNCE training requires batch_size >= 2")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 初始化编码器 ──
    encoder_config = ColPaliDualEncoderConfig(
        model_name=model_name,
        device=device,
        proj_dim=proj_dim,
        selected_layers=selected_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_query_length=max_query_length,
        gradient_checkpointing=gradient_checkpointing,
    )
    encoder = ColPaliDualEncoder(encoder_config)
    n_trainable = encoder.trainable_param_count()
    print(f"Trainable parameters: {n_trainable:,}")

    # ── 2. 正例快速查找 ──
    page_lookup: dict[str, Page] = {p.page_id: p for p in train_pages}

    # ── 3. 优化器（8-bit AdamW）──
    import bitsandbytes as bnb

    try:
        _AdamW = bnb.optim.AdamW8bit
    except AttributeError:
        _AdamW = bnb.optim.AdamW     # fallback for bitsandbytes >= 0.45

    optimizer = _AdamW(
        encoder.trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ── 4. 学习率调度器（linear warmup → linear decay）──
    queries_by_page: dict[str, list[Query]] = {}
    for query in train_queries:
        queries_by_page.setdefault(query.positive_page_ids[0], []).append(query)
    samples_per_epoch = len(queries_by_page)
    micro_batches_per_epoch = math.ceil(samples_per_epoch / batch_size)
    steps_per_epoch = max(
        1,
        math.ceil(micro_batches_per_epoch / gradient_accumulation_steps),
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── 5. 训练日志 ──
    log_path = model_dir / "training_log.csv"
    log_file = log_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "step",
            "loss",
            "lr",
            "val_mrr@10",
            "val_recall@3",
        ],
    )
    writer.writeheader()

    best_score = -float("inf")
    best_state: dict[str, object] = {}
    global_step = 0
    last_validation_step = -1

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════
    for epoch in range(1, epochs + 1):
        encoder.train()
        # Multiple DocVQA questions often point to the same page.  Putting two
        # such pairs in one InfoNCE batch incorrectly treats the second copy of
        # that page as a negative.  Sample one question per positive page per
        # epoch; different questions are seen across epochs.
        epoch_queries = [
            random.choice(page_queries)
            for page_queries in queries_by_page.values()
        ]
        random.shuffle(epoch_queries)
        if len(epoch_queries) % batch_size == 1:
            epoch_queries.pop()
        optimizer.zero_grad()
        epoch_loss = 0.0
        processed_micro_batches = 0

        for i in range(0, len(epoch_queries), batch_size):
            # ── 取 batch ──
            batch_qs = epoch_queries[i : i + batch_size]
            # A one-item InfoNCE batch has zero loss and no useful gradient.
            if len(batch_qs) < 2:
                continue
            batch_ps = [
                page_lookup[q.positive_page_ids[0]] for q in batch_qs
            ]
            processed_micro_batches += 1

            # ── 双塔编码 ──
            q_vecs = encoder.encode_query_batch(
                [q.text for q in batch_qs]
            )  # [B, proj_dim]
            p_vecs = encoder.encode_page_batch(batch_ps)  # [B, proj_dim]

            # ── InfoNCE 损失 ──
            loss = info_nce_loss(q_vecs, p_vecs, temperature)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            # ── 梯度累积 ──
            batch_idx_in_epoch = i // batch_size + 1
            is_last_batch = i + batch_size >= len(epoch_queries)
            should_step = (
                batch_idx_in_epoch % gradient_accumulation_steps == 0
                or is_last_batch
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    encoder.trainable_parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── 验证（每 50 步）──
            if should_step and global_step > 0 and global_step % 50 == 0:
                val_m = _validate(
                    encoder,
                    val_pages,
                    val_queries,
                    index_batch_size=index_batch_size,
                )
                last_validation_step = global_step
                writer.writerow(
                    dict(
                        epoch=epoch,
                        step=global_step,
                        loss=round(
                            epoch_loss / max(1, global_step), 6
                        ),
                        lr=round(scheduler.get_last_lr()[0], 8),
                        **{
                            "val_mrr@10": val_m["mrr@10"],
                            "val_recall@3": val_m["recall@3"],
                        },
                    )
                )
                log_file.flush()

                score = val_m["mrr@10"] + val_m["recall@3"]
                if score > best_score:
                    best_score = score
                    best_state = {
                        "epoch": epoch,
                        "step": global_step,
                        **val_m,
                    }
                    encoder.save(model_dir)
                    print(
                        f"  [Best @ step {global_step}] "
                        f"MRR@10={val_m['mrr@10']:.4f}, "
                        f"Recall@3={val_m['recall@3']:.4f}"
                    )

        avg_loss = epoch_loss / max(1, processed_micro_batches)
        print(
            f"Epoch {epoch}/{epochs} done. "
            f"avg_loss={avg_loss:.4f}"
        )

        # Always validate at epoch end.  This guarantees a checkpoint for
        # small datasets and smoke runs with fewer than 50 optimiser steps.
        if val_pages and val_queries and last_validation_step != global_step:
            val_m = _validate(
                encoder,
                val_pages,
                val_queries,
                index_batch_size=index_batch_size,
            )
            last_validation_step = global_step
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": round(avg_loss, 6),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "val_mrr@10": val_m["mrr@10"],
                    "val_recall@3": val_m["recall@3"],
                }
            )
            log_file.flush()

            score = val_m["mrr@10"] + val_m["recall@3"]
            if score > best_score:
                best_score = score
                best_state = {
                    "epoch": epoch,
                    "step": global_step,
                    **val_m,
                }
                encoder.save(model_dir)
                print(
                    f"  [Best @ epoch {epoch}] "
                    f"MRR@10={val_m['mrr@10']:.4f}, "
                    f"Recall@3={val_m['recall@3']:.4f}"
                )

    log_file.close()

    # 保存最终配置
    (model_dir / "retriever_config.json").write_text(
        json.dumps(best_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_state


# ═══════════════════════════════════════════════════════════
# 内部验证
# ═══════════════════════════════════════════════════════════


def _validate(
    encoder: ColPaliDualEncoder,
    val_pages: list[Page],
    val_queries: list[Query],
    *,
    index_batch_size: int = 2,
) -> dict[str, float]:
    """Evaluate MRR@10 and Recall@3 on the validation set.

    The encoder is temporarily switched to eval mode and restored to
    train mode before returning.
    """
    encoder.eval()
    retriever = DualTowerRetriever(encoder)
    retriever.index(val_pages, batch_size=index_batch_size)

    ranked: dict[str, list] = {}
    for q in val_queries:
        ranked[q.query_id] = retriever.search(q.text, top_k=10)

    encoder.train()  # restore training mode
    return {
        "mrr@10": round(mrr_at_k(val_queries, ranked, 10), 4),
        "recall@3": round(recall_at_k(val_queries, ranked, 3), 4),
    }
