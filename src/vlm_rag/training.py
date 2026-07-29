"""InfoNCE training loop for ColPali dual-tower retriever."""
from __future__ import annotations

import csv
import json
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
    model_name: str = "vidore/colpali-v1.3-merged",
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    epochs: int = 5,
    temperature: float = 0.07,
    warmup_ratio: float = 0.025,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
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
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 初始化编码器 ──
    encoder_config = ColPaliDualEncoderConfig(
        model_name=model_name,
        device=device,
    )
    encoder = ColPaliDualEncoder(encoder_config)
    n_trainable = encoder.trainable_param_count()
    print(f"Trainable parameters: {n_trainable:,}")

    # ── 2. 正例快速查找 ──
    page_lookup: dict[str, Page] = {p.page_id: p for p in train_pages}

    # ── 3. 优化器（8-bit AdamW）──
    import bitsandbytes as bnb

    optimizer = bnb.PagedAdamW8bit(
        encoder.trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # ── 4. 学习率调度器（linear warmup → linear decay）──
    steps_per_epoch = max(
        1, len(train_queries) // (batch_size * gradient_accumulation_steps)
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

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════
    for epoch in range(1, epochs + 1):
        encoder.train()
        random.shuffle(train_queries)
        optimizer.zero_grad()
        epoch_loss = 0.0

        for i in range(0, len(train_queries), batch_size):
            # ── 取 batch ──
            batch_qs = train_queries[i : i + batch_size]
            batch_ps = [
                page_lookup[q.positive_page_ids[0]] for q in batch_qs
            ]

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
            if batch_idx_in_epoch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    encoder.trainable_parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # ── 验证（每 50 步）──
            if global_step > 0 and global_step % 50 == 0:
                val_m = _validate(encoder, val_pages, val_queries)
                writer.writerow(
                    dict(
                        epoch=epoch,
                        step=global_step,
                        loss=round(
                            epoch_loss / max(1, global_step), 6
                        ),
                        lr=round(scheduler.get_last_lr()[0], 8),
                        val_mrr_at_10=val_m["mrr@10"],
                        val_recall_at_3=val_m["recall@3"],
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

        avg_loss = epoch_loss / max(1, len(train_queries))
        print(
            f"Epoch {epoch}/{epochs} done. "
            f"avg_loss={avg_loss:.4f}"
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
) -> dict[str, float]:
    """Evaluate MRR@10 and Recall@3 on the validation set.

    The encoder is temporarily switched to eval mode and restored to
    train mode before returning.
    """
    encoder.eval()
    retriever = DualTowerRetriever(encoder)
    retriever.index(val_pages)

    ranked: dict[str, list] = {}
    for q in val_queries:
        ranked[q.query_id] = retriever.search(q.text, top_k=10)

    encoder.train()  # restore training mode
    return {
        "mrr@10": round(mrr_at_k(val_queries, ranked, 10), 4),
        "recall@3": round(recall_at_k(val_queries, ranked, 3), 4),
    }
