from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True, slots=True)
class LateInteractionModelConfig:
    checkpoint_path: str
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    max_query_length: int = 96
    gradient_checkpointing: bool = True
    image_only_tokens: bool = True
    global_dim: int = 768
    dtype: str = "bfloat16"


class TokenAttentionPool(nn.Module):
    def __init__(self, token_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(token_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        valid = tokens[:, :, 0].ne(0)
        logits = self.gate(tokens.float()).squeeze(-1)
        logits = logits.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(tokens.float() * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.projection(pooled), p=2, dim=-1)


class LateInteractionRetriever(nn.Module):
    """Native ColPali token embeddings plus global coarse-retrieval heads."""

    def __init__(
        self,
        config: LateInteractionModelConfig,
        *,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.device_name = device
        from colpali_engine.models import ColPali, ColPaliProcessor

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.colpali = ColPali.from_pretrained(
            config.checkpoint_path,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.processor = ColPaliProcessor.from_pretrained(
            config.checkpoint_path,
            local_files_only=Path(config.checkpoint_path).is_dir(),
        )
        self.colpali.requires_grad_(False)
        self.colpali.custom_text_proj.requires_grad_(True)
        self.colpali.custom_text_proj.float()

        from peft import LoraConfig, TaskType, get_peft_model

        language_model = self.colpali.model.model.language_model
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        language_model = get_peft_model(language_model, lora_config)
        if config.gradient_checkpointing:
            language_model.gradient_checkpointing_enable()
            language_model.config.use_cache = False
            if hasattr(language_model, "enable_input_require_grads"):
                language_model.enable_input_require_grads()
        self.colpali.model.model.language_model = language_model
        head_device = next(language_model.parameters()).device
        token_dim = int(self.colpali.dim)
        self.query_pool = TokenAttentionPool(token_dim, config.global_dim).to(
            head_device
        )
        self.page_pool = TokenAttentionPool(token_dim, config.global_dim).to(
            head_device
        )
        self._keep_vision_frozen()

    def train(self, mode: bool = True) -> "LateInteractionRetriever":
        super().train(mode)
        self._keep_vision_frozen()
        return self

    def encode_queries(
        self,
        queries: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor.process_texts(queries)
        if inputs["input_ids"].shape[1] > self.config.max_query_length:
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in inputs:
                    inputs[key] = inputs[key][:, : self.config.max_query_length]
        tokens = self._forward_inputs(inputs, image_only=False)
        return tokens, self.query_pool(tokens)

    def encode_pages(
        self,
        images: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor.process_images(images)
        tokens = self._forward_inputs(
            inputs,
            image_only=self.config.image_only_tokens,
        )
        return tokens, self.page_pool(tokens)

    def _forward_inputs(
        self,
        inputs: Any,
        *,
        image_only: bool,
    ) -> torch.Tensor:
        device = next(self.colpali.parameters()).device
        values = {
            key: value.to(device=device, non_blocking=True)
            for key, value in dict(inputs).items()
            if isinstance(value, torch.Tensor)
        }
        if "pixel_values" in values:
            values["pixel_values"] = values["pixel_values"].to(
                dtype=next(self.colpali.parameters()).dtype
            )
        outputs = self.colpali.model.model(
            **values,
            use_cache=False,
            return_dict=True,
        )
        tokens = self.colpali.custom_text_proj(
            outputs.last_hidden_state.float()
        )
        tokens = F.normalize(tokens, p=2, dim=-1)
        if device.type == "cuda":
            tokens = tokens.to(torch.bfloat16)
        mask = values["attention_mask"].bool()
        if image_only:
            mask = mask & values["input_ids"].eq(
                self.colpali.config.image_token_index
            )
            # All pages use the fixed 448x448 processor, hence 1024 image tokens.
            token_rows = [row[row_mask] for row, row_mask in zip(tokens, mask)]
            lengths = {len(row) for row in token_rows}
            if len(lengths) != 1:
                raise RuntimeError(
                    f"Variable image-token lengths are unsupported: {lengths}"
                )
            return torch.stack(token_rows)
        return tokens * mask.unsqueeze(-1)

    def optimizer_parameter_groups(
        self,
        *,
        lora_lr: float,
        projection_lr: float,
        pooling_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        language = [
            parameter
            for parameter in self.colpali.model.model.language_model.parameters()
            if parameter.requires_grad
        ]
        projection = list(self.colpali.custom_text_proj.parameters())
        pooling = list(self.query_pool.parameters()) + list(
            self.page_pool.parameters()
        )
        return [
            {
                "params": language,
                "lr": lora_lr,
                "weight_decay": weight_decay,
                "name": "language_lora",
            },
            {
                "params": projection,
                "lr": projection_lr,
                "weight_decay": weight_decay,
                "name": "token_projection",
            },
            {
                "params": pooling,
                "lr": pooling_lr,
                "weight_decay": weight_decay,
                "name": "global_pooling",
            },
        ]

    def parameter_summary(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "vision_trainable": sum(
                parameter.numel()
                for parameter in self.colpali.model.model.vision_tower.parameters()
                if parameter.requires_grad
            ),
        }

    def save_adapter(self, output_dir: Path, extra: dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.colpali.model.model.language_model.save_pretrained(
            output_dir / "language_lora"
        )
        torch.save(
            {
                "custom_text_proj": self.colpali.custom_text_proj.state_dict(),
                "query_pool": self.query_pool.state_dict(),
                "page_pool": self.page_pool.state_dict(),
                "config": asdict(self.config),
                "extra": extra,
            },
            output_dir / "retrieval_heads.pt",
        )
        (output_dir / "retrieval_config.json").write_text(
            json.dumps(
                {"model": asdict(self.config), "extra": extra},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_adapter(self, checkpoint_dir: Path) -> dict[str, Any]:
        self.load_language_adapter(checkpoint_dir / "language_lora")
        payload = torch.load(
            checkpoint_dir / "retrieval_heads.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.colpali.custom_text_proj.load_state_dict(
            payload["custom_text_proj"]
        )
        self.query_pool.load_state_dict(payload["query_pool"])
        self.page_pool.load_state_dict(payload["page_pool"])
        return dict(payload.get("extra", {}))

    def load_language_adapter(self, adapter_dir: Path) -> None:
        """Load compatible rank/alpha LoRA weights without nesting PEFT."""
        from peft.utils.save_and_load import (
            load_peft_weights,
            set_peft_model_state_dict,
        )

        language_model = self.colpali.model.model.language_model
        adapter_state = load_peft_weights(
            str(adapter_dir),
            device=str(next(language_model.parameters()).device),
        )
        set_peft_model_state_dict(
            language_model,
            adapter_state,
            adapter_name="default",
        )

    def _keep_vision_frozen(self) -> None:
        vision = self.colpali.model.model.vision_tower
        vision.requires_grad_(False)
        vision.eval()
