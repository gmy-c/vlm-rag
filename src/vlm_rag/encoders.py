"""ColPali-based dual-tower encoder with memory-efficient layer pooling."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data import Page


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════


@dataclass
class ColPaliDualEncoderConfig:
    """Configuration for the ColPali dual-tower encoder.

    Attributes:
        model_name: HuggingFace model ID for the ColPali checkpoint.
        device: Torch device string (e.g. "cuda", "cuda:0").
        proj_dim: Dimension of the shared embedding space.
        selected_layers: Which ViT hidden layers participate in weighted pooling.
        use_lora: Whether to attach LoRA adapters to the language model.
        lora_rank: LoRA rank (r).
        lora_alpha: LoRA alpha scaling factor.
        max_query_length: Maximum number of tokens for query text.
        gradient_checkpointing: Recompute text activations during backward to
            reduce peak training memory.
    """

    model_name: str = "checkpoint"
    device: str = "cuda"
    proj_dim: int = 768
    selected_layers: tuple[int, ...] = (0, 8, 16, 23)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_rslora: bool = False
    max_query_length: int = 64
    gradient_checkpointing: bool = True


# ═══════════════════════════════════════════════════════════════
# ColPali 双塔编码器
# ═══════════════════════════════════════════════════════════════


class ColPaliDualEncoder(nn.Module):
    """ColPali-based dual-tower encoder for document image retrieval.

    Architecture:
        Text Tower:   Query → PaliGemma LLM (LoRA) → masked mean pool
                      → projection → L2 norm → [proj_dim]

        Vision Tower: Page Image → SigLIP ViT (frozen)
                      → hidden-layer weighted pooling
                      → mean pool over patches → projection
                      → L2 norm → [proj_dim]

    Both outputs are normalized vectors in the same proj_dim space,
    directly comparable via inner product (cosine similarity).

    Trainable components (when use_lora=True):
        - LoRA adapters on language_model (~18M params)
        - layer_weights (4 scalars) for hidden-layer pooling
        - vision_proj + text_proj projection heads (~2M params)
    """

    def __init__(self, config: ColPaliDualEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or ColPaliDualEncoderConfig()

        # ── 1. 加载 ColPali 模型 ──
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.model = ColPali.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.config.device,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.config.model_name)

        # ── 2. 获取内部模块引用 ──
        # PaliGemma 内部结构:
        #   model.model.vision_tower           → SigLIP ViT-SO400M
        #   model.model.multi_modal_projector  → 视觉→语言投影
        #   model.model.language_model         → Gemma-2B
        self.vision_tower = self.model.model.model.vision_tower
        self.language_model = self.model.model.model.language_model

        vision_hidden = self.vision_tower.config.hidden_size   # 1152
        text_hidden = self.language_model.config.hidden_size    # 2048

        # The vision backbone is a feature extractor only.  ``no_grad`` in
        # ``_encode_images`` avoids activation graphs, while requires_grad=False
        # also prevents accidental optimiser inclusion and makes the intent
        # explicit in parameter audits.
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        # ── 3. LoRA 适配器 ──
        if self.config.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=self.config.lora_dropout,
                use_rslora=self.config.use_rslora,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.language_model = get_peft_model(self.language_model, lora_cfg)
            if self.config.gradient_checkpointing:
                self.language_model.gradient_checkpointing_enable()
                self.language_model.config.use_cache = False
                # PEFT needs grad-enabled embedding outputs when the frozen
                # base model is used with gradient checkpointing.
                if hasattr(self.language_model, "enable_input_require_grads"):
                    self.language_model.enable_input_require_grads()

        # ── 4. 隐藏层加权池化权重 ⭐ 核心创新点 ──
        head_device = next(self.language_model.parameters()).device
        num_layers = len(self.config.selected_layers)
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers, device=head_device, dtype=torch.float32)
            / num_layers
        )

        # ── 5. 投影头 ──
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden, self.config.proj_dim),
            nn.LayerNorm(self.config.proj_dim),
        ).to(device=head_device, dtype=torch.float32)
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, self.config.proj_dim),
            nn.LayerNorm(self.config.proj_dim),
        ).to(device=head_device, dtype=torch.float32)

    # ═══════════════════════════════════════════════════════════
    # 视觉塔
    # ═══════════════════════════════════════════════════════════

    def encode_page(self, page: Page) -> torch.Tensor:
        """Encode a single page image → normalized vector [proj_dim]."""
        image = Image.open(page.image_path).convert("RGB")
        return self._encode_images([image]).squeeze(0)

    def encode_page_batch(self, pages: list[Page]) -> torch.Tensor:
        """Encode a batch of pages → [B, proj_dim]."""
        images = [Image.open(p.image_path).convert("RGB") for p in pages]
        return self._encode_images(images)

    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Core vision encoding with hidden-layer weighted pooling.

        Steps:
            1) Preprocess images via ColPali processor → pixel_values.
            2) Capture only the requested SigLIP hidden states via hooks.
            3) Apply learned softmax weights to the selected states.
            4) Mean-pool over the patch dimension.
            5) Project to proj_dim and L2-normalize.

        The old implementation requested ``output_hidden_states=True``, which
        retained every ViT layer.  For a validation index containing hundreds
        of pages this produced a very large peak even though gradients were
        disabled.  Hooks retain only ``selected_layers``.
        """
        # Step 1: preprocess
        inputs = self.processor.process_images(images)
        pixel_values = inputs["pixel_values"].to(
            device=next(self.vision_tower.parameters()).device,
            dtype=torch.bfloat16,
        )

        # Step 2: ViT forward (frozen), retaining only requested states.
        with torch.no_grad():
            selected = self._capture_selected_vision_states(pixel_values)

        # Step 3: hidden-layer weighted pooling  ⭐
        # softmax over layer weights → positive and sum to 1
        weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
        weighted = sum(
            w * h for w, h in zip(weights, selected)
        )  # [B, num_patches, 1152]

        # Step 4: mean pooling over patches
        pooled = weighted.mean(dim=1).float()  # [B, 1152]

        # Step 5: project & normalise
        projected = self.vision_proj(pooled)  # [B, proj_dim]
        return F.normalize(projected, p=2, dim=-1)

    def _capture_selected_vision_states(
        self,
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Run SigLIP while retaining only configured hidden-state indices.

        Hugging Face vision models define hidden state 0 as the patch
        embeddings before encoder layer 0, and hidden state ``i`` (i > 0) as
        the output of encoder layer ``i - 1``.  The hooks below preserve those
        semantics without materialising the complete hidden-state tuple.
        """
        encoder_layers = self._vision_encoder_layers()
        captures: dict[int, torch.Tensor] = {}
        handles: list[Any] = []

        def _as_tensor(output: Any) -> torch.Tensor:
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

        def _pre_hook(state_index: int) -> Callable:
            def capture(_module: nn.Module, args: tuple[Any, ...]) -> None:
                captures[state_index] = _as_tensor(args[0]).detach()

            return capture

        def _post_hook(state_index: int) -> Callable:
            def capture(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
            ) -> None:
                captures[state_index] = _as_tensor(output).detach()

            return capture

        try:
            for state_index in self.config.selected_layers:
                if state_index < 0 or state_index > len(encoder_layers):
                    raise ValueError(
                        f"Vision hidden-state index {state_index} is out of "
                        f"range [0, {len(encoder_layers)}]"
                    )
                if state_index == 0:
                    handles.append(
                        encoder_layers[0].register_forward_pre_hook(
                            _pre_hook(state_index)
                        )
                    )
                else:
                    handles.append(
                        encoder_layers[state_index - 1].register_forward_hook(
                            _post_hook(state_index)
                        )
                    )

            self.vision_tower(
                pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = [
            index
            for index in self.config.selected_layers
            if index not in captures
        ]
        if missing:
            raise RuntimeError(
                f"Failed to capture vision hidden states: {missing}"
            )
        return [captures[index] for index in self.config.selected_layers]

    def _vision_encoder_layers(self) -> nn.ModuleList:
        """Return the SigLIP encoder layers across supported HF layouts."""
        candidates = (
            ("vision_model", "encoder", "layers"),
            ("encoder", "layers"),
        )
        for path in candidates:
            current: Any = self.vision_tower
            for attr in path:
                if not hasattr(current, attr):
                    break
                current = getattr(current, attr)
            else:
                return current
        raise AttributeError(
            "Unable to locate SigLIP encoder layers on the vision tower"
        )

    # ═══════════════════════════════════════════════════════════
    # 文本塔
    # ═══════════════════════════════════════════════════════════

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a single query string → normalized vector [proj_dim]."""
        return self._encode_texts([text]).squeeze(0)

    def encode_query_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of queries → [B, proj_dim]."""
        return self._encode_texts(texts)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Core text encoding through PaliGemma LLM (LoRA active).

        Steps:
            1) Tokenize with ColPali processor (right-padding).
            2) Get input embeddings.
            3) LLM forward — LoRA adapters are active here, so NO torch.no_grad.
            4) Masked mean pooling (exclude padding tokens).
            5) Project to proj_dim and L2-normalize.
        """
        # Step 1: tokenize
        inputs = self.processor.process_texts(texts)
        text_device = next(self.language_model.parameters()).device
        input_ids = inputs["input_ids"][
            :, : self.config.max_query_length
        ].to(text_device)
        attention_mask = inputs["attention_mask"][
            :, : self.config.max_query_length
        ].to(text_device)

        # Step 2: word embeddings
        text_embeds = self.model.model.get_input_embeddings()(input_ids)
        # [B, L, 2048]

        # Step 3: LLM forward (LoRA active — gradients flow through here)
        outputs = self.language_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # [B, L, 2048]

        # Step 4: masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        pooled = (last_hidden.float() * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)
        # [B, 2048]

        # Step 5: project & normalise
        projected = self.text_proj(pooled)  # [B, proj_dim]
        return F.normalize(projected, p=2, dim=-1)

    # ═══════════════════════════════════════════════════════════
    # 训练管理
    # ═══════════════════════════════════════════════════════════

    def train(self, mode: bool = True) -> "ColPaliDualEncoder":
        """Set training mode while keeping the frozen vision tower in eval."""
        super().train(mode)
        self.vision_tower.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters that require gradients.

        Includes:
            - LoRA weights inside language_model
            - layer_weights (hidden-layer pooling)
            - vision_proj and text_proj weights
        """
        params: list[nn.Parameter] = []
        for p in self.language_model.parameters():
            if p.requires_grad:
                params.append(p)
        params.append(self.layer_weights)
        params.extend(self.vision_proj.parameters())
        params.extend(self.text_proj.parameters())
        return params

    def trainable_param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, save_dir: Path) -> None:
        """Persist LoRA weights, projection heads, layer weights and config."""
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            self.language_model.save_pretrained(save_dir / "lora")

        torch.save(
            {
                "vision_proj": self.vision_proj.state_dict(),
                "text_proj": self.text_proj.state_dict(),
                "layer_weights": self.layer_weights.data,
                "config": self.config,
            },
            save_dir / "head_weights.pt",
        )

    @classmethod
    def load(
        cls,
        save_dir: Path,
        base_model: str | None = None,
        device: str | None = None,
    ) -> "ColPaliDualEncoder":
        """Load a saved checkpoint from disk.

        Args:
            save_dir: Directory containing head_weights.pt and lora/.
            base_model: HF model ID to use if the saved config doesn't specify one.
        """
        checkpoint = torch.load(
            save_dir / "head_weights.pt", map_location="cpu", weights_only=False
        )
        saved_config: ColPaliDualEncoderConfig = checkpoint["config"]
        saved_config = replace(
            saved_config,
            model_name=base_model or saved_config.model_name or "checkpoint",
            device=device or saved_config.device,
        )

        # Load the bare language model first.  Creating LoRA in ``cls`` and
        # wrapping it again with ``PeftModel.from_pretrained`` would produce a
        # nested adapter and incorrect parameter/device state.
        init_config = replace(saved_config, use_lora=False)
        encoder = cls(init_config)
        encoder.config = saved_config
        encoder.vision_proj.load_state_dict(checkpoint["vision_proj"])
        encoder.text_proj.load_state_dict(checkpoint["text_proj"])
        with torch.no_grad():
            encoder.layer_weights.copy_(
                checkpoint["layer_weights"].to(encoder.layer_weights.device)
            )

        if saved_config.use_lora:
            from peft import PeftModel

            encoder.language_model = PeftModel.from_pretrained(
                encoder.language_model,
                save_dir / "lora",
            )

        return encoder


# ═══════════════════════════════════════════════════════════════
# 全局工具函数
# ═══════════════════════════════════════════════════════════════


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine similarity between two L2-normalised vectors.

    Since both inputs are already normalised, the inner product equals
    the cosine of the angle between them.
    """
    return (left @ right).item()


def info_nce_loss(
    query_vectors: torch.Tensor,     # [B, D]  L2-normalised
    positive_vectors: torch.Tensor,  # [B, D]  L2-normalised
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss with in-batch negatives.

    For a batch of B query-page pairs, the i-th query and the i-th page
    form a positive pair.  All other pages in the batch serve as negatives
    (in-batch negatives), so no explicit negative sampling is required.

    The loss is computed as cross-entropy over the B × B similarity matrix,
    where the correct label for row i is column i.
    """
    # Similarity matrix — inner product equals cosine because vectors are
    # already L2-normalised
    sim = (query_vectors @ positive_vectors.T) / temperature  # [B, B]

    # Labels: diagonal entries are the positives
    labels = torch.arange(sim.size(0), device=sim.device)

    return F.cross_entropy(sim, labels)
