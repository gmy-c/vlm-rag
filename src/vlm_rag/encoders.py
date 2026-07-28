"""ColPali-based dual-tower encoder with hidden-layer weighted pooling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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
    """

    model_name: str = "vidore/colpali-v1.2"
    device: str = "cuda"
    proj_dim: int = 768
    selected_layers: tuple[int, ...] = (0, 8, 16, 23)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    max_query_length: int = 64


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
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.config.model_name)

        # ── 2. 获取内部模块引用 ──
        # PaliGemma 内部结构:
        #   model.model.vision_tower           → SigLIP ViT-SO400M
        #   model.model.multi_modal_projector  → 视觉→语言投影
        #   model.model.language_model         → Gemma-2B
        self.vision_tower = self.model.model.vision_tower
        self.language_model = self.model.model.language_model

        vision_hidden = self.vision_tower.config.hidden_size   # 1152
        text_hidden = self.language_model.config.hidden_size    # 2048

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
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.language_model = get_peft_model(self.language_model, lora_cfg)

        # ── 4. 隐藏层加权池化权重 ⭐ 核心创新点 ──
        num_layers = len(self.config.selected_layers)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # ── 5. 投影头 ──
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden, self.config.proj_dim),
            nn.LayerNorm(self.config.proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, self.config.proj_dim),
            nn.LayerNorm(self.config.proj_dim),
        )

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
            2) SigLIP ViT forward with output_hidden_states=True.
            3) Select specified hidden layers and apply learned softmax weights.
            4) Mean-pool over the patch dimension.
            5) Project to proj_dim and L2-normalize.

        The vision tower is always kept frozen (torch.no_grad).
        """
        # Step 1: preprocess
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=self.vision_tower.device,
            dtype=torch.bfloat16,
        )

        # Step 2: ViT forward (frozen)
        with torch.no_grad():
            outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # Step 3: hidden-layer weighted pooling  ⭐
        # outputs.hidden_states: tuple of layer outputs, each [B, num_patches, 1152]
        selected = [
            outputs.hidden_states[i] for i in self.config.selected_layers
        ]
        # softmax over layer weights → positive and sum to 1
        weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
        weighted = sum(
            w * h for w, h in zip(weights, selected)
        )  # [B, num_patches, 1152]

        # Step 4: mean pooling over patches
        pooled = weighted.mean(dim=1)  # [B, 1152]

        # Step 5: project & normalise
        projected = self.vision_proj(pooled)  # [B, proj_dim]
        return F.normalize(projected, p=2, dim=-1)

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
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_query_length,
        )
        input_ids = inputs["input_ids"].to(self.language_model.device)
        attention_mask = inputs["attention_mask"].to(self.language_model.device)

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
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # [B, 2048]

        # Step 5: project & normalise
        projected = self.text_proj(pooled)  # [B, proj_dim]
        return F.normalize(projected, p=2, dim=-1)

    # ═══════════════════════════════════════════════════════════
    # 训练管理
    # ═══════════════════════════════════════════════════════════

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
        base_model: str = "vidore/colpali-v1.2",
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

        # Ensure the model name is valid (may differ across machines)
        if not saved_config.model_name:
            saved_config.model_name = base_model

        encoder = cls(saved_config)
        encoder.vision_proj.load_state_dict(checkpoint["vision_proj"])
        encoder.text_proj.load_state_dict(checkpoint["text_proj"])
        encoder.layer_weights.data = checkpoint["layer_weights"]

        if encoder.config.use_lora:
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
