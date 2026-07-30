from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


VISION_WEIGHT_PREFIXES = (
    "model.vision_tower.vision_model.",
    "model.model.vision_tower.vision_model.",
    "vision_tower.vision_model.",
)


@dataclass(frozen=True, slots=True)
class SensitivityModelConfig:
    checkpoint_path: str
    selected_layers: tuple[int, ...] = (0, 8, 16, 23, 27)
    dropout: float = 0.20
    unfreeze_last_n: int = 0
    dtype: str = "bfloat16"
    spatial_pool_size: int = 8
    head_dim: int = 768
    head_layers: int = 2
    head_attention_heads: int = 12
    gradient_checkpointing: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_layers"] = list(self.selected_layers)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensitivityModelConfig":
        return cls(
            checkpoint_path=str(value["checkpoint_path"]),
            selected_layers=tuple(int(index) for index in value["selected_layers"]),
            dropout=float(value.get("dropout", 0.20)),
            unfreeze_last_n=int(value.get("unfreeze_last_n", 0)),
            dtype=str(value.get("dtype", "bfloat16")),
            spatial_pool_size=int(value.get("spatial_pool_size", 8)),
            head_dim=int(value.get("head_dim", 768)),
            head_layers=int(value.get("head_layers", 2)),
            head_attention_heads=int(value.get("head_attention_heads", 12)),
            gradient_checkpointing=bool(
                value.get("gradient_checkpointing", False)
            ),
        )


class SensitivityClassifier(nn.Module):
    """Vision-only page classifier backed by the ColPali SigLIP tower.

    Only requested hidden states are retained. Unlike the legacy single-vector
    head, each layer is reduced to a grid of regional tokens. A small
    Transformer aggregates those tokens only after spatial evidence has been
    preserved.
    """

    def __init__(
        self,
        config: SensitivityModelConfig,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.device_name = str(device)
        self.vision_dtype = _resolve_dtype(config.dtype, device)
        checkpoint = Path(config.checkpoint_path).expanduser().resolve()
        self.vision_tower, self.image_processor = load_siglip_vision_tower(
            checkpoint,
            device=device,
            dtype=self.vision_dtype,
        )

        self.num_vision_layers = int(self.vision_tower.config.num_hidden_layers)
        _validate_selected_layers(config.selected_layers, self.num_vision_layers)
        if not 0 <= config.unfreeze_last_n <= self.num_vision_layers:
            raise ValueError(
                "unfreeze_last_n must be between 0 and "
                f"{self.num_vision_layers}, got {config.unfreeze_last_n}"
            )
        if (
            config.unfreeze_last_n > 0
            and self.num_vision_layers not in config.selected_layers
        ):
            raise ValueError(
                "When unfreeze_last_n > 0, selected_layers must include the final "
                f"hidden state {self.num_vision_layers} so gradients reach the "
                "unfrozen blocks."
            )

        self._configure_vision_trainability(config.unfreeze_last_n)
        if config.unfreeze_last_n and config.gradient_checkpointing:
            self.vision_tower.gradient_checkpointing_enable()
        hidden_size = int(self.vision_tower.config.hidden_size)
        if config.spatial_pool_size < 1:
            raise ValueError("spatial_pool_size must be positive")
        if config.head_dim % config.head_attention_heads:
            raise ValueError("head_dim must be divisible by head_attention_heads")
        head_device = next(self.vision_tower.parameters()).device
        self.layer_weights = nn.Parameter(
            torch.zeros(len(config.selected_layers), dtype=torch.float32, device=head_device)
        )
        region_count = config.spatial_pool_size**2
        self.regional_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.head_dim),
            nn.GELU(),
        ).to(device=head_device, dtype=torch.float32)
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, config.head_dim, device=head_device)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, region_count + 1, config.head_dim, device=head_device)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.head_dim,
            nhead=config.head_attention_heads,
            dim_feedforward=config.head_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.regional_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.head_layers,
            enable_nested_tensor=False,
        ).to(device=head_device, dtype=torch.float32)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.head_dim),
            nn.Linear(config.head_dim, config.head_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_dim // 2, 1),
        ).to(device=head_device, dtype=torch.float32)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def train(self, mode: bool = True) -> "SensitivityClassifier":
        super().train(mode)
        if self.config.unfreeze_last_n == 0:
            self.vision_tower.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        layer_features = self.extract_layer_features(pixel_values)
        return self.classify_layer_features(layer_features)

    def extract_layer_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return regional states as ``[batch, layers, regions, hidden]``."""

        pixel_values = pixel_values.to(
            device=next(self.vision_tower.parameters()).device,
            dtype=self.vision_dtype,
            non_blocking=True,
        )
        if self.config.unfreeze_last_n == 0:
            with torch.no_grad():
                selected = self._capture_selected_states(pixel_values)
        else:
            selected = self._capture_selected_states(pixel_values)

        pooled = [
            self._pool_spatial_regions(hidden).float() for hidden in selected
        ]
        return torch.stack(pooled, dim=1)

    def classify_layer_features(self, layer_features: torch.Tensor) -> torch.Tensor:
        layer_features = layer_features.to(
            device=self.layer_weights.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        weights = torch.softmax(self.layer_weights, dim=0)
        fused = torch.sum(
            layer_features * weights.view(1, -1, 1, 1),
            dim=1,
        )
        regional = self.regional_projection(fused)
        cls = self.cls_token.expand(regional.shape[0], -1, -1)
        tokens = torch.cat((cls, regional), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.regional_encoder(tokens)
        return self.classifier(encoded[:, 0]).squeeze(-1)

    def _pool_spatial_regions(self, hidden: torch.Tensor) -> torch.Tensor:
        patch_count = hidden.shape[1]
        grid_size = int(patch_count**0.5)
        if grid_size * grid_size != patch_count:
            raise RuntimeError(
                f"Expected a square SigLIP patch grid, got {patch_count} tokens"
            )
        feature_map = hidden.transpose(1, 2).reshape(
            hidden.shape[0],
            hidden.shape[2],
            grid_size,
            grid_size,
        )
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            feature_map,
            (self.config.spatial_pool_size, self.config.spatial_pool_size),
        )
        return pooled.flatten(2).transpose(1, 2)

    def _capture_selected_states(
        self,
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:
        captures: dict[int, torch.Tensor] = {}
        hooks: list[Any] = []

        def capture(index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                captures[index] = value

            return hook

        for hidden_index in self.config.selected_layers:
            module: nn.Module
            if hidden_index == 0:
                module = self.vision_tower.embeddings
            else:
                module = self.vision_tower.encoder.layers[hidden_index - 1]
            hooks.append(module.register_forward_hook(capture(hidden_index)))

        try:
            self.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()

        missing = [
            index for index in self.config.selected_layers if index not in captures
        ]
        if missing:
            raise RuntimeError(f"Failed to capture SigLIP hidden states: {missing}")
        return [captures[index] for index in self.config.selected_layers]

    def _configure_vision_trainability(self, unfreeze_last_n: int) -> None:
        self.vision_tower.requires_grad_(False)
        if unfreeze_last_n > 0:
            for layer in self.vision_tower.encoder.layers[-unfreeze_last_n:]:
                layer.requires_grad_(True)
        else:
            self.vision_tower.eval()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def optimizer_parameter_groups(
        self,
        *,
        vision_learning_rate: float,
        head_learning_rate: float,
        layer_weight_learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        vision_ids = {
            id(parameter)
            for parameter in self.vision_tower.parameters()
            if parameter.requires_grad
        }
        layer_weight_id = id(self.layer_weights)
        head = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
            and id(parameter) not in vision_ids
            and id(parameter) != layer_weight_id
        ]
        groups: list[dict[str, Any]] = [
            {
                "params": head,
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
                "name": "regional_head",
            },
            {
                "params": [self.layer_weights],
                "lr": layer_weight_learning_rate,
                "weight_decay": 0.0,
                "name": "layer_weights",
            },
        ]
        vision = [
            parameter
            for parameter in self.vision_tower.parameters()
            if parameter.requires_grad
        ]
        if vision:
            groups.append(
                {
                    "params": vision,
                    "lr": vision_learning_rate,
                    "weight_decay": weight_decay,
                    "name": "vision_last_layers",
                }
            )
        return groups

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        vision_total = sum(parameter.numel() for parameter in self.vision_tower.parameters())
        vision_trainable = sum(
            parameter.numel()
            for parameter in self.vision_tower.parameters()
            if parameter.requires_grad
        )
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "vision_total": vision_total,
            "vision_trainable": vision_trainable,
            "head_trainable": trainable - vision_trainable,
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }

    def load_trainable_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        allow_missing_newly_unfrozen_vision: bool = False,
    ) -> None:
        parameters = dict(self.named_parameters())
        expected = {name for name, value in parameters.items() if value.requires_grad}
        provided = set(state_dict)
        missing = expected - provided
        unexpected = provided - expected
        allowed_missing = (
            {
                name
                for name in missing
                if name.startswith("vision_tower.")
            }
            if allow_missing_newly_unfrozen_vision
            else set()
        )
        if unexpected or missing != allowed_missing:
            raise ValueError(
                f"Trainable checkpoint mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        with torch.no_grad():
            for name, value in state_dict.items():
                target = parameters[name]
                target.copy_(value.to(device=target.device, dtype=target.dtype))


def load_siglip_vision_tower(
    checkpoint: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any]:
    """Load only SigLIP tensors from a local PaliGemma/ColPali checkpoint."""

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory not found: {checkpoint}")
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    full_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    vision_config = full_config.vision_config
    with init_empty_weights():
        vision_tower = SiglipVisionModel(vision_config)

    expected_keys = set(vision_tower.state_dict())
    weight_map = _checkpoint_weight_map(checkpoint)
    prefix = _detect_vision_prefix(weight_map, expected_keys)
    source_by_target = {
        source_key[len(prefix) :]: (source_key, shard)
        for source_key, shard in weight_map.items()
        if source_key.startswith(prefix)
    }
    missing = expected_keys - set(source_by_target)
    unexpected = set(source_by_target) - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            "Vision checkpoint key mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for target_key, (source_key, shard) in source_by_target.items():
        by_shard.setdefault(shard, []).append((target_key, source_key))
    for shard, keys in by_shard.items():
        shard_path = checkpoint / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for target_key, source_key in keys:
                state_dict[target_key] = handle.get_tensor(source_key)

    incompatible = vision_tower.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Could not load vision tower strictly: {incompatible}")
    vision_tower.to(device=device, dtype=dtype)
    vision_tower.eval()
    del state_dict

    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    return vision_tower, processor


def save_sensitivity_checkpoint(
    path: Path,
    model: SensitivityClassifier,
    *,
    epoch: int,
    global_step: int,
    best_metric: float,
    pos_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "task": "page_sensitivity_classification",
        "model_config": model.config.to_dict(),
        "model_state": model.trainable_state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "pos_weight": float(pos_weight),
        "extra": dict(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_sensitivity_checkpoint(
    path: Path,
    *,
    device: str | torch.device = "cuda",
    checkpoint_path_override: str | None = None,
    load_optimizer: bool = False,
) -> tuple[SensitivityClassifier, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_config = SensitivityModelConfig.from_dict(payload["model_config"])
    if checkpoint_path_override is not None:
        model_config = replace(
            model_config,
            checkpoint_path=checkpoint_path_override,
        )
    model = SensitivityClassifier(model_config, device=device)
    model.load_trainable_state_dict(payload["model_state"])
    if not load_optimizer:
        payload.pop("optimizer_state", None)
        payload.pop("scheduler_state", None)
    return model, payload


def _resolve_dtype(value: str, device: str | torch.device) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        if torch.device(device).type == "cpu":
            return torch.float32
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported model dtype: {value!r}")


def _validate_selected_layers(
    selected_layers: Iterable[int],
    num_hidden_layers: int,
) -> None:
    values = tuple(selected_layers)
    if not values:
        raise ValueError("selected_layers must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"selected_layers contains duplicates: {values}")
    invalid = [index for index in values if not 0 <= index <= num_hidden_layers]
    if invalid:
        raise ValueError(
            f"selected_layers must be between 0 and {num_hidden_layers}; "
            f"invalid={invalid}"
        )


def _checkpoint_weight_map(checkpoint: Path) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid weight_map in {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    single_path = checkpoint / "model.safetensors"
    if single_path.is_file():
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as handle:
            return {key: single_path.name for key in handle.keys()}
    raise FileNotFoundError(
        f"No model.safetensors or model.safetensors.index.json in {checkpoint}"
    )


def _detect_vision_prefix(
    weight_map: Mapping[str, str],
    expected_keys: set[str],
) -> str:
    for prefix in VISION_WEIGHT_PREFIXES:
        matched = {
            key[len(prefix) :] for key in weight_map if key.startswith(prefix)
        }
        if matched == expected_keys:
            return prefix
    counts = {
        prefix: sum(key.startswith(prefix) for key in weight_map)
        for prefix in VISION_WEIGHT_PREFIXES
    }
    raise RuntimeError(
        "Could not identify a complete SigLIP vision tower in checkpoint; "
        f"candidate_counts={counts}, expected={len(expected_keys)}"
    )
