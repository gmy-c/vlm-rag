from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"
    index_dir: str = "indexes"
    log_dir: str = "logs"
    top_k: int = 3
    embedding_dim: int = 768
    temperature: float = 0.07
    epochs: int = 5
    hidden_layer_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
    train_ratio: float = 0.7
    dev_ratio: float = 0.15
    # ── 训练超参数 ──
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.025
    max_grad_norm: float = 1.0
    # ── ColPali 模型参数 ──
    colpali_model: str = "checkpoint"
    lora_rank: int = 32
    lora_alpha: int = 32
    selected_layers_str: str = "0,8,16,23"
    max_query_length: int = 64
    device: str = "cuda"
    # ── 生成器配置 ──
    generator_backend: str = "doubao"
    generator_doubao_model: str = "doubao-seed-1-6-vision-250815"
    generator_doubao_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    generator_max_tokens: int = 512
    generator_temperature: float = 0.1
    generator_timeout: int = 30

    def get_selected_layers(self) -> tuple[int, ...]:
        """将 selected_layers_str 解析为整数元组，例如 "0,8,16,23" → (0, 8, 16, 23)。"""
        if not self.selected_layers_str.strip():
            return ()
        return tuple(
            int(item.strip()) for item in self.selected_layers_str.split(",") if item.strip()
        )

    def get_api_key(self) -> str:
        """从环境变量 DOUBAO_API_KEY 获取 API Key。

        Raises:
            ValueError: 环境变量未设置时抛出，携带设置指引。
        """
        import os
        key = os.environ.get("DOUBAO_API_KEY", "")
        if not key:
            raise ValueError(
                "DOUBAO_API_KEY environment variable not set. "
                "Set it with: export DOUBAO_API_KEY='your-key'"
            )
        return key


def load_config(path: Path) -> ProjectConfig:
    if not path.exists():
        return ProjectConfig()

    values: dict[str, str] = {}
    current_prefix = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Detect nested block header (e.g. "generator:")
        if line.endswith(":") and " " not in line:
            current_prefix = line[:-1] + "_"  # "generator" → "generator_"
            continue
        # Exit nested block when indentation drops
        if not raw_line.startswith((" ", "\t")) and current_prefix:
            current_prefix = ""
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = current_prefix + key.strip()
        value = value.strip().strip("\"'")
        if value:
            values[key] = value

    defaults = ProjectConfig()
    return ProjectConfig(
        data_dir=values.get("data_dir", defaults.data_dir),
        output_dir=values.get("output_dir", defaults.output_dir),
        model_dir=values.get("model_dir", defaults.model_dir),
        index_dir=values.get("index_dir", defaults.index_dir),
        log_dir=values.get("log_dir", defaults.log_dir),
        top_k=_parse_int(values.get("top_k"), defaults.top_k),
        embedding_dim=_parse_int(values.get("embedding_dim"), defaults.embedding_dim),
        temperature=_parse_float(values.get("temperature"), defaults.temperature),
        epochs=_parse_int(values.get("epochs"), defaults.epochs),
        hidden_layer_weights=_parse_weights(
            values.get("hidden_layer_weights"),
            defaults.hidden_layer_weights,
        ),
        train_ratio=_parse_float(values.get("train_ratio"), defaults.train_ratio),
        dev_ratio=_parse_float(values.get("dev_ratio"), defaults.dev_ratio),
        # ── 新增字段 ──
        batch_size=_parse_int(values.get("batch_size"), defaults.batch_size),
        gradient_accumulation_steps=_parse_int(
            values.get("gradient_accumulation_steps"), defaults.gradient_accumulation_steps
        ),
        learning_rate=_parse_float(values.get("learning_rate"), defaults.learning_rate),
        warmup_ratio=_parse_float(values.get("warmup_ratio"), defaults.warmup_ratio),
        max_grad_norm=_parse_float(values.get("max_grad_norm"), defaults.max_grad_norm),
        colpali_model=values.get("colpali_model", defaults.colpali_model),
        lora_rank=_parse_int(values.get("lora_rank"), defaults.lora_rank),
        lora_alpha=_parse_int(values.get("lora_alpha"), defaults.lora_alpha),
        selected_layers_str=values.get("selected_layers_str", defaults.selected_layers_str),
        max_query_length=_parse_int(values.get("max_query_length"), defaults.max_query_length),
        device=values.get("device", defaults.device),
        # ── 生成器字段 ──
        generator_backend=values.get("generator_backend", defaults.generator_backend),
        generator_doubao_model=values.get("generator_doubao_model", defaults.generator_doubao_model),
        generator_doubao_base_url=values.get("generator_doubao_base_url", defaults.generator_doubao_base_url),
        generator_max_tokens=_parse_int(values.get("generator_max_tokens"), defaults.generator_max_tokens),
        generator_temperature=_parse_float(values.get("generator_temperature"), defaults.generator_temperature),
        generator_timeout=_parse_int(values.get("generator_timeout"), defaults.generator_timeout),
    )


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_weights(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        return default
    total = sum(weights)
    return tuple(weight / total for weight in weights)
