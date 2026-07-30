from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.dataset import SensitivityManifestDataset
from vlm_rag.sensitivity.model import SensitivityClassifier, SensitivityModelConfig
from vlm_rag.sensitivity.training import (
    SensitivityTrainingConfig,
    stratified_subset,
    train_sensitivity_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the vision-only page sensitivity classifier."
    )
    parser.add_argument("--config", default="configs/sensitivity.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--vision-learning-rate", type=float, default=None)
    parser.add_argument("--unfreeze-last-n", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize the trainable head from a prior stage without "
            "restoring optimizer state; used for frozen -> unfreeze transition."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _resolve_existing(args.config, "config")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML object in {config_path}")

    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else _resolve_existing(
            str(raw["data_root"]),
            "data root",
            required_child="docvqa_images",
        )
    )
    manifest_dir = (
        args.manifest_dir.expanduser().resolve()
        if args.manifest_dir is not None
        else (
            data_root / "manifests" / "sensitivity"
            if args.data_root is not None
            else _resolve_existing(
                str(raw["manifest_dir"]),
                "manifest directory",
                required_child="train.jsonl",
            )
        )
    )
    model_path = _resolve_existing(
        args.model or str(raw["model"]),
        "model checkpoint",
        required_child="config.json",
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / str(raw["output_dir"])).resolve()
    )
    device = args.device or str(raw.get("device", "cuda"))

    train_dataset = SensitivityManifestDataset(
        manifest_dir / "train.jsonl",
        data_root,
        load_images=True,
    )
    val_dataset = SensitivityManifestDataset(
        manifest_dir / "val.jsonl",
        data_root,
        load_images=True,
    )
    seed = int(raw.get("seed", 42))
    train_dataset = stratified_subset(
        train_dataset,
        args.max_train_samples,
        seed=seed,
    )
    val_dataset = stratified_subset(
        val_dataset,
        args.max_val_samples,
        seed=seed + 1,
    )

    unfreeze_last_n = (
        args.unfreeze_last_n
        if args.unfreeze_last_n is not None
        else int(raw.get("unfreeze_last_n", 0))
    )
    model_config = SensitivityModelConfig(
        checkpoint_path=str(model_path),
        selected_layers=tuple(int(index) for index in raw["selected_layers"]),
        dropout=float(raw.get("dropout", 0.20)),
        unfreeze_last_n=unfreeze_last_n,
        dtype=str(raw.get("dtype", "bfloat16")),
        spatial_pool_size=int(raw.get("spatial_pool_size", 8)),
        head_dim=int(raw.get("head_dim", 768)),
        head_layers=int(raw.get("head_layers", 2)),
        head_attention_heads=int(raw.get("head_attention_heads", 12)),
        gradient_checkpointing=bool(
            raw.get("gradient_checkpointing", False)
        ),
    )
    training_config = SensitivityTrainingConfig(
        batch_size=_override(args.batch_size, raw, "batch_size"),
        gradient_accumulation_steps=_override(
            args.gradient_accumulation_steps,
            raw,
            "gradient_accumulation_steps",
        ),
        epochs=_override(args.epochs, raw, "epochs"),
        learning_rate=_override(args.learning_rate, raw, "learning_rate"),
        head_learning_rate=(
            args.head_learning_rate
            if args.head_learning_rate is not None
            else float(
                raw.get("head_learning_rate", raw["learning_rate"])
            )
        ),
        vision_learning_rate=(
            args.vision_learning_rate
            if args.vision_learning_rate is not None
            else float(raw.get("vision_learning_rate", 1e-5))
        ),
        layer_weight_learning_rate=float(
            raw.get("layer_weight_learning_rate", 5e-4)
        ),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.05)),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 3)),
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(raw.get("num_workers", 0))
        ),
        prefetch_factor=int(raw.get("prefetch_factor", 2)),
        target_recall=float(raw.get("target_recall", 0.90)),
        seed=seed,
    )

    print("=" * 68)
    print("Vision-only sensitivity classification")
    print(f"data_root:      {data_root}")
    print(f"manifest_dir:   {manifest_dir}")
    print(f"base_model:     {model_path}")
    print(f"output_dir:     {output_dir}")
    print(f"train/val:      {len(train_dataset)}/{len(val_dataset)}")
    print(f"batch/accum:    {training_config.batch_size}/"
          f"{training_config.gradient_accumulation_steps}")
    print(f"device/dtype:   {device}/{model_config.dtype}")
    print(f"vision_unfrozen:{model_config.unfreeze_last_n}")
    print("OCR model input: disabled")
    print("=" * 68)

    model = SensitivityClassifier(model_config, device=device)
    print(f"parameters: {model.parameter_summary()}")
    summary = train_sensitivity_classifier(
        model,
        train_dataset,
        val_dataset,
        output_dir,
        training_config,
        resume_from=args.resume,
        initialize_from=args.init_checkpoint,
    )
    print(f"Training summary: {summary}")
    return 0


def _resolve_existing(
    value: str,
    description: str,
    *,
    required_child: str | None = None,
) -> Path:
    path = Path(value).expanduser()
    candidates = (
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT.parent / path,
    )
    for candidate in candidates:
        required_path = candidate / required_child if required_child else candidate
        if required_path.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve {description} {value!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _override(value: Any, raw: dict[str, Any], key: str) -> Any:
    return value if value is not None else raw[key]


if __name__ == "__main__":
    raise SystemExit(main())
