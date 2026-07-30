from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.dataset import SensitivityManifestDataset
from vlm_rag.sensitivity.metrics import binary_metrics_from_logits
from vlm_rag.sensitivity.model import (
    SensitivityClassifier,
    SensitivityModelConfig,
    save_sensitivity_checkpoint,
)
from vlm_rag.sensitivity.training import SensitivityBatchCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real vision-only forward/backward audit and overfit a tiny "
            "balanced sample."
        )
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--overfit-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Only run a real image forward pass and frozen-vision audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("Smoke batch size must be at least 2")
    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else _first_existing(
            PROJECT_ROOT / "data",
            PROJECT_ROOT.parent / "data",
            required_child="manifests/sensitivity/train.jsonl",
        )
    )
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else data_root / "manifests" / "sensitivity" / "train.jsonl"
    )
    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model is not None
        else _first_existing(
            PROJECT_ROOT / "checkpoint",
            PROJECT_ROOT.parent / "checkpoint",
            required_child="config.json",
        )
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "smoke_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    full_dataset = SensitivityManifestDataset(manifest, data_root, load_images=True)
    positive = [
        index
        for index, record in enumerate(full_dataset.records)
        if record.is_sensitive == 1
    ]
    negative = [
        index
        for index, record in enumerate(full_dataset.records)
        if record.is_sensitive == 0
    ]
    rng = random.Random(args.seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    selected: list[int] = []
    for pos_index, neg_index in zip(
        positive[: args.samples_per_class],
        negative[: args.samples_per_class],
    ):
        selected.extend((pos_index, neg_index))
    dataset = Subset(full_dataset, selected)

    model_config = SensitivityModelConfig(
        checkpoint_path=str(model_path),
        selected_layers=(0, 8, 16, 23, 27),
        dropout=0.10,
        unfreeze_last_n=0,
        dtype="bfloat16",
    )
    model = SensitivityClassifier(model_config, device=args.device)
    summary = model.parameter_summary()
    collator = SensitivityBatchCollator(model.image_processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    device = next(model.parameters()).device
    selected_positive_count = sum(
        full_dataset.records[index].is_sensitive for index in selected
    )
    smoke_pos_weight = (
        (len(selected) - selected_positive_count) / selected_positive_count
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(smoke_pos_weight, device=device)
    )

    # Full image forward/backward: verifies hooks, device placement and gradients.
    model.train()
    audit_batch = next(iter(loader))
    audit_labels = audit_batch["labels"].to(device)
    audit_logits = model(audit_batch["pixel_values"])
    audit_loss = criterion(audit_logits, audit_labels)
    if not bool(torch.isfinite(audit_logits).all()) or not torch.isfinite(audit_loss):
        raise FloatingPointError("Sensitivity forward pass produced non-finite values")
    if args.forward_only:
        peak_allocated_gb = 0.0
        peak_reserved_gb = 0.0
        if torch.device(args.device).type == "cuda":
            peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        result = {
            "status": "passed",
            "mode": "forward_only",
            "samples": len(audit_labels),
            "batch_size": args.batch_size,
            "loss": float(audit_loss.detach()),
            "logits_finite": True,
            "vision_trainable": summary["vision_trainable"],
            "parameter_summary": summary,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "ocr_used_as_model_input": False,
        }
        (output_dir / "forward_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("SENSITIVITY FORWARD SMOKE PASSED")
        return 0
    audit_loss.backward()
    vision_grad_tensors = sum(
        parameter.grad is not None for parameter in model.vision_tower.parameters()
    )
    head_grad_tensors = sum(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if not name.startswith("vision_tower.") and parameter.requires_grad
    )
    if vision_grad_tensors != 0:
        raise AssertionError(
            f"Frozen vision tower unexpectedly produced {vision_grad_tensors} gradients"
        )
    if head_grad_tensors == 0:
        raise AssertionError("Classification head produced no finite gradients")
    model.zero_grad(set_to_none=True)

    # Cache only the selected pooled visual features. This keeps the smoke
    # overfit fast while still using real page pixels and the real SigLIP tower.
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            all_features.append(
                model.extract_layer_features(batch["pixel_values"]).cpu()
            )
            all_labels.append(batch["labels"])
    features = torch.cat(all_features).to(device)
    labels = torch.cat(all_labels).to(device)

    model.train()
    optimizer = torch.optim.AdamW(
        [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not name.startswith("vision_tower.")
        ],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    with torch.no_grad():
        initial_loss = float(criterion(model.classify_layer_features(features), labels))
    final_loss = initial_loss
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model.classify_layer_features(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach())

    model.eval()
    with torch.no_grad():
        final_logits = model.classify_layer_features(features)
    metrics = binary_metrics_from_logits(final_logits.cpu(), labels.cpu(), loss=final_loss)
    if final_loss >= initial_loss * 0.50:
        raise AssertionError(
            f"Tiny overfit loss did not fall enough: {initial_loss:.6f} -> {final_loss:.6f}"
        )
    if metrics.accuracy < 0.99:
        raise AssertionError(f"Tiny overfit accuracy is too low: {metrics.accuracy:.4f}")

    checkpoint_path = output_dir / "smoke_checkpoint.pt"
    save_sensitivity_checkpoint(
        checkpoint_path,
        model,
        epoch=1,
        global_step=args.overfit_steps,
        best_metric=metrics.f1,
        pos_weight=smoke_pos_weight,
        optimizer=optimizer,
        extra={
            "smoke_only": True,
            "sample_page_ids": [
                full_dataset.records[index].page_id for index in selected
            ],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "metrics": metrics.to_dict(),
        },
    )
    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0
    if torch.device(args.device).type == "cuda":
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    result = {
        "status": "passed",
        "samples": len(dataset),
        "batch_size": args.batch_size,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "metrics": metrics.to_dict(),
        "vision_grad_tensors": vision_grad_tensors,
        "head_grad_tensors": head_grad_tensors,
        "parameter_summary": summary,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "checkpoint": str(checkpoint_path),
        "ocr_used_as_model_input": False,
    }
    (output_dir / "smoke_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("SENSITIVITY SMOKE TEST PASSED")
    return 0


def _first_existing(*paths: Path, required_child: str | None = None) -> Path:
    for path in paths:
        required_path = path / required_child if required_child else path
        if required_path.exists():
            return path.resolve()
    raise FileNotFoundError("None of these paths exists: " + ", ".join(map(str, paths)))


if __name__ == "__main__":
    raise SystemExit(main())
