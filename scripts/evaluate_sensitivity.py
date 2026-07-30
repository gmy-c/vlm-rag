from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vlm_rag.sensitivity.inference import (
    InferenceError,
    InferenceItem,
    PredictionResult,
    items_from_manifest,
    run_batched_inference,
)
from vlm_rag.sensitivity.metrics import (
    binary_metrics_from_probabilities,
    calibrate_thresholds,
)
from vlm_rag.sensitivity.model import load_sensitivity_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate the page sensitivity classifier."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Local PaliGemma/ColPali directory used as the frozen vision base.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else data_root / "manifests" / "sensitivity" / "val.jsonl"
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    base_model = _resolve_model(args.model)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "sensitivity_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, checkpoint_metadata = load_sensitivity_checkpoint(
        checkpoint,
        device=args.device,
        checkpoint_path_override=str(base_model),
    )
    baseline_allocated = _cuda_gb(torch.cuda.memory_allocated)

    items = list(items_from_manifest(manifest, data_root))
    items = _stratified_items(items, args.max_samples, seed=args.seed)
    predictions: list[PredictionResult] = []
    errors: list[InferenceError] = []
    for event in run_batched_inference(
        model,
        items,
        batch_size=args.batch_size,
        threshold=0.5,
    ):
        if isinstance(event, PredictionResult):
            predictions.append(event)
        else:
            errors.append(event)

    _write_predictions(output_dir, predictions)
    _write_errors(output_dir / "errors.jsonl", errors)
    if not predictions:
        raise RuntimeError("Evaluation produced no valid predictions")

    probabilities = torch.tensor(
        [prediction.probability for prediction in predictions],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [int(prediction.true_label) for prediction in predictions],
        dtype=torch.int64,
    )
    evaluation_loss = float(
        F.binary_cross_entropy(
            probabilities.clamp(1e-7, 1 - 1e-7),
            labels.to(torch.float64),
        )
    )
    default_metrics = binary_metrics_from_probabilities(
        probabilities,
        labels,
        loss=evaluation_loss,
        threshold=0.5,
    )
    calibration = calibrate_thresholds(
        probabilities,
        labels,
        target_recall=args.target_recall,
        default_threshold=0.5,
        loss=evaluation_loss,
    )
    smoke_only = bool(
        checkpoint_metadata.get("extra", {}).get("smoke_only", False)
    )
    calibration_payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "samples_requested": len(items),
        "samples_evaluated": len(predictions),
        "errors": len(errors),
        "smoke_evaluation": smoke_only or args.max_samples is not None,
        "warning": (
            "Smoke/tiny-sample metrics validate the pipeline only and are not "
            "a formal model-quality result."
            if smoke_only or args.max_samples is not None
            else None
        ),
        **calibration,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "manifest": str(manifest),
        "batch_size": args.batch_size,
        "samples_requested": len(items),
        "samples_evaluated": len(predictions),
        "errors": len(errors),
        "metrics_at_0.5": default_metrics.to_dict(),
        "calibration_file": str(output_dir / "calibration.json"),
        "baseline_gpu_allocated_gb": baseline_allocated,
        "peak_gpu_allocated_gb": _cuda_gb(torch.cuda.max_memory_allocated),
        "peak_gpu_reserved_gb": _cuda_gb(torch.cuda.max_memory_reserved),
        "smoke_evaluation": calibration_payload["smoke_evaluation"],
        "warning": calibration_payload["warning"],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Thresholds:")
    print(
        "  default=0.500000, "
        f"best_f1={calibration['best_f1']['threshold']:.6f}, "
        f"target_recall={calibration['target_recall']['threshold']:.6f}"
    )
    if errors:
        print(
            f"Evaluation completed with {len(errors)} image errors; "
            f"see {output_dir / 'errors.jsonl'}",
            file=sys.stderr,
        )
        return 2
    return 0


def _stratified_items(
    items: list[InferenceItem],
    maximum: int | None,
    *,
    seed: int,
) -> list[InferenceItem]:
    if maximum is None or maximum >= len(items):
        return items
    if maximum < 2:
        raise ValueError("max-samples must be at least 2")
    positive = [item for item in items if item.true_label == 1]
    negative = [item for item in items if item.true_label == 0]
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    positive_count = max(1, round(maximum * len(positive) / len(items)))
    positive_count = min(positive_count, len(positive), maximum - 1)
    negative_count = min(maximum - positive_count, len(negative))
    selected = positive[:positive_count] + negative[:negative_count]
    if len(selected) < maximum:
        remaining = positive[positive_count:] + negative[negative_count:]
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)])
    rng.shuffle(selected)
    return selected


def _write_predictions(
    output_dir: Path,
    predictions: list[PredictionResult],
) -> None:
    jsonl_path = output_dir / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n"
            )
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page_id",
                "probability",
                "threshold",
                "predicted_label",
                "input_path",
                "true_label",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.to_dict())


def _write_errors(path: Path, errors: list[InferenceError]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in errors:
            handle.write(json.dumps(error.to_dict(), ensure_ascii=False) + "\n")


def _resolve_model(value: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, PROJECT_ROOT / path, PROJECT_ROOT.parent / path):
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve local base model: {value}")


def _cuda_gb(getter) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(getter() / 1024**3)


if __name__ == "__main__":
    raise SystemExit(main())
